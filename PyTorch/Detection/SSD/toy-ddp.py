import os
import sys
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import torchvision
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

#from torch.nn.parallel import DistributedDataParallel as DDP
# apex things
from apex.parallel.LARC import LARC
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *

def cleanup():
    dist.destroy_process_group()

class ToyNet(nn.Module):
    # toy model for regular MNIST dataset
    def __init__(self, num_classes=10):
        super(ToyNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class FashionNet(nn.Module):
    # model for the fashion mnist dataset
    # assuming input is (28, 28, 1)
    def __init__(self):
        super(FashionNet, self).__init__()
        self.layer1 = nn.Sequential(
            # shape should always be (batch_size, channel, height, width) until flatten
            nn.Conv2d(1, 32, kernel_size=3), # after conv2d, shape is (batch_size, 32, 26, 26)
            nn.ReLU(), # after relu, shape: (batch_size, 32, 26, 26)
            nn.MaxPool2d(kernel_size=2), # after maxpool2d, shape: (batch_size, 32, 13, 13)
            nn.Conv2d(32, 64, kernel_size=3), # after conv2d, shape: (batch_size, 64, 11, 11)
            nn.ReLU(), # after relu, shape: (batch_size, 64, 11, 11)
            nn.MaxPool2d(kernel_size=2), # after maxpool2d, shape: (batch_size, 64, 5, 5)
            nn.Flatten(),# after flatten: (batch_size, 1600)
            # input shape for linear needs to be 
            nn.Linear(1600, 64, bias=False), # after linear, shape: (batch_size, 64)
            nn.ReLU(), # after relu, shape: (batch_size, 64)
            nn.Linear(64, 10, bias=False) # output layer, after linear, shape: (batch_size, 10)
        )
    
    def forward(self, x):
        out = self.layer1(x)
        return out

def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())

def train(index, args):
    # multiprocessing stuff
    # get rank
    print("In train method!")
    rank = args.nr * args.gpus + index # rank of the worker
    drop_chance = args.drop_chance # drop chance
    runs = args.runs # number of runs to do
    hadamard = args.hadamard # whether or not to use the hadamard transform

    # setup data collection
    # aggregate training accuracy
    agg_acc_stats = [['']]
    for i in range(args.epochs):
        agg_acc_stats[0].append("Epoch " + str(i))

    agg_test_acc = [['']]
    agg_test_acc[0].append("Test Accuracy")

    print("Setting device")
    torch.cuda.set_device(0)

    print("Initializing process group")
    # change these
    dist.init_process_group(backend="nccl", init_method='env://')

    # start runs
    print("Starting runs!")
    for run in range(runs):
        print("Training model, run:", run)
        train_acc_time = [] # training accuracy over time
        test_acc_time = [] # test accuracy over time
        model = FashionNet() # using fashionMNIST model now
        model.cuda()
        batch_size = 128
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.012)

        # wrap the model
        print("Creating model and initializing AMP")
        model = DDP(model, hadamard=hadamard, drop_chance=drop_chance)

        # initialize AMP
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        # Data loading code
        # downloading fashion mnist now
        print("Downloading dataset")
        fashion_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                train=True,
                                                transform=transforms.ToTensor(),
                                                download=True)

        train_dataset, test_set = torch.utils.data.random_split(fashion_dataset, [50000, 10000])

        # train sampler for "sharding"
        # make sure each process gets a different piece of the data
        print("Creating sampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.nodes, rank=rank)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True,
                                                sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                batch_size=batch_size,
                                                shuffle=True)

        start = datetime.now()
        total_step = len(train_loader)
        #model.train()
        print("Starting epochs")
        for epoch in range(args.epochs):
            train_loss = 0.0
            train_acc = 0.0
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.cuda()
                labels = labels.cuda()
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                # using amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                # calculate accuracy
                train_loss += loss.item()
                train_acc += accuracy(outputs, labels)
                if (i + 1) % 5 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch + 1, 
                        args.epochs, 
                        i + 1, 
                        total_step,
                        loss.item())
                    )
            # print accuracy per epoch
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            train_acc_time.append(train_acc.item()) # append to train accuracy over time array
            print("Epoch {}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1, train_loss, train_acc))

        train_acc_time.insert(0, "Run:" + str(run))
        agg_acc_stats.append(train_acc_time)
        print("Training run", run, "complete in: " + str(datetime.now() - start))
        # run on test set if worker 0
        if rank == 0:
            model.eval()
            test_loss = 0.0
            test_acc = 0.0
            for i, (images, labels) in enumerate(test_loader):
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_acc += accuracy(outputs, labels)
            
            # print test loss and test accuracy
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            test_acc_time.append(test_acc.item())
            print("Test Loss: {:.3f}, Test Accuracy: {:.3f}".format(test_loss, test_acc))
        
            test_acc_time.insert(0, "Run:" + str(run))
            agg_test_acc.append(test_acc_time)

    if rank == 0:  
        # consolidate agg test acc to df
        agg_test_acc = np.array(agg_test_acc)
        df_test_acc = pd.DataFrame(data=agg_test_acc[1:, 1:], index=agg_test_acc[1:, 0], columns=agg_test_acc[0, 1:])
        _dir = os.path.dirname(__file__) # current directory
        test_acc_path = "data/results/%d-drop-test-acc-ddp.csv" % (int(drop_chance * 100))
        test_abs_path = os.path.join(_dir, test_acc_path)
        with open(test_abs_path, 'w') as f:
            df_test_acc.to_csv(f)

    # consolidate aggregated training accuracy stats to dataframe
    agg_acc_stats = np.array(agg_acc_stats)
    df_acc = pd.DataFrame(data=agg_acc_stats[1:, 1:], index=agg_acc_stats[1:, 0], columns=agg_acc_stats[0, 1:])
    # save data to files
    _dir = os.path.dirname(__file__) # current directory
    acc_fpath = "data/results/%d-drop-train-acc-ddp.csv" % (int(drop_chance * 100))
    acc_abspath = os.path.join(_dir, acc_fpath)
    with open(acc_abspath, 'w') as f:
        df_acc.to_csv(f)
        
def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def main():
    parser = argparse.ArgumentParser()
    # nodes are the number of machines/workers
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    # gpu is the number of gpus per machine to use
    # ignore GPU for now, since we're doing cpu distributed
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    # current rank of this node
    # 0 is the master process
    # goes from 0-args.nodes - 1
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    # number of runs to do
    parser.add_argument('-rn', '--runs', default=10, type=int, help='number of runs to do training')
    # drop chance
    parser.add_argument('-dc', '--drop_chance', default=0.0, type=float, help='drop chance for gradients')
    # epochs for training
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    # use hadamard transform or not
    parser.add_argument('-hd', '--hadamard', default=0, type=int, help='Use hadamard transform? 1 for yes, 0 for no (default is 0)')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes

    print("Running DDP!")

    # multi processing stuff
    os.environ['MASTER_ADDR'] = '172.31.18.167'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.nr)
    print(f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    print("Spawning nodes")
    #mp.spawn(train, nprocs=1, args=(args,))
    train(args.nr, args)

if __name__ == '__main__':
    main()