import os
import sys
import sqlite3
import json

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

        # metadata params
        self.model_name = "Toy Model"
        self.dataset = "Fashion MNIST"
    
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
    gpu = args.local_rank
    drop_chance = args.drop_chance # drop chance
    runs = args.runs # number of runs to do
    hadamard = args.hadamard # whether or not to use the hadamard transform
    node_rank = args.nr
    tail = args.tail_drops
    comm_backend = args.comm
    setup = args.setup
    # set metadata
    model_name = ""
    dataset = ""

    # setup data collection
    # aggregate training accuracy
    agg_acc_stats = [['']]
    for i in range(args.epochs):
        agg_acc_stats[0].append("Epoch " + str(i))

    agg_test_acc = [['']]
    agg_test_acc[0].append("Test Accuracy")

    print("Setting device")
    torch.cuda.set_device(gpu)

    print("Initializing process group")
    # change these
    dist.init_process_group(backend=comm_backend, init_method='env://')

    print("Getting world size")
    args.world_size = torch.distributed.get_world_size()

    # start runs
    print("Starting runs!")
    for run in range(runs):
        print("Training model, run:", run)
        train_acc_time = [] # training accuracy over time
        test_acc_time = [] # test accuracy over time
        model = FashionNet() # using fashionMNIST model now

        # set metadata
        dataset = model.dataset
        model_name = model.model_name

        model = model.cuda()
        batch_size = 128
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.012)

        # initialize AMP
        #print("Initializing AMP")
        #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        # wrap the model
        print("Creating model")
        model = DDP(model, hadamard=hadamard, drop_chance=drop_chance, rseed=node_rank, tail=tail, delay_allreduce=True)

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
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.nodes)

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
                loss.backward()
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
        if node_rank == 0:
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

    # consolidate aggregated training accuracy stats to dataframe
    agg_acc_stats = np.array(agg_acc_stats)
    df_acc = pd.DataFrame(data=agg_acc_stats[1:, 1:], index=agg_acc_stats[1:, 0], columns=agg_acc_stats[0, 1:])
    train_row = df_acc.iloc[0]
    train_acc_json = train_row.to_json(orient="columns")

    # store into database
    # db: ml-multicast.db
    # schema: acc_experiments
    drop_method = 0
    if tail == 1:
        drop_method = 1

    if node_rank == 0:
        # consolidate agg test acc to df
        agg_test_acc = np.array(agg_test_acc)
        df_test_acc = pd.DataFrame(data=agg_test_acc[1:, 1:], index=agg_test_acc[1:, 0], columns=agg_test_acc[0, 1:])
        final_acc = df_test_acc.iloc[0]["Test Accuracy"]
        con = sqlite3.connect('/home/ml-multicast.db')
        cur = con.cursor()
        cur.execute(f"INSERT INTO acc_experiments VALUES ('{model_name}', '{dataset}', '{setup}', '{comm_backend}', 'Ring', {args.nodes}, {args.epochs}, {drop_chance}, {drop_method}, {hadamard}, '', '{train_acc_json}', '', {float(final_acc)})")
        con.commit()
        con.close()
    """
    if node_rank == 0:  
        # consolidate agg test acc to df
        agg_test_acc = np.array(agg_test_acc)
        df_test_acc = pd.DataFrame(data=agg_test_acc[1:, 1:], index=agg_test_acc[1:, 0], columns=agg_test_acc[0, 1:])
        _dir = os.path.dirname(__file__) # current directory
        if hadamard == 1:
            test_acc_path = "/data/%d-drop-hd-tail-test-acc.csv" % (int(drop_chance * 100))
        else:
            if tail == 1:
                test_acc_path = "/data/%d-drop-nohd-tail-test-acc.csv" % (int(drop_chance * 100))
            else:
                test_acc_path = "/data/%d-drop-nohd-rand-test-acc.csv" % (int(drop_chance * 100))
        with open(test_acc_path, 'w') as f:
            df_test_acc.to_csv(f)
    """

    # consolidate aggregated training accuracy stats to dataframe
    """
    agg_acc_stats = np.array(agg_acc_stats)
    df_acc = pd.DataFrame(data=agg_acc_stats[1:, 1:], index=agg_acc_stats[1:, 0], columns=agg_acc_stats[0, 1:])
    # save data to files
    _dir = os.path.dirname(__file__) # current directory
    if hadamard == 1:
        acc_fpath = "/data/%d-drop-hd-tail-train-acc.csv" % (int(drop_chance * 100))
    else:
        if tail == 1:
            acc_fpath = "/data/%d-drop-nohd-tail-train-acc.csv" % (int(drop_chance * 100))
        else:
            acc_fpath = "/data/%d-drop-nohd-rand-train-acc.csv" % (int(drop_chance * 100))
    with open(acc_fpath, 'w') as f:
        df_acc.to_csv(f)
    """

def main():
    parser = argparse.ArgumentParser()
    # local rank added by pytorch launcher
    parser.add_argument("--local_rank", default=0, type=int)
    # nodes are the number of machines/workers
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    # current rank of this node
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    # number of runs to do
    parser.add_argument('-rn', '--runs', default=10, type=int, help='number of runs to do training')
    # drop chance
    parser.add_argument('-dc', '--drop_chance', default=0.0, type=float, help='drop chance for gradients')
    # tail drops
    parser.add_argument('-td', '--tail_drops', default=0, type=int, help='do tail drops or no')
    # epochs for training
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    # use hadamard transform or not
    parser.add_argument('-hd', '--hadamard', default=0, type=int, help='Use hadamard transform? 1 for yes, 0 for no (default is 0)')
    
    # metadata parameters
    parser.add_argument('-sp', '--setup', default="AWS", type=str, help='Experiment hardware setup')
    parser.add_argument('-cm', '--comm', default="nccl", type=str, help='Distributed learning communication backend')

    args = parser.parse_args()

    print("Running DDP!")
    print(f"Node rank: {args.nr}")
    print(f"Local rank: {args.local_rank}")

    # multi processing stuff
    os.environ['MASTER_ADDR'] = '172.31.18.167'
    os.environ['MASTER_PORT'] = '12355'
    #os.environ['NCCL_SOCKET_IFNAME'] = 'ens5'
    #os.environ['RANK'] = str(args.local_rank)
    os.environ['NCCL_DEBUG'] = 'INFO'
    train(args.local_rank, args)

if __name__ == '__main__':
    main()