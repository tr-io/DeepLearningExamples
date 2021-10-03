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
from src.distributed import DistributedDataParallel as DDP

"""
# Hadamard seed stuff
seed = 42
sgen = torch.Generator(device='cpu')
sgen.manual_seed(seed)

rgen = torch.Generator(device='cpu')
rgen.manual_seed(seed)
"""

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
    rank = args.nr # index of the worker
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

    torch.cuda.set_device(rank)

    dist.init_process_group(backend="gloo", world_size=args.nodes, rank=rank)

    # start runs
    for run in range(runs):
        print("Training model, run:", run)
        train_acc_time = [] # training accuracy over time
        test_acc_time = [] # test accuracy over time
        #torch.manual_seed(0) # set seed
        model = FashionNet() # using fashionMNIST model now
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set it to cpu
        model.to(device)
        batch_size = 128
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.012)

        # wrap the model
        model = DDP(model, hadamard, drop_chance)

        # Data loading code
        # downloading fashion mnist now
        fashion_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                train=True,
                                                transform=transforms.ToTensor(),
                                                download=True)

        train_dataset, test_set = torch.utils.data.random_split(fashion_dataset, [50000, 10000])

        # train sampler for "sharding"
        # make sure each process gets a different piece of the data
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
        model.train()
        for epoch in range(args.epochs):
            train_loss = 0.0
            train_acc = 0.0
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                # DEPRECATED: wrong way to do it
                """
                if hadamard == 1:
                    print("Using hadamard!")
                    
                    for p in model.parameters():
                        print(p.grad.size())
                        vec_shape = p.grad.size() # save gradient size so we can return it
                        print("Initial grad vec: ", p.grad)
                        vec = torch.flatten(p.grad) # flatten to 1 dimension before applying Hadamard transform
                        print("Flattened vec: ", vec)
                        dim = len(vec)
                        h_vec = random_hadamard_encode(vec, dim, prng=sgen)
                        print("Transformed vec: ", h_vec)
                        # drop individual elements in gradient
                        # individual 'axes' or 'dimensions', so to speak
                        ndropped = int(np.round(drop_chance * vec.numel()))
                        dropped_idx = torch.randperm(dim)[:ndropped]
                        h_vec[dropped_idx] = 0
                        # inverse hadamard transform
                        decompressed_vec = random_hadamard_decode(h_vec, dim, prng=rgen, frac=dim/(dim-ndropped))
                        print("restored vec: ", decompressed_vec)
                        print(decompressed_vec.size())
                        # now reshape and save back to grad
                        reshaped_vec = torch.reshape(decompressed_vec, vec_shape)
                        p.grad = reshaped_vec
                else:
                    num_params = sum(1 for p in model.parameters()) # get the number of weights
                    r_vec = np.random.choice([0, 1], size=num_params, p=[drop_chance, 1-drop_chance]) # create random drop vector

                    # loop through model parameters (weights) and change gradients
                    j = 0 # index for iterating through r_vec
                    for p in model.parameters():
                        if r_vec[j] == 0:
                            p.grad *= 0 # drop entire gradient
                        j += 1 # increment iterator
                """
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
                images = images.to(device)
                labels = labels.to(device)
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

    # multi processing stuff
    os.environ['MASTER_ADDR'] = '172.31.18.167'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(train, nprocs=1, args=(args,))

if __name__ == '__main__':
    main()