import torch
import statistics
import matplotlib.pyplot as plt
from torch import nn
from helper import *

from sklearn.preprocessing import StandardScaler

import os
import sys
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

maxbucketlen = int(sys.argv[1])

# Number of features, equal to number of buckets
INPUT_SIZE = maxbucketlen

# Number of previous time steps taken into account
SEQ_LENGTH = 2

# Number of stacked rnn layers
NUM_LAYERS = int(sys.argv[2])

# We have a set of 10 training inputs divided into sequences of length 2 to get
BATCH_SIZE = int(sys.argv[3])

# Output Size
OUTPUT_SIZE = maxbucketlen

# Number of hidden units
HIDDEN_DIM = int(sys.argv[4])

error = float(sys.argv[9])

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class CliqueBucketDataset(Dataset):
    def __init__(self, csv_file, input_length, seq_length):
        self.buckets_frame = pd.read_csv(csv_file, delim_whitespace = True)
        self.seq_length = seq_length
        self.input_length = input_length

    def __len__(self):
        return len(self.buckets_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        train = self.buckets_frame.iloc[idx, :self.input_length]
        train = np.array([train])

        target = self.buckets_frame.iloc[idx, self.input_length:]
        target = np.array([target])

        # Below can be used to reshape data to sequence data
        train = train.astype('float').reshape(-1, self.input_length)
        target = target.astype('float').reshape(-1, 1)

        sample = {'train': train, 'target': target}

        return sample

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim

        #Defining the layers
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.flatten(x)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        logits = self.linear_relu_stack(x)

        return logits

# Instantiate the model with hyperparameters
model = torch.load(str(sys.argv[5]))
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)
results_file = open(str(sys.argv[14]), "a")

def predict(model, counts):
    out = model(counts)
    return out

def main_count(model):
    model.eval() # eval mode

    directory = sys.argv[6]
    for filename in os.listdir(directory):
        if "_dedup" in filename and not "_train" in filename:
            f = os.path.join(directory, filename)
            m = {}
            print("Processing " + filename)

            data_file = open(f, 'r')
            lines = data_file.readlines()

            true_counts = read_counts(f)
            triangle_counts = []

            test_dataset = CliqueBucketDataset(f, INPUT_SIZE, SEQ_LENGTH)
            test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False,
                num_workers = 1)

            true_counts = true_counts[1:]
            for j, data in enumerate(test_loader):
                counts = predict(model, data["train"].cuda().float())
                triangle_counts.append(counts.cpu().detach().numpy().flatten())


            tc = []
            for sublist in triangle_counts:
                for t in sublist:
                    tc.append(t)
            if not len(true_counts) == len(tc):
                print("ERROR: counts not equal", len(true_counts), len(tc))
            triangle_counts = tc
            errors = []
            for i in range(len(true_counts)):
                if (float(true_counts[i]) > 0):
                    error = abs(1 - (float(triangle_counts[i]))/(float(true_counts[i])))
                    errors.append(error)
            mean_error = statistics.mean(errors)
            std_error = statistics.stdev(errors)
            print("Mean and standard deviation: ", mean_error, std_error)
            print(counts)
            print(true_counts)
            print(errors)
            results_file.write(filename + ", " + str(mean_error) + ", " + str(std_error) + ", " +
                    str(max(errors)) + ", " + str(min(errors)) + ", " + ", ".join([str(x) for x in errors]) + "\n")

main_count(model)
results_file.close()
