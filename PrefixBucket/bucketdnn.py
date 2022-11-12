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
import random

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

error = 1.1

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(int(sys.argv[5]))

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

train_dataset = CliqueBucketDataset(str(sys.argv[6]),
        INPUT_SIZE, SEQ_LENGTH)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True,
        num_workers = 2)

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
model = Model(input_size=INPUT_SIZE, output_size=1, hidden_dim=HIDDEN_DIM)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define hyperparameters
n_epochs = int(sys.argv[7])
lr=float(sys.argv[8])
momen = 0.9

# Define Loss, Optimizer, MLP
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Training Started")

test_dataset = CliqueBucketDataset(str(sys.argv[9]),
        INPUT_SIZE, SEQ_LENGTH)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False,
        num_workers = 1)

def predict(model, counts):
    out = model(counts)
    return out

# Training Run
training_losses = []
validation_losses = []
for epoch in range(1, n_epochs + 1):
    for j, data in enumerate(train_loader):
        model.zero_grad()
        logits = model(data['train'].cuda().float())

        loss = criterion(logits.flatten(), data['target'].cuda().float().flatten())
        loss.backward() # Does backpropagation and calculates gradients

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1000)
        optimizer.step() # Updates the weights accordingly

    # Perform validation
    for k, data_test in enumerate(test_loader):
        with torch.no_grad():
            counts = predict(model, data_test["train"].cuda().float())
            val_loss = criterion(counts.flatten(), data_test['target'].cuda().float().flatten())


    training_losses.append(loss.item())
    validation_losses.append(val_loss.item())
    if epoch % 1 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Training Loss: {:.4f}".format(loss.item()))
        print("Validation Loss: {:.4f}".format(val_loss.item()))

plt.plot(training_losses, label="training loss")
plt.plot(validation_losses, label = "validation loss")
plt.savefig("Losses")

torch.save(model, str(sys.argv[6]) + ".pt")

counts_file = str(sys.argv[9])
true_counts = read_counts(counts_file)
def main_count(model, test_loader):
    model.eval() # eval mode
    triangle_counts = []

    true_counts = read_counts(counts_file)[1:]
    for j, data in enumerate(test_loader):
        counts = predict(model, data["train"].cuda().float())
        triangle_counts.append(counts.cpu().detach().numpy().flatten())

    tc = []
    for sublist in triangle_counts:
        for t in sublist:
            tc.append(t)
    triangle_counts = tc

    if not len(true_counts) == len(triangle_counts):
        print("ERROR: counts not equal", len(true_counts), len(triangle_counts))

    print(triangle_counts, true_counts)

    errors = []
    for i in range(len(true_counts)):
        if (float(true_counts[i]) > 0):
            error = abs(1 - (float(triangle_counts[i]))/(float(true_counts[i])))
            errors.append(error)
    return triangle_counts, errors

counts, errors = main_count(model, test_loader)
mean_error = statistics.mean(errors)
std_error = statistics.stdev(errors)
print("Mean and standard deviation: ", mean_error, std_error)
print(counts)
print(true_counts)
print(errors)
