import pickle as cPickle
import numpy as np
from matplotlib import pyplot as plt
from scipy import pi
from scipy.fftpack import fft
from timeToFreq import *
from plotFreq import *
from transformer import *
import torch.optim as optim
from torch.utils.data import DataLoader



# Import s01.dat from DEAP
with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s01.dat', 'rb') as f:
    x = cPickle.load(f, encoding='latin1')

# Split to data and labels
# and convert to tensors
data_np = x['data']
labels_np = x['labels']

# convert to tensors
data_tensor = torch.from_numpy(data_np)
labels_tensor = torch.from_numpy(labels_np)
print(data_tensor.shape)
print(labels_tensor.shape)

# Get time series from 1st video and 1st channel
p1 = data_tensor[0, :, :]
# Get time series from 1st video and 40th channel
p2 = data_tensor[1, :, :]

#other way to make them tensors
#data_tensor = torch.tensor(data_np, dtype=torch.float32)
#print(data_tensor.shape)

#freq_data = timeToFreq(data_np)

#plotFreq(freq_data)

input_shape = (40, 40, 8064)  # Desired input shape
embedding_dim = 168  # Adjust the embedding dimension as needed
hidden_size = 240  # Hidden size for the transformer
num_layers = 4  # Number of transformer layers
num_heads = 4  # Number of attention heads
dropout_prob = 0.1  # Dropout probability
num_classes = 2  # Number of output classes (1 to 10 labels)




