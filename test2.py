import pickle as cPickle
import numpy as np
from matplotlib import pyplot as plt
from scipy import pi
from scipy.fftpack import fft
from subtractBaseAvg import*
from plotTime import *
from transformer import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from dataPipeline import *
from torchmetrics.functional import accuracy

subjects = loadData()

# Example usage
embedding_dim = 168  # Adjust as needed
num_heads = 4  # Adjust as needed
ff_dim = 6  # Adjust as needed
num_layers = 6
num_classes = 2
input_length = 40

model = Transformer(num_layers, embedding_dim, num_heads, ff_dim, num_classes)

# training
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# Training loop
num_epochs = 30  # Set the number of epochs you want


for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()
    
    for j in range(24):
        embeddedPatches, labels = dataPipeline(subjects[j])
        outputs = model(embeddedPatches)

        print(f"{j+1}th subject")
         # Compute loss
        loss = criterion(outputs, labels)  # Assuming labels is your 40x2 labels

        # Zero the gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        print(f"Loss: {loss:.20f}")
    # Process a single sample# Reshape the input to match the expected input shape of the model
    #outputs = model(embeddedPatches)
    
    #outputsTotal = outputsTotal.view(-1)
    #labelsTotal = labelsTotal.view(400,1)
    
    print(outputs.shape)
    print(labels.shape)
    print(outputs-labels)

    # Print the loss for this epoch
    
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item()}")


embeddedPatchesTest, labelsTest = dataPipeline(subjects[24])
# Set the model to evaluation mode (important for models with layers like dropout)
model.eval()

# Pass the validation dataset through the model
with torch.no_grad():
    predictions = model(embeddedPatchesTest)
predictions = (predictions >= 0.5).float()
print(predictions)
print(labelsTest)

# Assuming val_predictions is a tensor of predictions, compute accuracy as an example
arousal_accuracy = accuracy(predictions[:, 0], labelsTest[:, 0], num_classes=2, task='binary')
valence_accuracy = accuracy(predictions[:, 1], labelsTest[:, 1], num_classes=2, task='binary')

print("Arousal Accuracy:", arousal_accuracy)
print("Valence Accuracy:", valence_accuracy)