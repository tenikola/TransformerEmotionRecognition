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
import torchmetrics


subjects = loadData()


# Example usage
embedding_dim_list = [168, 168, 168, 168, 168, 168, 168]  # Adjust as needed
num_heads_list = [2, 4, 2, 4, 4, 4, 4]  # Adjust as needed
mlp_ratio_list = [4.0, 3.5, 3.5, 4.0, 3.0, 3.5, 3.5]
#ff_dim = mlp_ratio*embedding_dim  # Adjust as needed
num_layers=len(embedding_dim_list)
num_classes = 1
input_length = 40


# Instantiate the transformer with different configurations for each layer
model = Transformer(num_layers, 
                          embedding_dim_list=embedding_dim_list, 
                          num_heads_list=num_heads_list, 
                          ff_ratio_list=mlp_ratio_list, 
                          num_classes=num_classes)  # Adjust num_classes as needed



# training
# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 400  # Set the number of epochs you want

embeddedPatches, labels = dataPipeline(subjects[0])
labels = labels.to(torch.float)

# Shuffle and split data to train and test
embeddedPatchesTrain, labelsTrain, embeddedPatchesTest, labelsTest = splitData(embeddedPatches, labels)


loss = 0
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()
    
    
    outputs = model(embeddedPatchesTrain)  
    outputs = outputs.to(torch.float)  # Ensure the output is in float format
    
    print(outputs-labelsTrain)
    #print(outputsTotal-labelsTotal)
    
    
    # Compute loss
    loss = criterion(outputs, labelsTrain)  # Assuming labels is your 40x2 labels

    # Zero the gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    # Print the loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item()}")

    #arousal_accuracy = torchmetrics.functional.classification.accuracy(outputs, labels, task = 'binary', threshold=0.5)

    #print("Arousal Accuracy:", arousal_accuracy)


# Set the model to evaluation mode (important for models with layers like dropout)
model.eval()

# Pass the validation dataset through the model
with torch.no_grad():
    predictions = model(embeddedPatchesTest)


print(predictions)
print(labelsTest)

# Assuming val_predictions is a tensor of predictions, compute accuracy as an example
arousal_accuracy = torchmetrics.functional.classification.accuracy(predictions, labelsTest, task = 'binary', threshold=0.5)

print("Arousal Accuracy:", arousal_accuracy)



