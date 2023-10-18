from subtractBaseAvg import*
from plotTime import *
from transformer import *
import torch
import torch.optim as optim
from dataPipeline import *
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import math
import sys
import matplotlib.pyplot as plt

# Load data as list of 40x32x128
subjects = loadData()


# Example usage
embedding_dim_list = [168, 168, 168, 168, 168, 168, 168]  # Adjust as needed
num_heads_list = [2, 4, 2, 4, 4, 4, 4]  # Adjust as needed
mlp_ratio_list = [4.0, 3.5, 3.5, 4.0, 3.0, 3.5, 3.5]
#ff_dim = mlp_ratio*embedding_dim  # Adjust as needed
num_layers=len(embedding_dim_list)
num_classes = 1


# Instantiate the transformer with different configurations for each layer
model = Transformer(num_layers, 
                          embedding_dim_list=embedding_dim_list, 
                          num_heads_list=num_heads_list, 
                          ff_ratio_list=mlp_ratio_list, 
                          num_classes=num_classes)  # Adjust num_classes as needed



# training
# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 400  # Set the number of epochs you want

embeddedPatches, labels = dataPipeline(subjects[2], label_type="valence")
labels = labels.to(torch.float)

# Shuffle and split data to train and test
embeddedPatchesTrain, labelsTrain, embeddedPatchesTest, labelsTest = splitData(embeddedPatches, labels)


loss = 0
for epoch in range(num_epochs):
    # Set the model to training mode
    model.eval()
    
    embeddedPatchesEpoch, labelsEpoch = shuffleTrain(embeddedPatchesTrain, labelsTrain, subset_size=20)
    
    outputs = model(embeddedPatchesEpoch)  
    outputs = outputs.to(torch.float)  # Ensure the output is in float format
    
    print(outputs-labelsEpoch)
    #print(outputsTotal-labelsTotal)
    
    
    # Compute loss
    loss = criterion(outputs, labelsEpoch)  # Assuming labels is your 40x2 labels

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

# Forward pass
with torch.no_grad():
    outputs = model(testData)
    outputs = outputs.view(-1)
    predictions = (outputs> 0.5).float()


# Convert tensors to numpy arrays for metric calculation
predictions = predictions.numpy()
labelsTest = testLabels.detach().numpy()
print(predictions)
print(labelsTest)
print(predictions-labelsTest)

# Calculate F1 score and accuracy
f1 = f1_score(labelsTest, predictions)
accuracy = accuracy_score(labelsTest, predictions)

# Baseline F1 and accuracy
random_predictions = np.random.randint(2, size=len(predictions))

# Convert integers to floats (0.0 or 1.0)
random_preds = random_predictions.astype(float)
base_f1 = f1_score(labelsTest, random_predictions)
base_accuracy = accuracy_score(labelsTest, random_predictions)


print(f'Test F1 Score: {f1}')
print(f'Test Accuracy: {accuracy}')

print(f'Test Base F1 Score: {base_f1}')
print(f'Test Base Accuracy: {base_accuracy}')

#Save the model
torch.save(model.state_dict(), 'model_s1_arousal.pth')
