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

images, labels = dataPipeline2(subjects[9])

trainData, testData, trainLabels, testLabels = splitData2(images, labels, 1.0)


print(trainData.shape)
print(testData.shape)
print(trainLabels.shape)
print(testLabels.shape)


#model = VisionTransformer()


# training
num_folds = 7
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!change to train data and labels !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
fold_splits = k_fold_split(trainData, trainLabels, num_folds)



# Set the number of training epochs and batch size
# Set the number of training epochs and batch size
num_epochs = 20
batch_size = 5
early_stop_thresh = 5


# Lists to store F1 scores and loss_scores for each fold
accuracy_scores = []
f1_scores = []
val_loss_scores = []
baseline_f1_scores = []


# Loop through each fold in the cross-validation splits
for fold, (train_data, val_data, train_labels, val_labels) in enumerate(fold_splits):
    print(train_data.shape)

    average_loss_scores = []

    model = VisionTransformer()
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    
    
    best_loss = 100
    best_epoch = -1
    # Loop through each epoch for training
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        #print(f'Fold: {fold+1}, Epoch: {epoch+1}')

        train_loss_scores = []
        # Training loop
        for i in range(0, len(train_data), batch_size):
            # Get a batch of data and labels
            batch_data, batch_labels = train_data[i:i+batch_size], train_labels[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_data)
            outputs = outputs.view(-1)
            batch_labels = batch_labels.view(-1)
            #outputs = (outputs> 0.5).float()
            
            loss = criterion(outputs, batch_labels)

            train_loss_scores.append(loss)

            batch_num = int(i/batch_size)
            #print(f'Fold {fold+1}, Epoch {epoch+1}, Batch {batch_num+1}: Training Loss: {loss}')
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        average_train_epoch_loss = sum(train_loss_scores)/len(train_loss_scores)
        average_loss_scores.append(average_train_epoch_loss)

        print(f'Epoch {epoch+1}, Average Train Loss: {average_train_epoch_loss}, Best Loss: {best_loss}')
        
        if average_train_epoch_loss < best_loss:
            best_loss = average_train_epoch_loss
            best_epoch = epoch
            #checkpoint(model, "best_model.pth")
        elif epoch - best_epoch > early_stop_thresh:
            print(f"Early stopped training at epoch {epoch+1}")
            break  # terminate the training loop

    #resume(model, "best_model.pth")

    # Create the plot
    # Convert the list to a PyTorch tensor
    average_loss_scores_tensor = torch.tensor(average_loss_scores)

    # Convert the PyTorch tensor to a NumPy array
    average_loss_scores_np = average_loss_scores_tensor.detach().numpy()

    # Generate x values (assuming equally spaced data points)
    x_values = range(len(average_loss_scores_np))

    #plt.plot(x_values, average_loss_scores_np, label='Average epoch loss for this fold')

    # Add labels and title
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.title('Plotting Average Loss')

    # Display a legend
    #plt.legend()

    # Show the plot
    #plt.show()

    #Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():     
        # Forward pass
        outputs = model(val_data)
        outputs = outputs.view(-1)
        val_labels = val_labels.view(-1)
        val_loss = criterion(outputs, val_labels)

        # Baseline f1
        # Generate random integers (0 or 1)
        random_preds = np.random.randint(2, size=len(outputs))

        # Convert integers to floats (0.0 or 1.0)
        random_preds = random_preds.astype(float)
        base_f1 = f1_score(val_labels, random_preds)
        baseline_f1_scores.append(base_f1)
        
        print(val_labels)
        print(outputs)

        # Convert to Binary predictions
        predicted = (outputs> 0.5).float()
        print(val_labels)
        print(predicted)

        # Calculate F1 score
        f1 = f1_score(val_labels, predicted)
        accuracy = accuracy_score(val_labels, predicted)

        #append scores to list, so we can calculate average
        f1_scores.append(f1)
        val_loss_scores.append(val_loss)
        accuracy_scores.append(accuracy)

    # Print validation results for this epoch
    print(f'Fold {fold+1}: Validation Loss: {val_loss}, Accuracy: {accuracy}, F1 Score: {f1}, Baseline F1 Score: {base_f1}')

average_f1 = sum(f1_scores)/len(f1_scores)
average_val_loss = sum(val_loss_scores)/len(val_loss_scores)
average_base_f1 = sum(baseline_f1_scores)/len(baseline_f1_scores)
average_accuracy = sum(accuracy_scores)/len(accuracy_scores)
print(f'Average Validation Loss: {average_val_loss}, AverageAccuracy Score: {average_accuracy}, AverageF1 Score: {average_f1}, Average BaselineF1 Score: {average_base_f1}')



### TESTING ###

# Set the model to evaluation mode
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
