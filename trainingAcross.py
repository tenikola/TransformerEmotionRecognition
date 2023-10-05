from subtractBaseAvg import*
from plotTime import *
from transformer import *
import torch
import torch.optim as optim
from dataPipeline import *
import torchmetrics
from torch.utils.data import Dataset, DataLoader

# Load data as list of 40x32x128
subjects = loadData()

images, labels = ConcatSubjectsToTensor2(subjects)

trainData, testData, trainLabels, testLabels = splitData2(images, labels)

print(trainData.shape)
print(testData.shape)
print(trainLabels.shape)
print(testLabels.shape)


model = VisionTransformer()


# training
num_folds = 7
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!change to train data and labels !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
fold_splits = k_fold_split(testData, testLabels, num_folds)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Set the number of training epochs and batch size
num_epochs = 100
batch_size = 40
early_stop_thresh = 5


# Lists to store F1 scores and loss_scores for each fold
f1_scores = []
val_loss_scores = []

# Loop through each fold in the cross-validation splits
for fold, (train_data, val_data, train_labels, val_labels) in enumerate(fold_splits):
    print(train_data.shape)
    
    best_loss = 100
    best_epoch = -1
    # Loop through each epoch for training
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        print(f'Fold: {fold+1}, Epoch: {epoch+1}')

        train_loss_scores = []
        # Training loop
        for i in range(0, len(train_data), batch_size):
            # Get a batch of data and labels
            batch_data, batch_labels = train_data[i:i+batch_size], train_labels[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_data)
            outputs = outputs.view(-1)
            
            loss = criterion(outputs, batch_labels)

            train_loss_scores.append(loss)

            batch_num = int(i/batch_size)
            print(f'Fold {fold+1}, Epoch {epoch+1}, Batch {batch_num+1}: Training Loss: {loss}')
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        average_train_epoch_loss = sum(train_loss_scores)/len(train_loss_scores)
        
        print(f'Epoch {epoch+1}, Average Train Loss: {average_train_epoch_loss}, Best Loss: {best_loss}')
        
        if average_train_epoch_loss < best_loss:
            best_loss = average_train_epoch_loss
            best_epoch = epoch
            checkpoint(model, "best_model.pth")
        elif epoch - best_epoch > early_stop_thresh:
            print(f"Early stopped training at epoch {epoch+1}")
            break  # terminate the training loop

    resume(model, "best_model.pth")


    #Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():     
        # Forward pass
        outputs = model(val_data)
        outputs = outputs.view(-1)
        val_loss = criterion(outputs, val_labels)
                
        # Convert to Binary predictions
        predicted = (outputs> 0.5).float()

        # Calculate F1 score
        f1 = f1_score(val_labels, predicted)

        #append scores to list, so we can calculate average
        f1_scores.append(f1)
        val_loss_scores.append(val_loss)

    # Print validation results for this epoch
    print(f'Fold {fold+1}: Validation Loss: {val_loss}, F1 Score: {f1}')

average_f1 = sum(f1_scores)/len(f1_scores)
average_val_loss = sum(val_loss_scores)/len(val_loss_scores)
print(f'Average Validation Loss: {average_val_loss}, AverageF1 Score: {average_f1}')