from subtractBaseAvg import*
from plotTime import *
from transformer import *
import torch
import torch.optim as optim
from dataPipeline import *
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score

# Load data as list of 40x32x128
subjects = loadData()

# Create flattened patches and convert to tensors of 1280x245760 embeddings and 1280 labels
flattenedPatches, labels = ConcatSubjectsToTensor(subjects)

# Create embeddings
embeddedPatches =toEmbeddings(flattenedPatches, embedding_dim=240)
# embeddedPatches = flattenedPatches


embeddedPatchesTrain, labelsTrain, embeddedPatchesVal, labelsVal, embeddedPatchesTest, labelsTest = splitData(embeddedPatches, labels, trainP=0.875, valP=0.0,  testP=0.125)

print(embeddedPatchesTrain.shape)
print(embeddedPatchesVal.shape)
print(embeddedPatchesTest.shape)


num_folds = 7
fold_splits = k_fold_split(embeddedPatchesTrain, labelsTrain, num_folds)


### CREATE MODEL ###
# embedding_dim_list = [168, 168, 168, 168, 168, 168, 168]  # Adjust as needed
embedding_dim_list = [240] * 7
#num_heads_list = [2, 4, 2, 4, 4, 4, 4]  # Adjust as needed
num_heads_list = [6] * 7
#mlp_ratio_list = [4.0, 3.5, 3.5, 4.0, 3.0, 3.5, 3.5]
mlp_ratio_list = [4.0] * 7
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

# Set the number of training epochs and batch size
num_epochs = 100
batch_size = 40
early_stop_thresh = 5


# Lists to store F1 scores and loss_scores for each fold
f1_scores = []
val_loss_scores = []

# Loop through each fold in the cross-validation splits
for fold, (train_data, val_data, train_labels, val_labels) in enumerate(fold_splits):
    
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
        val_loss = criterion(outputs, val_labels)
                
        # Convert to Binary predictions
        predicted = (torch.sigmoid(outputs) > 0.5).float()

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

### TESTING ###

# Set the model to evaluation mode
model.eval()

# Forward pass
with torch.no_grad():
    outputs = model(embeddedPatchesTest)
    predictions = (torch.sigmoid(outputs) > 0.5).float()


# Convert tensors to numpy arrays for metric calculation
predictions = predictions.numpy()
labelsTest = labelsTest.detach().numpy()

# Calculate F1 score and accuracy
f1 = f1_score(labelsTest, predictions)
accuracy = accuracy_score(labelsTest, predictions)

print(f'Test F1 Score: {f1}')
print(f'Test Accuracy: {accuracy}')



#Save the model
torch.save(model.state_dict(), 'model_across_arousal.pth')

