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

# Create flattened patches and convert to tensors of 1280x245760 embeddings and 1280 labels
flattenedPatches, labels = ConcatSubjectsToTensor(subjects)

# Create embeddings
embeddedPatches =toEmbeddings(flattenedPatches, embedding_dim=240)
# embeddedPatches = flattenedPatches


embeddedPatchesTrain, labelsTrain, embeddedPatchesVal, labelsVal, embeddedPatchesTest, labelsTest = splitData(embeddedPatches, labels)

print(embeddedPatchesTrain.shape)
print(embeddedPatchesVal.shape)
print(embeddedPatchesTest.shape)


# Create custom datasets
train_dataset = CustomDataset(embeddedPatchesTrain, labelsTrain)
val_dataset = CustomDataset(embeddedPatchesVal, labelsVal)
test_dataset = CustomDataset(embeddedPatchesTest, labelsTest)

# Create data loaders
batch_size = 128  # Adjust batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(train_loader)


# embedding_dim_list = [168, 168, 168, 168, 168, 168, 168]  # Adjust as needed
embedding_dim_list = [240] * 7
#num_heads_list = [2, 4, 2, 4, 4, 4, 4]  # Adjust as needed
num_heads_list = [8] * 7
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
optimizer = optim.Adam(model.parameters(), lr=0.001, eps = 0.0001)

num_epochs = 10


# Placeholder for average validation loss and accuracy
average_val_loss = 0.0
average_val_f1 = 0.0
average_val_precision = 0.0
average_val_recall = 0.0

# Perform 10-fold cross-validation
num_folds = 10
# for fold in range(num_folds):
for fold, (data, labels) in enumerate(train_loader):
    print(f"Fold {fold + 1}/{num_folds}")
    train_loader_temp = CustomDataset(data, labels)

    # Train the model
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader_temp, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Evaluate on validation set
    val_loss, val_f1, val_precision, val_recall = evaluate_model(model, val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, Validation f1: {val_f1:.2%}")
    print(f"Validation precision: {val_precision:.4f}, Validation recall: {val_recall:.2%}")
    # Accumulate validation metrics for averaging later
    average_val_loss += val_loss
    average_val_f1 += val_f1
    average_val_precision += val_precision
    average_val_recall += val_recall
# Average validation metrics over all folds
average_val_loss /= num_folds
average_val_f1 /= num_folds
average_val_precision /= num_folds
average_val_recall /= num_folds

print("Average Validation Loss:", average_val_loss)
print("Average Validation f1:", average_val_f1)
print("Average Validation precision:", average_val_precision)
print("Average Validation recall:", average_val_recall)
