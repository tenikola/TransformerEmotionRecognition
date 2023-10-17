from dataPipeline import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

subjects = loadData()

subject = subjects[0]
# Split to data and labels
data = subject['data']
labels = subject['labels']

# get only eeg, ignore the peripheral signals
data = data[:, :32, :]

labels = labelsToBinary(labels)
labels = labels[:, 0]
data = subtractBaseAvg(data)

print(labels.shape)
print(data.shape)




# Define the Transformer model
class VisionTransformer(nn.Module):
    def __init__(self, num_patches=60, patch_embedding_size=168, num_transformer_layers=6):
        super(VisionTransformer, self).__init__()
        
        # Linear transformation for each patch
        self.patch_embedding = nn.Linear(32 * 128, patch_embedding_size)

        # Positional Encoding
        self.positional_encoding = self.positional_encoding(patch_embedding_size, num_patches)

        # Transformer Encoder
        #self.transformer = nn.TransformerEncoder(
         #   nn.TransformerEncoderLayer(
          #      d_model=patch_embedding_size,
           #     nhead=4,  # Number of attention heads
            #),
            #num_layers=num_transformer_layers
        #)
        # Define the transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(patch_embedding_size) for _ in range(num_transformer_layers)
        ])

        # Classification head
        self.classification_head = nn.Linear(num_patches * patch_embedding_size, 1)

    def forward(self, x):
        # Reshape the input to have patches as separate dimensions
        x = x.view(x.size(0), -1, 32 * 128)
        
        # Transform each patch into embeddings
        x = self.patch_embedding(x)

        # Add positional encodings
        x = x + self.positional_encoding

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Permute for transformer input
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, patch_embedding_size)

        # Pass the embeddings through the transformer
        #x = self.transformer(x)

        # Reshape and flatten the sequence of embeddings
        x = x.permute(1, 0, 2).contiguous()  # (batch_size, seq_length, patch_embedding_size)
        x = x.view(x.size(0), -1)

        # Classification
        #x = self.classification_head(x)
        x = torch.sigmoid(self.classification_head(x))

        return x
    

    def positional_encoding(self, d_model, n_position):
        position_enc = torch.zeros(n_position, d_model)
        position = torch.arange(0, n_position).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        position_enc[:, 0::2] = torch.sin(position.float() * div_term)
        position_enc[:, 1::2] = torch.cos(position.float() * div_term)
        return position_enc.unsqueeze(0)




class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, ff_dim=256):
        super(TransformerBlock, self).__init__()

        # Multi-Head Self Attention
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads)

        # Layer normalization and residual connection for MSA
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # MLP Block
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            #nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )

        # Layer normalization and residual connection for MLP
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Multi-Head Self Attention
        attn_output, _ = self.self_attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)

        # MLP Block
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)

        return x

# Create the dataset and dataloader
# Assuming you have 'data' (40, 32, 7680) and 'labels' (40,)
# Convert the Python array to a NumPy array
np_data = np.array(data)
np_labels = np.array(labels)

# Convert the NumPy array to a PyTorch tensor
data_tensor = torch.tensor(np_data).float() 
labels_tensor = torch.tensor(np_labels).view(-1, 1).float() 

# training
num_folds = 7
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!change to train data and labels !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
fold_splits = k_fold_split(data_tensor, labels_tensor, num_folds)



# Set the number of training epochs and batch size
num_epochs = 200
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

