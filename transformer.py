import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


def train_model(model, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    all_true = []
    all_predicted = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            # Forward pass
            outputs = model(inputs)

            # Convert probabilities to predicted class (0 or 1)
            predicted = (outputs > 0.5).float()
            
            # Collect true and predicted labels
            all_true.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    # Convert lists to numpy arrays
    all_true = np.array(all_true)
    all_predicted = np.array(all_predicted)

    # Calculate F1 score, precision, and recall
    f1 = f1_score(all_true, all_predicted, average='weighted')
    precision = precision_score(all_true, all_predicted, average='weighted')
    recall = recall_score(all_true, all_predicted, average='weighted')

    return total_loss / len(val_loader), f1, precision, recall

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_ratio, dropout=0.1):
        super(TransformerLayer, self).__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)

        # Feedforward neural network
        ff_dim = int(embedding_dim * ff_ratio)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )

        # Layer normalization for the residual connection
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head self-attention
        attn_output, _ = self.self_attn(x, x, x)

        # Residual connection and layer normalization
        x = self.norm1(x + attn_output)

        # Feedforward neural network
        ff_output = self.feed_forward(x)

        # Residual connection and layer normalization
        x = self.norm2(x + ff_output)

        # Apply dropout
        x = self.dropout(x)

        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, embedding_dim_list, num_heads_list, ff_ratio_list, num_classes, dropout=0.1):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.transformer_layers = nn.ModuleList()

        for i in range(num_layers):
            layer = TransformerLayer(embedding_dim_list[i], num_heads_list[i], ff_ratio_list[i], dropout)
            self.transformer_layers.append(layer)

        self.fc = nn.Linear(embedding_dim_list[-1], num_classes)

    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x.squeeze()
        return x


def createTransformerPatches(data, patch_size = (60, 4096)):
    # data should be a tensor

    # Create patches using unfold
    patches = data.unfold(0, patch_size[0], patch_size[0]).unfold(1, patch_size[1], patch_size[1])
    
    patches = patches.squeeze()

    return patches



def flattenPatches(data):
    # Get the number of patches
    num_patches = data.shape[0]
    
    # Flatten each patch (60x4096) into a 1D vector
    flattened_patches = data.view(num_patches, -1)
    
    #print(flattened_patches.shape)
    # Flatten the resulting tensor into a final 1D tensor
    final_flattened_patches = flattened_patches.flatten()

    return flattened_patches


def toEmbeddings(flattened_patches, embedding_dim = 168):
    # Initialize the EmbeddingLayer
    
    input_dim = 245760
    embedding_layer = EmbeddingLayer(input_dim, embedding_dim)

    # Call the EmbeddingLayer
    embedded_patches = embedding_layer(flattened_patches)
    

    return embedded_patches



class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim = 245760, embedding_dim=168):
        #input_dim = 4096 * 60
        super(EmbeddingLayer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, embedding_dim)

    def forward(self, flattened_patches):
        # Assuming flattened_patches is a tensor of shape (batch_size, num_patches)
        embedded_patches = self.embedding(flattened_patches)
        return embedded_patches
