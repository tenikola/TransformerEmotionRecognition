import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import KFold
import torch.nn.functional as F
import math

def k_fold_split(data, labels, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold_splits = []
    
    for train_idx, test_idx in kf.split(data):
        train_data, val_data = data[train_idx], data[test_idx]
        train_labels, val_labels = labels[train_idx], labels[test_idx]
        fold_splits.append((train_data, val_data, train_labels, val_labels))
    
    return fold_splits



# Define the training function
def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        target = target.long()
        loss = nn.functional.nll_loss(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()


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
    #flattened_patches = data.view(num_patches, -1)
    flattened_patches = data.reshape(num_patches, -1)
    
    #print(flattened_patches.shape)
    # Flatten the resulting tensor into a final 1D tensor
    final_flattened_patches = flattened_patches.flatten()

    return flattened_patches


def toEmbeddings(flattened_patches, embedding_dim = 168):
    # Initialize the EmbeddingLayer
    
    # 245760 = 60x128x32
    # 307200 = 60x128x40
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




class VisionTransformer(nn.Module):
    def __init__(self, num_patches=60, patch_embedding_size=240, num_transformer_layers=6):
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
            nn.ReLU(),
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

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))