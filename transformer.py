import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        
        # Feedforward neural network
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
    def __init__(self, num_layers, embedding_dim, num_heads, ff_dim, num_classes, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(embedding_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        # Global average pooling
        #x = x.mean(dim=1)
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


def embeddingLayer(flattened_patches, embedding_dim = 168):
    # Initialize the EmbeddingLayer
    
    input_dim = 245760
    embedding_layer = EmbeddingLayer(input_dim, embedding_dim)

    # Call the EmbeddingLayer
    embedded_patches = embedding_layer(flattened_patches)
    

    return embedded_patches



class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim = 245760, embedding_dim=168):
        super(EmbeddingLayer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, embedding_dim)

    def forward(self, flattened_patches):
        # Assuming flattened_patches is a tensor of shape (batch_size, num_patches)
        embedded_patches = self.embedding(flattened_patches)
        return embedded_patches
