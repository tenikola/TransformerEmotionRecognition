from transformer import *


embedding_dim_list = [168, 168, 168, 168, 168, 168, 168]  # Adjust as needed
num_heads_list = [2, 4, 2, 4, 4, 4, 4]  # Adjust as needed
mlp_ratio_list = [4.0, 3.5, 3.5, 4.0, 3.0, 3.5, 3.5]
#ff_dim = mlp_ratio*embedding_dim  # Adjust as needed
num_layers=len(embedding_dim_list)
num_classes = 1

model = Transformer(num_layers, embedding_dim_list, num_heads_list, mlp_ratio_list, num_classes)

model.load_state_dict(torch.load('model_s1_arousal.pth'))