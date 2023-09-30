from transformer import *


### LOAD MODEL ###
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

model = Transformer(num_layers, embedding_dim_list, num_heads_list, mlp_ratio_list, num_classes)

model.load_state_dict(torch.load('model_across_arousal.pth'))