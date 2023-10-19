import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, GlobalAveragePooling1D, Input, Layer, LayerNormalization, MultiHeadAttention
from dataPipeline import *
from subtractBaseAvg import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers


def create_model(learning_rate=0.001, num_heads=4, num_blocks=6, projection_dim=240, batch_size=6):
    model = create_VisionTransformer(1, num_heads=num_heads, num_blocks=num_blocks, projection_dim=projection_dim)  # Your model creation function
    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        metrics=['accuracy', tf.keras.metrics.F1Score(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    model.summary()
    return model


def dataSubjectPipeline(subject):
    # Split to data and labels
    data = subject['data']
    labels = subject['labels']

    # get only eeg, ignore the peripheral signals
    data = data[:, :32, :]

    labels = labelsToBinary(labels)
    labels = labels[:, 0]
    data = subtractBaseAvg(data)
    data = reshapeInput2(data)

    numpy_data = np.array(data)
    numpy_labels = np.array(labels)

    data_tensor = tf.constant(numpy_data, dtype=tf.float32)
    data_tensor = tf.transpose(data_tensor, [0, 2, 3, 1])
    labels_tensor = tf.constant(numpy_labels, dtype=tf.float32)

    return data_tensor, labels_tensor



def split_data(data, labels, test_size=0.2, random_state=42):
    """
    Split the data and labels into training and testing sets.

    Parameters:
    data (tensorflow.Tensor): The data tensor of shape (num_samples, height, width, channels).
    labels (tensorflow.Tensor): The labels tensor of shape (num_samples,).
    test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
    random_state (int, optional): Seed for the random number generator. Defaults to 42.

    Returns:
    tuple: (X_train, X_test, y_train, y_test) containing training and testing data and labels.
    """
    # Reshape the data to (num_samples, height, width*channels) for compatibility with train_test_split
    data_reshaped = tf.reshape(data, (data.shape[0], data.shape[1], -1))

    # Convert labels to numpy array for compatibility with train_test_split
    labels_np = labels.numpy()

    # Split the data and labels into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_reshaped.numpy(), labels_np, test_size=test_size, random_state=random_state, shuffle=True)

    # Reshape the data back to its original shape
    X_train = tf.reshape(X_train, (X_train.shape[0], data.shape[1], -1, data.shape[-1]))
    X_test = tf.reshape(X_test, (X_test.shape[0], data.shape[1], -1, data.shape[-1]))

    return X_train, X_test, tf.constant(y_train), tf.constant(y_test)


class PatchExtractor(Layer):
    def __init__(self):
        super(PatchExtractor, self).__init__()

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, 128, 1, 1],
            strides=[1, 128, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(Layer):
    def __init__(self, num_patches=60, projection_dim=4096, linear_projection_dim=240):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        #self.projection_dim = projection_dim

        self.linear_projection_dim = linear_projection_dim  # Added linear_projection_dim

        w_init = tf.random_normal_initializer()

        #class_token = w_init(shape=(1, projection_dim), dtype="float32")
        class_token = w_init(shape=(1, self.linear_projection_dim), dtype="float32")  # Changed projection_dim to linear_projection_dim

        self.class_token = tf.Variable(initial_value=class_token, trainable=True)

        #self.projection = Dense(units=projection_dim)
        #self.position_embedding = Embedding(input_dim=num_patches+1, output_dim=projection_dim)
        self.projection = Dense(units=self.linear_projection_dim)  # Changed projection_dim to linear_projection_dim
        self.position_embedding = Embedding(input_dim=num_patches + 1, output_dim=self.linear_projection_dim)  # Changed projection_dim to linear_projection_dim

    def call(self, patch):
        batch = tf.shape(patch)[0]
        # reshape the class token embedins
        class_token = tf.tile(self.class_token, multiples = [batch, 1])

        #class_token = tf.reshape(class_token, (batch, 1, self.projection_dim))
        class_token = tf.reshape(class_token, (batch, 1, self.linear_projection_dim))  # Changed projection_dim to linear_projection_dim

        # calculate patches embeddings
        patches_embed = self.projection(patch)
        patches_embed = tf.concat([patches_embed, class_token], 1)
        # calcualte positional embeddings
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)
        positions_embed = self.position_embedding(positions)

        # add both embeddings
        print(patches_embed.shape)
        print(positions_embed.shape)
        encoded = patches_embed + positions_embed
        return encoded

class MLP(Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.dense1 = Dense(hidden_features, activation=tf.nn.gelu)
        self.dense2 = Dense(out_features)
        self.dropout = Dropout(dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y = self.dropout(x)
        return y
    

class Block(Layer):
    def __init__(self, projection_dim, num_heads=4, dropout_rate=0.1):
        super(Block, self).__init__()
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(projection_dim * 2, projection_dim, dropout_rate)
        #self.mlp = MLP(512, projection_dim, dropout_rate)

    def call(self, x):
        # Layer normalization 1.
        x1 = self.norm1(x) # encoded_patches
        # Create a multi-head attention layer.
        attention_output = self.attn(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, x]) #encoded_patches
        # Layer normalization 2.
        x3 = self.norm2(x2)
        # MLP.
        x3 = self.mlp(x3)
        # Skip connection 2.
        y = Add()([x3, x2])
        return y
    
class TransformerEncoder(Layer):
    def __init__(self, projection_dim, num_heads=4, num_blocks=12, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.blocks = [Block(projection_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.5)

    def call(self, x):
        # Create a [batch_size, projection_dim] tensor.
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        y = self.dropout(x)
        return y
    
def create_VisionTransformer(num_classes, num_heads=4, num_blocks=12, num_patches=60, projection_dim=240, input_shape=(128, 60, 32)):
    inputs = Input(shape=input_shape)
    # Patch extractor
    patches = PatchExtractor()(inputs)
    # Patch encoder
    patches_embed = PatchEncoder(num_patches, linear_projection_dim=projection_dim)(patches)
    # Transformer encoder
    representation = TransformerEncoder(projection_dim = projection_dim, num_heads=num_heads, num_blocks=num_blocks)(patches_embed)
    representation = GlobalAveragePooling1D()(representation)
    # MLP to classify outputs
    logits = MLP(projection_dim, num_classes, 0.5)(representation)
    # Create model
    model = Model(inputs=inputs, outputs=logits)
    return model
