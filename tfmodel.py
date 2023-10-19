from dataPipeline import *
from subtractBaseAvg import *
import numpy as np
import tensorflow as tf
from VisionTransformer import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model, save_model



subjects = loadData()

subject = subjects[0]
data_tensor, labels_tensor = dataSubjectPipeline(subject)

print(labels_tensor.shape)
print(data_tensor.shape)

X_train, X_test, y_train, y_test = split_data(data_tensor, labels_tensor, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

param_grid = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'num_heads': [2, 4, 6],
    'num_blocks': [6, 12],
    'projection_dim': [128, 240],
    'batch_size': [3, 6, 10]
}
#param_grid = {
 #   'learning_rate': [0.1, 0.01, 0.001, 0.0001],
  #  'num_heads': [2, 4, 6, 8],
   # 'num_blocks': [6, 12, 24],
    #'projection_dim': [128, 240, 512],
    #'batch_size': [3, 6, 10, 18]
#}

# Define the full path to the model checkpoint file
checkpoint_path = "model_checkpoint.keras"

# Create a ModelCheckpoint callback to save the best model
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)

# Create an EarlyStopping callback to stop training if the validation loss doesn't improve for a certain number of epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

#model = KerasClassifier(model=create_model(), epochs=100, batch_size=8, validation_split=0.2, callbacks=[checkpoint, early_stopping])
# Get the list of available parameters

num_folds = 10
# Loop through the grid search
best_model = None
best_accuracy = 0

for params in ParameterGrid(param_grid):
    model = create_model(**params)

    batch_size = params['batch_size']  # Extract batch_size from the params dictionary
    # Fit the model with checkpoint and early stopping
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = batch_size, callbacks=[checkpoint, early_stopping], epochs=100)

    # Evaluate the model on the test set
    test_accuracy = accuracy_score(y_test, model.predict(X_test).argmax(axis=1))

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        # Save the model to an HDF5 file
        model.save("best_model.keras")
        best_params = params

# Load the best model from the checkpoint
# Load your model with custom layer classes
best_model = tf.keras.models.load_model('best_model.keras', custom_objects={
    'PatchExtractor': PatchExtractor,
    'PatchEncoder': PatchEncoder,
    'MLP': MLP,
    'Block': Block,
    'TransformerEncoder': TransformerEncoder
})

print("Best Parameters: \n", best_params)
best_model.summary()
print("Best test accuracy:", best_accuracy)