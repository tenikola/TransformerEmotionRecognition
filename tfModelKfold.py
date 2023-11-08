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
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import load_model, save_model



subjects = loadData()

subject = subjects[0]
data_tensor, labels_tensor = dataSubjectPipeline(subject)

print(labels_tensor.shape)
print(data_tensor.shape)


param_grid = {
    'learning_rate': [0.00002],
    'num_heads': [4],
    'num_blocks': [12],
    'projection_dim': [168],
    'batch_size': [8],
    'loss': [tf.keras.losses.binary_crossentropy],
    'optimizer': [tf.keras.optimizers.RMSprop, tf.keras.optimizers.Adagrad, tf.keras.optimizers.Adadelta, tf.keras.optimizers.Ftrl]
}
#param_grid = {
 #   'learning_rate': [0.1, 0.01, 0.001, 0.0001],
  #  'num_heads': [2, 4, 6, 8],
   # 'num_blocks': [6, 12, 24],
    #'projection_dim': [128, 240, 512],
    #'batch_size': [3, 6, 10, 18],
    #'loss': [tf.keras.losses.binary_crossentropy, tf.keras.losses.mse, tf.keras.losses.hinge, tf.keras.losses.binary_focal_crossentropy],
    #'optimizer': [tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop, tf.keras.optimizers.Adagrad, tf.keras.optimizers.Adadelta, tf.keras.optimizers.Nadam, tf.keras.optimizers.Ftrl]
#}

# Define the full path to the model checkpoint file
checkpoint_path = "model_checkpoint.keras"

# Create a ModelCheckpoint callback to save the best model
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)

# Create an EarlyStopping callback to stop training if the validation loss doesn't improve for a certain number of epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

#model = KerasClassifier(model=create_model(), epochs=100, batch_size=8, validation_split=0.2, callbacks=[checkpoint, early_stopping])
# Get the list of available parameters

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)  # You can change random_state as needed
# Loop through the grid search
best_model = None
best_accuracy = 0
best_f1 = 0

for params in ParameterGrid(param_grid):
    # Initialize lists to store accuracy and F1 scores for each fold
    accuracy_scores = []
    f1_scores = []

    # Initialize lists to store predictions and true labels for total accuracy and F1
    total_predictions = []
    true_labels = []

    for train_index, test_index in kf.split(data_tensor):
        #X_train, X_test = data_tensor[train_index], data_tensor[test_index]
        #y_train, y_test = labels_tensor[train_index], labels_tensor[test_index]
        # Use TensorFlow's gather method to get the selected indices
        X_train, X_test = tf.gather(data_tensor, train_index), tf.gather(data_tensor, test_index)
        y_train, y_test = tf.gather(labels_tensor, train_index), tf.gather(labels_tensor, test_index)

        model = create_model(**params)

        batch_size = params['batch_size']  # Extract batch_size from the params dictionary

        # Fit the model with checkpoint and early stopping
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = batch_size, callbacks=[checkpoint, early_stopping], epochs=100)

        # Make predictions on X_test
        y_pred = model.predict(X_test)

        y_pred = (y_pred > 0.5)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        # Store the true labels and predictions for total accuracy and F1
        total_predictions.extend(y_pred)
        true_labels.extend(y_test)      

    #total_predictions = (total_predictions > 0.5)
    #total_predictions = tf.cast(total_predictions, dtype=tf.float32)

    # Calculate total accuracy and F1
    total_accuracy = accuracy_score(true_labels, total_predictions)
    total_f1 = f1_score(true_labels, total_predictions)

    print(f"Total Accuracy: {total_accuracy}")
    print(f"Total F1 Score: {total_f1}")

    # Baseline predictions
    # Get the length of your_tensor
    n = tf.shape(true_labels)[0]

    # Generate a tensor of random 0s and 1s of the same length as your_tensor
    baseline_predictions = tf.random.uniform(shape=(n,), minval=0, maxval=2, dtype=tf.int32)
    baseline_predictions = tf.cast(baseline_predictions, tf.float32)

    # Calculate baseline accuracy and F1
    baseline_accuracy = accuracy_score(true_labels, baseline_predictions)
    baseline_f1 = f1_score(true_labels, baseline_predictions)

    print(f"Baseline Accuracy: {baseline_accuracy}")
    print(f"Baseline F1 Score: {baseline_f1}")

    if total_accuracy > best_accuracy:
        best_accuracy = total_accuracy
        # Save the model to an HDF5 file
        model.save("best_model_ac.keras")
        best_params_ac = params

    if total_f1 > best_f1:
        best_f1 = total_f1
        # Save the model to an HDF5 file
        model.save("best_model_f1.keras")
        best_params_f1 = params

# Load the best model from the checkpoint
# Load your model with custom layer classes
best_model_ac = tf.keras.models.load_model('best_model_ac.keras', custom_objects={
    'PatchExtractor': PatchExtractor,
    'PatchEncoder': PatchEncoder,
    'MLP': MLP,
    'Block': Block,
    'TransformerEncoder': TransformerEncoder
})

print("Best Parameters for Accuracy: \n", best_params_ac)
best_model_ac.summary()
print("Best test accuracy:", best_accuracy)


# Load the best model from the checkpoint
# Load your model with custom layer classes
best_model_f1 = tf.keras.models.load_model('best_model_f1.keras', custom_objects={
    'PatchExtractor': PatchExtractor,
    'PatchEncoder': PatchEncoder,
    'MLP': MLP,
    'Block': Block,
    'TransformerEncoder': TransformerEncoder
})

print("Best Parameters for F1: \n", best_params_f1)
best_model_f1.summary()
print("Best test f1:", best_f1)