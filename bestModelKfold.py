from dataPipeline import *
from subtractBaseAvg import *
import numpy as np
import tensorflow as tf
from VisionTransformer import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import tensorflow_probability as tfp



subjects = loadData()

subject = subjects[0]
data_tensor, labels_tensor = dataSubjectPipeline(subject)

print(labels_tensor.shape)
print(data_tensor.shape)

X_train, X_test, y_train, y_test = split_data(data_tensor, labels_tensor, test_size=0.2, random_state=4)
print(X_train.shape)
print(X_test.shape)


# Set Hyperparameters
learning_rate = 0.00002
num_heads = 4
num_blocks = 12
projection_dim = 168
batch_size = 8
loss = tf.keras.losses.binary_crossentropy


# Define the full path to the model checkpoint file
checkpoint_path = "model_checkpoint.keras"

# Create a ModelCheckpoint callback to save the best model
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)

# Create an EarlyStopping callback to stop training if the validation loss doesn't improve for a certain number of epochs
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)  # You can change random_state as needed

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

    # Train your model on X_train and y_train
    # Fit your model, e.g., model.fit(X_train, y_train)
    model = create_model(learning_rate=learning_rate, num_heads=num_heads, num_blocks=num_blocks, projection_dim=projection_dim, batch_size=batch_size, loss=loss)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = batch_size, callbacks=[checkpoint, early_stopping], epochs=100)

    # Make predictions on X_test
    y_pred = model.predict(X_test)
    print("initial", y_pred)
    # Convert model predictions and labels to binary classes
    y_pred = (y_pred > 0.5)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    print("AFTER", y_pred)

    # Calculate and store accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracyMed", accuracy)
    accuracy_scores.append(accuracy)

    # Calculate and store F1 score for this fold
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

    # Store the true labels and predictions for total accuracy and F1
    total_predictions.extend(y_pred)
    true_labels.extend(y_test)

# Calculate the average accuracy and F1 score across all folds
average_accuracy = np.mean(accuracy_scores)
average_f1 = np.mean(f1_scores)

# Calculate total accuracy and F1
total_accuracy = accuracy_score(true_labels, total_predictions)
total_f1 = f1_score(true_labels, total_predictions)

print(f"Average Accuracy: {average_accuracy}")
print(f"Average F1 Score: {average_f1}")
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




# Evaluate the model on the test set
#test_accuracy = accuracy_score(y_test, model.predict(X_test).argmax(axis=1))

