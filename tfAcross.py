from dataPipeline import *
from subtractBaseAvg import *
import numpy as np
import tensorflow as tf
from VisionTransformer import *



subjects = loadData()

# Initialize an empty tensors
data_tensors = []
labels_tensors = []

for i in range(32):
    subject = subjects[i]
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


    data_tensor_temp = tf.constant(numpy_data, dtype=tf.float32)
    data_tensor_temp = tf.transpose(data_tensor_temp, [0, 2, 3, 1])
    labels_tensor_temp = tf.constant(numpy_labels, dtype=tf.float32)

    # Append the new data and labels to the respective lists
    data_tensors.append(data_tensor_temp)
    labels_tensors.append(labels_tensor_temp)


# Append the new data to the existing tensor
data_tensor = tf.concat(data_tensors, axis=0)
labels_tensor = tf.concat(labels_tensors, axis=0)


print(labels_tensor.shape)
print(data_tensor.shape)

X_train, X_test, y_train, y_test = split_data(data_tensor, labels_tensor, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)



model = create_VisionTransformer(1)

# Compile the model
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adjust learning rate as needed
    metrics=['accuracy', tf.keras.metrics.F1Score(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
model.summary()


# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=40)  # Adjust epochs and batch_size as needed

# Evaluate the model on the test data
test_loss, test_accuracy, f1, prec, recall = model.evaluate(X_test, y_test, verbose=1)

print("Test accuracy:", test_accuracy)
print("Test Loss:", test_loss)
print("Test F1:", f1)
print("Test precision:", prec)
print("Test Recall:", recall)
