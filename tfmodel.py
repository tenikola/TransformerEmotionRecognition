from dataPipeline import *
from subtractBaseAvg import *
import numpy as np
import tensorflow as tf
from VisionTransformer import *



subjects = loadData()

subject = subjects[0]
# Split to data and labels
data = subject['data']
labels = subject['labels']

# get only eeg, ignore the peripheral signals
data = data[:, :32, :]

labels = labelsToBinary(labels)
labels = labels[:, 0]
data = subtractBaseAvg(data)
data = reshapeInput2(data)

print(labels.shape)
print(data.shape)

numpy_data = np.array(data)
numpy_labels = np.array(labels)


data_tensor = tf.constant(numpy_data, dtype=tf.float32)
data_tensor = tf.transpose(data_tensor, [0, 2, 3, 1])
labels_tensor = tf.constant(numpy_labels, dtype=tf.float32)


print(labels_tensor.shape)
print(data_tensor.shape)

X_train, X_test, y_train, y_test = split_data(data_tensor, labels_tensor, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)



model = create_VisionTransformer(1)
# Compile the model
model.compile(
    loss='hinge',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adjust learning rate as needed
    metrics=['accuracy']
)
model.summary()


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=6)  # Adjust epochs and batch_size as needed

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print("Test accuracy:", test_accuracy)
print("Test Loss:", test_loss)
