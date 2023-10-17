from VisionTransformer import *
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

numpy_data = np.array(data)
numpy_labels = np.array(labels)


data_tensor = tf.constant(numpy_data, dtype=tf.float32)
data_tensor = tf.transpose(data_tensor, [0, 2, 3, 1])
labels_tensor = tf.constant(numpy_labels, dtype=tf.float32)


print(labels_tensor.shape)
print(data_tensor.shape)


batch = tf.expand_dims(data_tensor[0], axis=0)
patches = PatchExtractor()(batch)
print(patches.shape)

n = int(np.sqrt(patches.shape[1]))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (128, 1, 32))
    ax.imshow(patch_img.numpy().astype("uint8"))
    ax.axis("off")