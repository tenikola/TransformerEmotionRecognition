import numpy as np


# Finds the baseline average, subtracts it and then removes it
def subtractBaseAvg(data, Hz = 128, baseline_seconds = 3):
    num_of_baseline_data = Hz * baseline_seconds
    
    # calculate the mean for each trail and each channel,
    # so we calculate 40x32 differents means
    baseline_average = np.mean(data[:, :, :num_of_baseline_data], axis=2, keepdims=True)

    # subtract the average from the data
    data = data - baseline_average

    # remove the baseline data completely
    data = data[:, :, num_of_baseline_data:]

    return data


# converts the labels from 1-9 to high and low values
def labelsToBinary(labels):
    #keep only valence and arousal
    labels = labels[:, [0, 2]]

    # If value is >5 is set to 1, otherwise is set to zero
    threshold_value = 5
    boolean_labels = (labels > threshold_value).astype(int)
    return boolean_labels


# reshapes the input from 40x32x7680, to 2400x128x32
def reshapeInput(data, trials = 40, channels = 32, seconds = 60, Hz = 128):
    
    # Assuming eeg_data is your initial EEG data with shape (40, 32, 7680)
    initial_shape = data.shape

    # middle shape is splitting to seconds, to 40x32x60x128
    middle_shape = (trials, channels, seconds, Hz)

    # Check if the shapes are compatible
    assert np.prod(initial_shape) == np.prod(middle_shape), "Shapes are incompatible"

    # Reshape to the final shape
    reshaped_data = data.reshape(middle_shape)
    
    final_data = reshaped_data.reshape(-1, reshaped_data.shape[3], reshaped_data.shape[1])
    # Shape of final_data is (2400, 128, 32)
    return final_data



# divides the 2400x128x32 input into a sequence of flatten 2D patches
def divideToFlatten2Dpatches(data, p1 = 128, p2 = 1):
    
    # Step 1: Reshape to Patches
    num_patches = int(data.shape[0]*data.shape[1]/(p1*p2))  # Number of patches
    patches = data.reshape(num_patches, p1, p2, data.shape[2])

    # Step 2: Flatten Patches and Sequence
    flattened_patches = patches.reshape(num_patches, -1)  # Flatten the patches
    final_data = flattened_patches  # Input sequence for the ViT

    # Now, input_sequence has the shape (2400, P1*P2*32), which is the input for the ViT
    return final_data



def reshapeInput2(data, trials = 40, channels = 32, seconds = 60, Hz = 128):
    
    # Assuming eeg_data is your initial EEG data with shape (40, 32, 7680)
    initial_shape = data.shape

    # middle shape is splitting to seconds, to 40x32x60x128
    middle_shape = (trials, channels, seconds, Hz)

    # Check if the shapes are compatible
    assert np.prod(initial_shape) == np.prod(middle_shape), "Shapes are incompatible"

    # Reshape to the final shape
    reshaped_data = data.reshape(middle_shape)
    
    return reshaped_data
