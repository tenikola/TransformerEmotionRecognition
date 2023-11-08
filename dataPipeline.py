import pickle as cPickle
import numpy as np
from matplotlib import pyplot as plt
from scipy import pi
from scipy.fftpack import fft
from subtractBaseAvg import*
from plotTime import *
from transformer import *
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


def ConcatSubjectsToTensor(subjects):
    Patches = torch.empty(0)  # Initialize as an empty tensor
    labels = torch.empty(0)  # Initialize as an empty tensor

    for i in range(32):
        tempPatch, tempLabel = dataPipeline(subjects[i], label_type="arousal")
        print(tempPatch.shape)
        tempLabel = tempLabel.to(torch.float)

        # Append the tensors to embeddedPatches and labels
        Patches = torch.cat((Patches, tempPatch), dim=0)
        labels = torch.cat((labels, tempLabel), dim=0)
        print(i)


    print(Patches.shape)
    print(labels.shape)
    return Patches, labels


def dataPipeline(x, label_type = "arousal"):

    # Split to data and labels
    data = x['data']
    labels = x['labels']

    # get only eeg, ignore the peripheral signals
    data = data[:, :32, :]

    labels = labelsToBinary(labels)
    data = subtractBaseAvg(data)

    data = reshapeInput(data, channels=32)

    data = divideToFlatten2Dpatches(data)

    # Convert your data and labels to PyTorch tensors
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    #labels = labels.to(torch.float)

    patches = createTransformerPatches(data)
    print(f'shape of patches: {patches.shape}')

    flattened_patches = flattenPatches(patches)
    print(f'shape of flattened patches: {flattened_patches.shape}')

    if label_type == "arousal":
        labels = labels[:, 0]
    elif label_type == "valence":
        labels = labels[:, 1]
    else:
        print("Wrong label_type input")
        KeyError
    return flattened_patches, labels







def splitData(embeddedPatches, labels, trainP = 0.7, valP = 0.2, testP = 0.1):

    # Combine embeddings and labels
    combined_data = torch.cat((embeddedPatches, labels.unsqueeze(1)), dim=1)

    print(combined_data.shape)
    # Shuffle the combined data randomly
    shuffled_data = shuffle(combined_data, random_state=42)  # Set an appropriate random_state for reproducibility

    # Separate the shuffled data back into embeddings and labels
    shuffled_embeddings = shuffled_data[:, :-1]  # Exclude the last column (labels)
    shuffled_labels = shuffled_data[:, -1]  # Only the last column (labels)

    num_of_train = int(embeddedPatches.shape[0] * trainP)
    num_of_val = int(embeddedPatches.shape[0]*valP)
    num_of_test = embeddedPatches.shape[0]-num_of_train-num_of_val

    index1 = num_of_train
    index2 = num_of_train + num_of_val

    # Split the shuffled data and labels into training and testing sets
    embeddedPatchesTrain = shuffled_embeddings[:index1]  # First 30 rows for training data
    labelsTrain = shuffled_labels[:index1]  # First 30 labels for training

    embeddedPatchesVal = shuffled_embeddings[index1:index2]  # First 30 rows for training data
    labelsVal = shuffled_labels[index1:index2]  # First 30 labels for training

    embeddedPatchesTest = shuffled_embeddings[index2:]  # Last 10 rows for testing data
    labelsTest = shuffled_labels[index2:]  # Last 10 labels for testing

    return embeddedPatchesTrain, labelsTrain, embeddedPatchesVal, labelsVal, embeddedPatchesTest, labelsTest


def shuffleTrain(data, labels, subset_size=10):
    # Ensure data and labels have the same number of samples
    assert len(data) == len(labels), "Data and labels must have the same length"

    # Combine data and labels
    combined = list(zip(data, labels))

    # Shuffle the combined data and labels
    np.random.shuffle(combined)

    # Unzip the shuffled data and labels
    shuffled_data, shuffled_labels = zip(*combined)

    # Convert to tensors
    shuffled_data = torch.stack(shuffled_data[:subset_size])
    shuffled_labels = torch.stack(shuffled_labels[:subset_size])

    return shuffled_data, shuffled_labels


def loadData():
    subjects = [dict() for i in range(32)]
    # Import s01.dat from DEAP
    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s01.dat', 'rb') as f:
        subjects[0] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s02.dat', 'rb') as f:
        subjects[1] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s03.dat', 'rb') as f:
        subjects[2] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s04.dat', 'rb') as f:
        subjects[3] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s05.dat', 'rb') as f:
        subjects[4] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s06.dat', 'rb') as f:
        subjects[5] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s07.dat', 'rb') as f:
        subjects[6] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s08.dat', 'rb') as f:
        subjects[7] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s09.dat', 'rb') as f:
        subjects[8] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s10.dat', 'rb') as f:
        subjects[9] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s11.dat', 'rb') as f:
        subjects[10] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s12.dat', 'rb') as f:
        subjects[11] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s13.dat', 'rb') as f:
        subjects[12] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s14.dat', 'rb') as f:
        subjects[13] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s15.dat', 'rb') as f:
        subjects[14] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s16.dat', 'rb') as f:
        subjects[15] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s17.dat', 'rb') as f:
        subjects[16] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s18.dat', 'rb') as f:
        subjects[17] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s19.dat', 'rb') as f:
        subjects[18] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s20.dat', 'rb') as f:
        subjects[19] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s21.dat', 'rb') as f:
        subjects[20] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s22.dat', 'rb') as f:
        subjects[21] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s23.dat', 'rb') as f:
        subjects[22] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s24.dat', 'rb') as f:
        subjects[23] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s25.dat', 'rb') as f:
        subjects[24] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s26.dat', 'rb') as f:
        subjects[25] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s27.dat', 'rb') as f:
        subjects[26] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s28.dat', 'rb') as f:
        subjects[27] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s29.dat', 'rb') as f:
        subjects[28] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s30.dat', 'rb') as f:
        subjects[29] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s31.dat', 'rb') as f:
        subjects[30] = cPickle.load(f, encoding='latin1')

    with open(r'C:\Users\nickt\OneDrive\data_preprocessed_python\s32.dat', 'rb') as f:
        subjects[31] = cPickle.load(f, encoding='latin1')

    return subjects



def ConcatSubjectsToTensor2(subjects):
    Patches = torch.empty(0)  # Initialize as an empty tensor
    labels = torch.empty(0)  # Initialize as an empty tensor

    for i in range(32):
        tempPatch, tempLabel = dataPipeline2(subjects[i], label_type="arousal")
        print(tempPatch.shape)
        tempLabel = tempLabel.to(torch.float)

        # Append the tensors to embeddedPatches and labels
        Patches = torch.cat((Patches, tempPatch), dim=0)
        labels = torch.cat((labels, tempLabel), dim=0)
        print(i)


    print(Patches.shape)
    print(labels.shape)
    return Patches, labels


def dataPipeline2(x, label_type = "arousal"):

    # Split to data and labels
    data = x['data']
    labels = x['labels']

    # get only eeg, ignore the peripheral signals
    #data = data[:, :32, :]

    labels = labelsToBinary(labels)
    data = subtractBaseAvg(data)

    data = reshapeInput2(data, channels=40)

    # Convert your data and labels to PyTorch tensors
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)


    if label_type == "arousal":
        labels = labels[:, 0]
    elif label_type == "valence":
        labels = labels[:, 1]
    else:
        print("Wrong label_type input")
        KeyError
    return data, labels


def splitData2(data, labels, train_percentage = 0.875):
    """
    Split data and labels into training and testing sets while maintaining the correspondence between data and labels.

    Parameters:
    data (numpy.ndarray): The data tensor of shape (1280, 32, 60, 128).
    labels (numpy.ndarray): The labels tensor of shape (1280,).
    train_percentage (float): The percentage of data to be used for training.

    Returns:
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: 
    A tuple containing the training data, testing data, training labels, and testing labels.
    """
    # Ensure the data and labels have the same number of samples
    assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples"

    # Reshape the data to match with labels
    data_reshaped = data.reshape(data.shape[0], -1)  # Shape: (1280, 32*60*128)

    # Shuffle the data and labels while maintaining the correspondence
    indices = np.random.permutation(data_reshaped.shape[0])
    shuffled_data = data_reshaped[indices].reshape(data.shape)  # Reshape back to original shape
    shuffled_labels = labels[indices]

    # Calculate the number of samples for training
    num_train_samples = int(train_percentage * data_reshaped.shape[0])

    # Split the shuffled data and labels into training and testing sets
    train_data = shuffled_data[:num_train_samples]
    test_data = shuffled_data[num_train_samples:]
    train_labels = shuffled_labels[:num_train_samples]
    test_labels = shuffled_labels[num_train_samples:]

    return train_data, test_data, train_labels, test_labels




def ConcatSubjectsToTensor3(subjects):
    Patches = torch.empty(0)  # Initialize as an empty tensor
    labels = torch.empty(0)  # Initialize as an empty tensor

    tempPatch, tempLabel = dataPipeline2(subjects, label_type="arousal")
    print(tempPatch.shape)
    tempLabel = tempLabel.to(torch.float)

    # Append the tensors to embeddedPatches and labels
    Patches = tempPatch
    labels = tempLabel


    print(Patches.shape)
    print(labels.shape)
    return Patches, labels



def loadDataDrive():
    subjects = [dict() for i in range(32)]
    # Import s01.dat from DEAP
    with open(r"/content/drive/MyDrive/data_preprocessed_python/s01.dat", 'rb') as f:
        subjects[0] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s02.dat", 'rb') as f:
        subjects[1] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s03.dat", 'rb') as f:
        subjects[2] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s04.dat", 'rb') as f:
        subjects[3] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s05.dat", 'rb') as f:
        subjects[4] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s06.dat", 'rb') as f:
        subjects[5] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s07.dat", 'rb') as f:
        subjects[6] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s08.dat", 'rb') as f:
        subjects[7] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s09.dat", 'rb') as f:
        subjects[8] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s10.dat", 'rb') as f:
        subjects[9] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s11.dat", 'rb') as f:
        subjects[10] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s12.dat", 'rb') as f:
        subjects[11] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s13.dat", 'rb') as f:
        subjects[12] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s14.dat", 'rb') as f:
        subjects[13] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s15.dat", 'rb') as f:
        subjects[14] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s16.dat", 'rb') as f:
        subjects[15] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s17.dat", 'rb') as f:
        subjects[16] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s18.dat", 'rb') as f:
        subjects[17] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s19.dat", 'rb') as f:
        subjects[18] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s20.dat", 'rb') as f:
        subjects[19] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s21.dat", 'rb') as f:
        subjects[20] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s22.dat", 'rb') as f:
        subjects[21] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s23.dat", 'rb') as f:
        subjects[22] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s24.dat", 'rb') as f:
        subjects[23] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s25.dat", 'rb') as f:
        subjects[24] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s26.dat", 'rb') as f:
        subjects[25] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s27.dat", 'rb') as f:
        subjects[26] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s28.dat", 'rb') as f:
        subjects[27] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s29.dat", 'rb') as f:
        subjects[28] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s30.dat", 'rb') as f:
        subjects[29] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s31.dat", 'rb') as f:
        subjects[30] = cPickle.load(f, encoding='latin1')

    with open(r"/content/drive/MyDrive/data_preprocessed_python/s32.dat", 'rb') as f:
        subjects[31] = cPickle.load(f, encoding='latin1')

    return subjects


def loadDataJupyter():
    subjects = [dict() for i in range(32)]
    # Import s01.dat from DEAP
    with open(r'/home/t/tenikola/data_preprocessed_python/s01.dat', 'rb') as f:
        subjects[0] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s02.dat', 'rb') as f:
        subjects[1] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s03.dat', 'rb') as f:
        subjects[2] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s04.dat', 'rb') as f:
        subjects[3] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s05.dat', 'rb') as f:
        subjects[4] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s06.dat', 'rb') as f:
        subjects[5] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s07.dat', 'rb') as f:
        subjects[6] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s08.dat', 'rb') as f:
        subjects[7] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s09.dat', 'rb') as f:
        subjects[8] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s10.dat', 'rb') as f:
        subjects[9] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s11.dat', 'rb') as f:
        subjects[10] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s12.dat', 'rb') as f:
        subjects[11] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s13.dat', 'rb') as f:
        subjects[12] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s14.dat', 'rb') as f:
        subjects[13] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s15.dat', 'rb') as f:
        subjects[14] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s16.dat', 'rb') as f:
        subjects[15] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s17.dat', 'rb') as f:
        subjects[16] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s18.dat', 'rb') as f:
        subjects[17] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s19.dat', 'rb') as f:
        subjects[18] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s20.dat', 'rb') as f:
        subjects[19] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s21.dat', 'rb') as f:
        subjects[20] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s22.dat', 'rb') as f:
        subjects[21] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s23.dat', 'rb') as f:
        subjects[22] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s24.dat', 'rb') as f:
        subjects[23] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s25.dat', 'rb') as f:
        subjects[24] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s26.dat', 'rb') as f:
        subjects[25] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s27.dat', 'rb') as f:
        subjects[26] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s28.dat', 'rb') as f:
        subjects[27] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s29.dat', 'rb') as f:
        subjects[28] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s30.dat', 'rb') as f:
        subjects[29] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s31.dat', 'rb') as f:
        subjects[30] = cPickle.load(f, encoding='latin1')

    with open(r'/home/t/tenikola/data_preprocessed_python/s32.dat', 'rb') as f:
        subjects[31] = cPickle.load(f, encoding='latin1')

    return subjects
