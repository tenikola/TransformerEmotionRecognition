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

def dataPipeline(x, label_type = "arousal"):

    # Split to data and labels
    data = x['data']
    labels = x['labels']

    # get only eeg, ignore the peripheral signals
    data = data[:, :32, :]

    labels = labelsToBinary(labels)
    data = subtractBaseAvg(data)

    data = reshapeInput(data)

    data = divideToFlatten2Dpatches(data)

    # Convert your data and labels to PyTorch tensors
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    #labels = labels.to(torch.float)

    patches = createTransformerPatches(data)
    #print(patches.shape)

    flattened_patches = flattenPatches(patches)
    #print(flattened_patches.shape)

    embeddedPatches = embeddingLayer(flattened_patches)
    #print(embeddedPatches.shape)

    if label_type == "arousal":
        labels = labels[:, 0]
    elif label_type == "valence":
        labels = labels[:, 1]
    else:
        print("Wrong label_type input")
        KeyError
    return embeddedPatches, labels


def splitData(embeddedPatches, labels):

    # Combine embeddings and labels
    combined_data = torch.cat((embeddedPatches, labels.unsqueeze(1)), dim=1)

    print(combined_data.shape)
    # Shuffle the combined data randomly
    shuffled_data = shuffle(combined_data, random_state=42)  # Set an appropriate random_state for reproducibility

    # Separate the shuffled data back into embeddings and labels
    shuffled_embeddings = shuffled_data[:, :-1]  # Exclude the last column (labels)
    shuffled_labels = shuffled_data[:, -1]  # Only the last column (labels)

    # Split the shuffled data and labels into training and testing sets
    embeddedPatchesTrain = shuffled_embeddings[:30]  # First 30 rows for training data
    labelsTrain = shuffled_labels[:30]  # First 30 labels for training

    embeddedPatchesTest = shuffled_embeddings[30:]  # Last 10 rows for testing data
    labelsTest = shuffled_labels[30:]  # Last 10 labels for testing

    return embeddedPatchesTrain, labelsTrain, embeddedPatchesTest, labelsTest

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