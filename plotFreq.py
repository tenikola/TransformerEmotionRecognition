import numpy as np
from matplotlib import pyplot as plt


def plotFreq(freq_data, N = 8064):

    y = 2/N * np.abs (freq_data [0, 0, 0:int(N/2)])
    frequency = np.linspace (0.0, 64, int(N/2))

    plt.figure()
    plt.plot(frequency, y)
    plt.title('Frequency domain Signal')
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Amplitude')
    plt.show()