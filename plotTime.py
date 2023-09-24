import numpy as np
from matplotlib import pyplot as plt

def plotTime(data, sample_rate = 128):

    N = data.size
    t = np.arange(N)
    t = t/sample_rate

    plt.figure()
    plt.plot(t, data)
    plt.title('Time domain Signal')
    plt.xlabel('Time in seconds')
    plt.ylabel('Voltage')
    plt.show()