import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as numpy

"""
def viewer(orig, decode):
    fig = plt.figure(figsize= (10,10))
    for i in range(20):
        plt.subplot(2,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if i+1 <= 10:
            plt.imshow(orig[i], cmap=plt.cm.binary)
        else:
            plt.imshow(decode[i], cmap=plt.cm.binary)
    plt.show()
"""


def viewer(orig, decode):
    fig = plt.figure()
    for i in range(orig.shape[0]):
        ax1 = fig.add_subplot(1, 10, i+1)
        ax2 = fig.add_subplot(2, 10, i+1)

        ax2.imshow(orig[i], cmap=plt.cm.binary)
        ax1.imshow(decode[i], cmap=plt.cm.binary)

    plt.show()



def convert_to_fullyconntected(x_data):
    return x_data.reshape(len(x_data), 784)
    
def rebuild_image(decoded_data):
    return decoded_data.reshape(len(decoded_data), 28, 28)
