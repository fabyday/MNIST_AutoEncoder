import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as numpy


def viewer(orig, decode):
    plt.figure()
    for i in range(24):
        plt.subplot(6,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(orig[i], cmp=plt.cm.binary)
        plt.imshow(decode[i], cmp=plt.cm.binary)

    plt.show()


def convert_to_fullyconntected(x_data):
    return x_data.reshape(len(x_data), 784)
    
def rebuild_image(decoded_data):
    return decoded_data.reshape(len(decoded_data), 28, 28)
