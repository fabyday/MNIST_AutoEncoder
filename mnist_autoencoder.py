import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg


input_size = 784




"""
if you want to save weights. edit it for using tf.Saver!
I don't mind it.
find "Session initializer!"
"""




def normalizer(x_data):
    return (x_data/255.)
def convert_to_fullyconntected(x_data):
    pass
    

def rebuild_image(decoded_data):
    return x.reshape(len(x), 28, 28)


class Autoencoder():

    def __init__(self, input_size, n_hidden, batch_size=100, epochs=2000, learning_rate = 0.01):
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate


    def inference(self, x, n_hidden):
        def weight_var(shape):
            weight = tf.Variable(tf.truncated_normal(shape))
            return weight

        def bias_var(shape):
            bias = tf.Variable(tf.zeros(shape))
            return bias

        def hidden(x, n_hidden):
            layer = None
            for number in range(0, len(n_hidden) - 1) :
                w = weight_var(n_hidden[number:number+2])
                b = bias_var(n_hidden[number+1])

                if number == 0 :
                    layer = tf.nn.relu(tf.add(tf.matmul(x,w), b))
                else:
                    layer = tf.nn.relu(tf.add(tf.matmul(x,w), b))
            return layer

        encoder = hidden(x, n_hidden)
        reversed_hidden = [x for x in n_hidden[::-1]]
        self.decoder = hidden(encoder, reversed_hidden)

        return self.decoder


    #cost function
    def cost_func(self, x_data, decoder):
        self.cost = tf.reduce_mean(tf.pow(x_data-decoder, 2), -1)
        return self.cost

    def training(self, cost, lr = 0.01):
        self.train = tf.train.AdamOptimizer(learning_rate = lr)
        self.train = self.train.minimize(cost)
        return self.train

    def fit(self, x_data, sess, training_epoch=self.epochs, batch_size= self.batch_size, cost = self.cost):
        print("==start training==\n")
            for epoch in range(training_epoch):
                print("epochs : ", epoch)
                for batch in range(batch_size):
                    start = epoch*batch_size



        



