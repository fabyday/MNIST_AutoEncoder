import tensorflow as tf 
import numpy as np 
import viewer

input_size = 784




"""
if you want to save weights. edit it for using tf.Saver!
I don't mind it.
find "Session initializer!"
viewer is custom viewer that made by matplotlib
"""


def normalizer(x_data):
    return (x_data/255.)
    

class Autoencoder():

    def __init__(self, input_size, n_hidden, batch_size=100, epochs=2000, learning_rate = 0.01):
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate


    def inference(self, x):
        def weight_var(shape):
            weight = tf.Variable(tf.truncated_normal(shape))
            return weight

        def bias_var(shape):
            bias = tf.Variable(tf.zeros(shape))
            return bias

        def hidden(x, n_hidden):
            layer = None
            for number in range(0, len(n_hidden) - 1) :
                print(n_hidden[number:number+2])
                w = weight_var(n_hidden[number:number+2])
                b = bias_var(n_hidden[number+1])

                if number == 0 :
                    layer = tf.nn.relu(tf.add(tf.matmul(x,w), b))
                else:
                    layer = tf.nn.relu(tf.add(tf.matmul(layer,w), b))
            return layer
        self.x = x
        encoder = hidden(self.x, self.n_hidden)
        reversed_hidden = [x for x in self.n_hidden[::-1]]
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

    def fit(self, x_data, sess):
        print("==start training==\n")
        for epoch in range(self.epochs):
            np.random.shuffle(x_data)
            print("epochs : ", epoch)
            for batch in range(len(x_data)//self.batch_size+1):
                start = batch*self.batch_size
                end = start+self.batch_size
                _, loss = sess.run([self.train, self.cost], feed_dict = {self.x:x_data[start:end]})
                print("batch : ", end,"/",len(x_data), end = '\r')
                #print("eval cost : ", self.eval_cost(x_data, sess), end="\r")
        print("end training")

    """
    def eval_cost(self, x_data, sess):
        co = sess.run(self.cost, feed_dict = { self.x : x_data })
        return co
    """






        



