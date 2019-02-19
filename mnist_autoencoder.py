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
def normalizer(data):
  mean = np.mean(data)
  std = np.std(data)
  data = (data - mean) / std
  return data
"""
def normalizer(x_data):
    return (x_data/255.)
"""

def unnormalizer(x_data):
    return x_data*255.
    

class Autoencoder():

    def __init__(self, input_size, n_hidden, batch_size=256, epochs=20, learning_rate = 0.01):
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def inference_flatted(self, x):
        def weight_var(shape, name):
            #weight = tf.Variable(tf.truncated_normal(shape, stddev = 0.01))
            weight = tf.get_variable(name = 'W_'+name, shape=shape, initializer = tf.contrib.layers.xavier_initializer())
            #weight = tf.reshape(weight, [1]+shape)
            return weight

        def bias_var(shape, name):
            #bias = tf.Variable(tf.zeros(shape))
            bias = tf.get_variable(name = 'b_'+name, shape=shape, initializer = tf.contrib.layers.xavier_initializer())
            return bias

        def hidden(x, n_hidden, func_name):
            layer = None
            for number in range(0, len(n_hidden) - 1) :
                w = weight_var(n_hidden[number:number+2], str(number)+func_name)
                b = bias_var(n_hidden[number+1], str(number)+func_name)

                if number == 0:
                    layer = tf.nn.relu(tf.add(tf.matmul(x,w), b))
                elif number == len(n_hidden)-2 and func_name == 'decoder':
                    layer = tf.add(tf.matmul(layer, w), b)
                else:
                    layer =tf.nn.relu(tf.add(tf.matmul(layer,w), b))                
            return layer
        self.x = x
        encoder = hidden(self.x, self.n_hidden, 'encoder')
        reversed_hidden = [x for x in self.n_hidden[::-1]]
        self.decoder = hidden(encoder, reversed_hidden, 'decoder')

        return self.decoder


    def inference_normal(self, x):
        def weight_var(shape, name):
            #weight = tf.Variable(tf.truncated_normal(shape, stddev = 0.01))
            weight = tf.get_variable(name = 'W_'+name, shape=shape, initializer = tf.contrib.layers.xavier_initializer())
            #weight = tf.reshape(weight, [1]+shape)
            return weight

        def bias_var(shape, name):
            #bias = tf.Variable(tf.zeros(shape))
            bias = tf.get_variable(name = 'b_'+name, shape=shape, initializer = tf.contrib.layers.xavier_initializer())
            return bias

        def hidden(x, n_hidden, func_name):
            layer = None
            for number in range(0, len(n_hidden) - 1) :
                w = weight_var(n_hidden[number:number+2], str(number)+func_name)
                b = bias_var(n_hidden[number+1], str(number)+func_name)

                if number == 0:
                    layer = tf.nn.relu(tf.add(tf.einsum("ijk,kh->ijh",x,w), b))
                elif number == len(n_hidden)-2 and func_name == 'decoder':
                    layer = tf.add(tf.einsum("ijk,kh->ijh",layer, w), b)
                else:
                    layer =tf.nn.relu(tf.add(tf.einsum("ijk,kh->ijh", layer,w), b))                
            return layer
        self.x = x
        encoder = hidden(self.x, self.n_hidden, 'encoder')
        reversed_hidden = [x for x in self.n_hidden[::-1]]
        self.decoder = hidden(encoder, reversed_hidden, 'decoder')

        return self.decoder

    #cost function
    def cost_func(self, x_data, decoder):
        #self.cost = tf.reduce_mean(tf.pow(x_data-decoder, 2))
        self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(x_data - decoder), axis = -1))
        return self.cost

    def training(self, cost, lr = 0.01):
        self.train = tf.train.AdamOptimizer(learning_rate = lr)
        self.train = self.train.minimize(cost)
        return self.train

    def fit(self, x_data, sess):
        print("==start training==\n")
        for epoch in range(self.epochs):
            np.random.shuffle(x_data)
            for batch in range(len(x_data)//self.batch_size):
                start = batch*self.batch_size
                end = start+self.batch_size
                _, loss = sess.run([self.train, self.cost], feed_dict = {self.x:x_data[start:end]})
                print("epochs : ",epoch, "batch : ", end,"/",len(x_data)//self.batch_size, "loss : ", loss, end = '\r')
            print("epochs : ",epoch, "batch : ", end,"/",len(x_data)//self.batch_size, "loss : ", loss)

        print("===end training===")

    """
    def eval_cost(self, x_data, sess):
        co = sess.run(self.cost, feed_dict = { self.x : x_data })
        return co
    """

    def test(self, test_data, sess):
        print("test")
        decode_data, loss = sess.run([self.decoder, self.cost], feed_dict={self.x:test_data})
        print(loss)
        #decode_data = unnormalizer(decode_data)
        test_data = viewer.rebuild_image(test_data)
        decode_data = viewer.rebuild_image(decode_data)
        viewer.viewer(test_data[0:7], decode_data[0:7])

        







        



