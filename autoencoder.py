import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import os 




"""
if you want to save weights. edit it for using tf.Saver!
I don't mind it.
find "Session initializer!"
"""

training_epoch = 2000
learning_rate = 0.01
input_size = 784
batch_size = 100
n_hidden_size = 784
n_hiddens = [784, 300, 150, 50]
n_out = 10



(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()


def normalizer(x):
    return (x/255.)


def inference_encoder(x, n_hidden = None):
    def weight(shape):
        weight_var = tf.Variable(tf.truncated_normal(shape))
        return weight_var
    
    def bias(shape):
        bias_var = tf.Variable(tf.zeros(shape))
        return bias_var


    
    def hidden(x, n_hidden): 
        layer = None
        for layer_num in range(0,len(n_hidden)-1):
            w = weight(n_hidden[layer_num:layer_num+2])
            b = bias(n_hidden[layer_num+1])
        
            if layer_num == 0 :
                layer = tf.add(tf.matmul(x, w),b)
                layer = tf.nn.relu(layer)
            else :
                layer = tf.add(tf.matmul(layer, w), b)
                layer = tf.nn.relu(layer)
        return layer

    encoder = hidden(x, n_hidden)
    
    n_hidden.reverse()
    decoder = hidden(encoder, n_hidden)

    return decoder



def cost(x, decoder):
    cost = tf.reduce_mean(tf.pow(x-decoder, 2), -1)
    return cost


def train(cost, lr = 0.01):
    train = tf.train.AdamOptimizer(learning_rate=lr)
    train=train.minimize(cost)
    return train


    
X = x_train.reshape(len(x_train), 784)

normalizer(X)
t = tf.placeholder(tf.float32, [None, input_size])

decoder = inference_encoder(t, n_hiddens)
cost_func = cost(t, decoder)
optimizer = train(cost_func, lr = learning_rate)



#Session initilizer!
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print("start training!\n")
for epoch in range(training_epoch):
    
    start = epoch*batch_size
    end = start+batch_size+1

    op = sess.run(optimizer, feed_dict = {t: X[start:end]})

    print("epochs :",epoch, "op : ", op)
print("end training!\n")


print("test eval start\n")


print(x_test.shape)

X= x_test.reshape(len(x_test), 784)








