import viewer as vr 
import mnist_autoencoder as auto 
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as  tf 


training_epoch = 2000
learning_rate = 0.01
input_size = 784
batch_size = 100
n_hidden_size = 784
n_hiddens = [784, 300, 150, 50]
n_out = 10

def session():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return sess

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

if __name__ == '__main__':
    x_train = auto.normalizer(x_train)
    x_test = auto.normalizer(x_test)
    autoencoder = auto.Autoencoder(input_size, n_hiddens, batch_size, training_epoch)
    t = tf.placeholder(tf.float32, [None, input_size])
    decoder = autoencoder.inference(t)
    cost = autoencoder.cost_func(t, decoder )
    optimizer = autoencoder.training(cost)
    
    
    x_data = vr.convert_to_fullyconntected(x_train)
    autoencoder.fit(x_data, session())





