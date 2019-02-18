import viewer as vr 
import mnist_autoencoder as auto 
from tensorflow import keras
import tensorflow as  tf 


training_epoch = 30
learning_rate = 0.01
input_size = 784
batch_size = 100
n_hidden_size = 784
#n_hiddens = [784, 512, 256, 128, 64]
n_hiddens = [784, 128, 128]
n_out = 10

def session():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return sess
fashion_mnist = keras.datasets.fashion_mnist

(x_train, _), (x_test, _) = fashion_mnist.load_data()

if __name__ == '__main__':
    x_train = auto.normalizer(x_train)
    x_test = auto.normalizer(x_test)

    autoencoder = auto.Autoencoder(input_size, n_hiddens, batch_size, training_epoch)
    t = tf.placeholder(tf.float32, [None, input_size])
    decoder = autoencoder.inference(t)
    cost = autoencoder.cost_func(t, decoder)
    optimizer = autoencoder.training(cost)
    
    
    x_data = vr.convert_to_fullyconntected(x_train)
    sess = session()
    autoencoder.fit(x_data, sess)
    
    x_test = vr.convert_to_fullyconntected(x_test)
    autoencoder.test(x_test, sess)





