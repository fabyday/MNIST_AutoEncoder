import viewer as vr 
import mnist_autoencoder as auto 
from tensorflow import keras
import tensorflow as  tf 


import sys

training_epoch = 30
learning_rate = 0.01
input_size = 28
batch_size = 100
n_hidden_size = 784


#if you use flatted image. using this arrays for autoencoder input.
flatted_n_hiddens = [784, 128, 128]
#if you use normal, not flatted. use this array.
normal_n_hiddens = [28, 128, 128]

n_out = 10

def session():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return sess
fashion_mnist = keras.datasets.fashion_mnist

(x_train, _), (x_test, _) = fashion_mnist.load_data()

if __name__ == '__main__':
    
    #typing 'normal or flatted' which you want. it make autoencoder automatically!
    if len(sys.argv) == 1 : 
        print("[autoencoder] : can't find argument. please write right argument.\n")
        print("ex) training [flatted | normal]")
        sys.exit()

    flag = sys.argv[1]
    print(flag)

    x_train = auto.normalizer(x_train)
    x_test = auto.normalizer(x_test)

    if flag == 'flatted':
        autoencoder = auto.Autoencoder(input_size**2, flatted_n_hiddens, batch_size, training_epoch)
        t = tf.placeholder(tf.float32, [None, input_size])
    
        x_data = vr.convert_to_fullyconntected(x_train)
        x_test = vr.convert_to_fullyconntected(x_test)
        decoder = autoencoder.inference_flatted(t)

    elif flag == 'normal':
        autoencoder = auto.Autoencoder(input_size, normal_n_hiddens, batch_size, training_epoch)
        t = tf.placeholder(tf.float32, [None, input_size, input_size])
        x_data = x_train
        x_test = x_test

        decoder = autoencoder.inference_normal(t)

    else :
        print("[autoencoder] : you write wrong arguments. please type right argument.")
        print("ex) training [flatted | normal]")
        sys.exit()


    cost = autoencoder.cost_func(t, decoder)
    optimizer = autoencoder.training(cost)

    sess = session()
    autoencoder.fit(x_data, sess)
    autoencoder.test(x_test, sess)
