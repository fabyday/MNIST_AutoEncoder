# AutoEncoder
__Ver_KR__

MNIST 데이터를 사용한 autoencoder이며, CNN(Convolution Neural Network)을 사용하지 않은 간단한 오토인코더입니다.
간단한 데이터(MNIST)에 대해 잘 작동되는 모습을 볼 수 있습니다.


MNIST데이터에 대한 정보를 보고 싶으시다면 아래의 사이트를 참조해 주세요.
<br>tensorflow site : https://www.tensorflow.org/tutorials/keras/basic_classification



# 학습 & 원본데이터와 디코딩된 데이터 비교분석.
<pre>
<code>
python training [flatted | normal]
</code>
</pre>
[flatted] : [28,28]의 2차원 데이터를 1차원 데이터의 형식[28*28]으로 변환시켜서 학습시킵니다.
[normal] 원본 데이터(2차원 [28, 28])을 그대로 학습시킵니다.

__주의사항!__ 
학습가중치를 저장하는 tf.saver 코드를 사용하지 않았습니다. 
사용하고 싶으시다면 직접 코드를 삽입해야 합니다.


# 예시





# etc
autoencoder.py는 사용되지 않는 파일입니다. 다른 파일들을 사용해주세요.
mnist_autoencoder.py는 autoencoder.py를 기본 베이스로 하여 만들어진 것입니다.

