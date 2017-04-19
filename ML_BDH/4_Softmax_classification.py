import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('ml_data/data_softmax.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7  # 0 ~ 6

#print shape을 통해서 확인 가능함
X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6 데이터 개수는 n -> none

Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot 0~6 까지 숫자를 바꿔준다. 몇개의 클래스가 있는가?
# 만약 인풋 랭크가 n이면 출력 랭크는 차원을 하나더한 n+1이 된다 그래서 아래와 같은 작업이 필요하다

print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # -1은 나머지를 말함! , reshape 을 통해서 우리가 원하는 결과를 얻음
print("reshape", Y_one_hot)

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y_one_hot * tf.log(hypothesis), reduction_indices=1)) #Y는 one hot
#reduction indices 가 0이면 열합계, 1이면 행합계 아무것도 전달하지 않으면 전체합계

#cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(X, W) + b,
#                                                 labels=Y_one_hot) #entropy가 감소하는 방향으로 진행하다 보면 최저점을 만나게 된다.
#cross entropy 와 logistic cost는 비슷하다 !

#cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1) #argmax -> one hot encoder
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                                 X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()): # y flatten->> [[],[],] -> [ ]
        #각각의 element 들을 넘겨주기 편하게 zip으로 묶어준다
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))