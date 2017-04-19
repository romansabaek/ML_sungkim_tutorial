import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [1., 2., 3., 4.]
y_data = [2., 4., 6., 8.]

W = tf.Variable(tf.random_uniform([1], -100., 100.))
b = tf.Variable(tf.random_uniform([1], -100., 100.))


#언제든지 다른 데이터를 전달할 수 있도록 placeholder X를 사용하고 있다.
# cost를 계산하는 코드에서도 y_data 대신 placeholder Y를 전달했다.
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])


hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

#for문 바깥에서 X에 새로운 데이터를 넣어서 구동하는 코드가 중요하다.
# 우리의 목적은 비용을 최소로 만드는 기울기(W)와 y 절편( b)를 구하는 것

