#cost에 대한 미분 결과를 어떻게 계산하는지 보여준다.
# 여기서 한발 더 나아가서 텐서플로우가 계산한 값과 직접 계산한 값이 똑같다는 것까지 보여준다.

import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]


W = tf.Variable(tf.random_uniform([1], -10000., 10000.))        # tensor 객체 반환
#W = tf.Variable(tf.random_normal([1]), name='weight')


X = tf.placeholder(tf.float32)      # 반복문에서 x_data, y_data로 치환됨
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 동영상에서 미분을 적용해서 구한 새로운 공식. cost를 계산하는 공식
# Minimize
#cost = tf.reduce_mean(tf.square(hypothesis - Y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#train = optimizer.minimize(cost)
# 와 비교해보기
learning_rate = 0.1
gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))   # 변경된 W가 mean에도 영향을 준다
descent = W - learning_rate * gradient
# W 업데이트. tf.assign(W, descent). 호출할 때마다 변경된 W의 값이 반영되기 때문에 업데이트된다.
update = W.assign(descent)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    uResult = sess.run(update, feed_dict={X: x_data, Y: y_data})  # 이 코드를 호출하지 않으면 W가 바뀌지 않는다.
    cResult = sess.run(cost, feed_dict={X: x_data, Y: y_data})  # update에서 바꾼 W가 영향을 주기 때문에 같은 값이 나온다.
    wResult = sess.run(W)
    mResult = sess.run(gradient, feed_dict={X: x_data, Y: y_data})

    # 결과가 오른쪽과 왼쪽 경사를 번갈아 이동하면서 내려온다. 기존에 한 쪽 경계만 타고 내려오는 것과 차이가 있다.
    # 최종적으로 오른쪽과 왼쪽 경사의 중앙에서 최소 비용을 얻게 된다. (생성된 난수값에 따라 한쪽 경사만 타기도 한다.)
    # descent 계산에서 0.1 대신 0.01을 사용하면 오른쪽 경사만 타고 내려오는 것을 확인할 수 있다. 결국 step이 너무 커서 발생한 현상
    print('{} {} {} [{}, {}]'.format(step, mResult, cResult, wResult, uResult))

print('-' * 50)
print('[] 안에 들어간 2개의 결과가 동일하다. 즉, update와 cost 계산값이 동일하다.')

print(sess.run(hypothesis, feed_dict={X: 5.0}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))

sess.close()

# update 계산을 통해 얻은 W를 cost에 전달하고 있다.
# 그럼에도 불구하고 cost는 정상적으로 최저점을 찾아서 진행한다.
# 이 말은 update 계산에 포함된 공식이 올바르게 GradientDescentOptimizer 함수의 역할을 하고 있다는 뜻이 된다.
# 즉, GradientDescentOptimizer 함수 없이 직접 만들어서 구동할 수도 있다는 것을 보여주는 예제이다.
# 또한 새로 적용된 cost 함수 또한 본래 함수와 비교했을때 잘 작동함을 볼 수 있다
