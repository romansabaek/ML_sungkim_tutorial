import tensorflow as tf
tf.set_random_seed(777)

# data
# 실수 대신 정수 데이터를 사용하기 위해서는 데이터와 변수를 함께 바꾸어야 한다.
# 그럼에도 여전히 난수의 내부 타입은 float32를 사용해야 한다. int32 등의 자료형은 에러.
x_train = [1, 2, 3]
y_train = [1, 2, 3]

#y_data = x_data * W + b
#W를 1로 b를 0 으로 만들고 싶다.
W = tf.Variable(tf.random_uniform([1],0,32, dtype=tf.float32)) #난수 1개를 생성
b = tf.Variable(tf.random_uniform([1],0,32)) # 정수 생성의 경우 2**n 규칙을 따르는 것이 좋다. 구글 문서.


#hypothesis XW+b
hypothesis = x_train * W + b
# W는 1x1, x_data는 1x3 행렬이기 때문에, 행렬 연산에 따라 결과는 1x3의 행렬이 나온다.
# 행렬과 행렬이 아닌 값을 더하거나 곱할 때는 행렬의 모든 요소에 영향을 주기 때문에 덧셈 +b 또한 3회 발생한다.
# 하지만 아직 계산이 일어난 상태는 아님


# cost/loss function
# cost 즉 타겟값과 예측값과의 차이가 적어야함
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
# minimize 방법은 gradient 방법을 사용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#텐서플로우를 구동하기 위해서는 그래프에 연결된 모든 변수를 초기화해야 한다. 이 코드는 run 함수를 호출하기 전에 나와야 한다.
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))