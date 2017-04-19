import tensorflow as tf
import numpy as np

unique = 'helo'
#. language model은 다음에 올 글자나 단어를 예측하는 모델이어서, 마지막 글자가 입력으로 들어와도 예측할 수가 없다.
batch_size = 1
time_step_size = 4
rnn_size = 4

y_data = [1, 2, 2, 3]   # 'ello'. index from 'helo'
x_data = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0]], dtype='f')  # 'hell'

cells  = tf.nn.rnn_cell.BasicRNNCell(4)  # 출력 결과(4가지 중에서 선택) 4-> rnn_size ->output 4개
state  = tf.zeros([batch_size, cells.state_size]) # shape(1, 4), [[0, 0, 0, 0]]
x_data = tf.split(0, time_step_size, x_data)          # layer에 포함될 cell 갯수(4). time_step_size
#tf.split 함수는 데이터를 지정한 갯수로 나눈 다음, 나누어진 요소를 Tensor 객체로 만들어 리스트에 넣어서 반환한다.




# outputs = [shape(1, 4), shape(1, 4), shape(1, 4), shape(1, 4)]
# state = shape(1, 4)
outputs, state = tf.nn.rnn(cells, x_data, state) # x_data 입력 staet 상태

# tf.reshape(tensor, shape, name=None)
# tf.concat(1, outputs) --> shape(1, 16)
logits  = tf.reshape(tf.concat(1, outputs), [-1, rnn_size]) # shape(4, 4)
targets = tf.reshape(y_data, [-1])                   # shape(4), [1, 2, 2, 3]
weights = tf.ones([time_step_size * batch_size])                               # shape(4), [1, 1, 1, 1]


loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights]) # 예측값 실제값 weight
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        sess.run(train_op)
        r0, r1, r2, r3 = sess.run(tf.argmax(logits, 1)) # 예측값 을 찍어봄
        print(r0, r1, r2, r3, ':', unique[r0], unique[r1], unique[r2], unique[r3])

