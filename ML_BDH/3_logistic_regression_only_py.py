import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+math.e**-z)

#test
print(sigmoid(100))
print(sigmoid(0))
print(sigmoid(-10))
print(sigmoid(np.array([100,0,-10])))

def cost_function(W,X,y):

    m = y.size
    h = sigmoid(np.dot(W,X))
    cost = -(1/m) * sum(y * np.log(h) + (1-y)*np.log(1-h))
    grad = (1/m) * np.dot(X,h-y)

    return cost,grad


def graph():
    # 옥타브와 비슷한 형태로 그래프 출력
    # x_data[0,pos]에서 0은 행, pos는 열을 가리킨다. 쉼표 양쪽에 범위 또는 인덱스 배열 지정 가능.
    t1 = plt.plot(x_data[0, pos], x_data[1, pos], color='black', marker='+', markersize=7)
    t2 = plt.plot(x_data[0, neg], x_data[1, neg], markerfacecolor='yellow', marker='o', markersize=7)

    plt.xlabel('exam 1 score')
    plt.ylabel('exam 2 score')
    plt.legend([t1[0], t2[0]], ['Admitted', 'Not admitted'])  # 범례

    plt.show()


xy = np.loadtxt('ex2data1.txt',unpack=True, dtype='float32', delimiter=',')

print("\ndata set")
print(xy.shape)
print(xy[:,:5])

x_data = xy[:-1]
y_data = xy[-1]

pos = np.where(y_data==1)
neg = np.where(y_data==0)

graph()

#----------------------------------------------#

n,m = x_data.shape
print('m,n :',m,n)

x_data = np.vstack((np.ones(m), x_data))

print(x_data.shape)         # 100
print(x_data[:,:5])

# [[  1.           1.           1.           1.           1.        ]
#  [ 34.62366104  30.28671074  35.84740829  60.18259811  79.03273773]
#  [ 78.02469635  43.89499664  72.90219879  86.3085556   75.34437561]]

W = np.zeros(n+1)           # [ 0.  0.  0.]. 1행 3열

cost, grad = cost_function(W, x_data, y_data)
print('------------------------------')
print('cost :',  cost)      # cost : 0.69314718056
print('grad :', *grad)      # grad : -0.1 -12.0092164707 -11.2628421021

plt.plot([28.059, 101.828], [96.166, 20.653], 'b')

