def cost(W, X, y):
    s = 0
    for i in range(len(X)):
        s += (W * X[i] - y[i]) ** 2

    return s / len(X)


def gradients(W, X, y):
    s = 0
    for i in range(len(X)):
        s += (W * X[i] - y[i]) * X[i]

    return s / len(X)



X = [1., 2., 3.]
Y = [1., 2., 3.]
W_val, cost_val = [], []

W = 100

for i in range(1000):
    c = cost(W, X, Y)

    g = gradients(W, X, Y)
    W = W - g * 0.01

    # cost는 거리의 제곱을 취하기 때문에 W가 음수이건 양수이건 상관없다.
    if c < 1.0e-15:
        break

    if i % 20 == 19:
        print('{:4} : {:17.12f} {:12.8f} {:12.8f}'.format(i + 1, c, g, W))

    #print('{:.1f}, {:.1f}'.format(W, c))

    W_val.append(W)
    cost_val.append(c)

# ------------------------------------------ #
import matplotlib.pyplot as plt

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()


