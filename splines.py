import numpy as np
import matplotlib.pyplot as plt


def evaluate_loss(f, S_i):
    error = (f-S_i)**2
    return error


def evaluate_MSE(a, b, c, d, x, x_vals):
    total_error = 0
    count = 0
    for x_val in x_vals:
        for i in range(len(x) - 1):
            if x[i] <= x_val <= x[i + 1]:
                dx = x_val - x[i]
                y_val = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
                true_val = true_func(x_val)
                total_error += np.abs((true_val - y_val)**2)
                count += 1
                break
    return total_error / count if count > 0 else 0


def polynom_interpolate(x, y):
    n = len(x)
    A = np.zeros((n, n))

    for i in range(0, n):
       for j in range(0, n):
           A[i,j] = x[i]**j

    c = np.linalg.solve(A, y)
    return c


def evaluate_polynom(x,c,xs):
    n = len(x)
    ys = []
    for x_val in xs:
        y_val = 0
        for i in range(n):
            y_val += c[i] * x_val ** i
        ys.append(y_val)
    return ys


def polynom_MSE(x,c,xs):
    n = len(x)
    ys = []
    total_error = 0
    count = 0
    for x_val in xs:
        y_val = 0
        for i in range(n):
            y_val += c[i] * x_val ** i
        true_val = true_func(x_val)
        total_error += (true_val - y_val)**2
        count +=1
    return total_error / count if count > 0 else 0


def cubic_spline(x, y, mode):
    n = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(n)]

    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    # 13
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        B[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    if mode == 2:   #natural
        A[0, 0] = 1
        A[n, n] = 1
        B[0] = 0
        B[n] = 0
    elif mode == 3: #not a knot
        A[0, 0] = h[1]
        A[0, 1] = -(h[0] + h[1])
        A[0, 2] = h[0]
        B[0] = 0

        A[n, n - 2] = h[n - 1]
        A[n, n - 1] = -(h[n - 2] + h[n - 1])
        A[n, n] = h[n - 2]
        B[n] = 0
    else:   #clumped
        f_prime_0 = (y[1] - y[0]) / h[0]
        f_prime_n = (y[n] - y[n-1]) / h[n-1]

        A[0,0] = 2*h[0]
        A[0,1] = h[0]
        B[0] = 3*((y[1] - y[2])/h[0] - f_prime_0)

        A[n,n-1] = h[n-1]
        A[n,n] = 2*h[n-1]
        B[n] = 3*(f_prime_n - (y[n] - y[n-1])/h[n-1])

    c = np.linalg.solve(A, B)

    a = [y[i] for i in range(n)]
    b = []
    d = []

    for i in range(n):
        b_i = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3   #9
        d_i = (c[i + 1] - c[i]) / (3 * h[i])                                #6
        b.append(b_i)
        d.append(d_i)

    return a, b, c[:-1], d


def evaluate_spline(x, a, b, c, d, xs):
    n = len(x) - 1
    ys = []
    for x_val in xs:
        for i in range(n):
            if x[i] <= x_val <= x[i + 1]:
                dx = x_val - x[i]
                y_val = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
                ys.append(y_val)
                break
    return ys


def plot_splines(t, S_i, labels, f, x, y):
    plt.title('Cubic spline interpolation')
    plt.plot(t, f, color='black', label='sin(x)/x')
    for i in range(len(S_i)):
        plt.plot(t, S_i[i], label=labels[i], linestyle='--')
    plt.plot(x, y, 'o', markersize=5, label='initial points')
    plot_conf()


def plot_losses(t, losses, labels, MSE):
    plt.title('interpolation loss graph')
    for i in range(len(losses)):
        label = f"{labels[i]} loss (MSE = {MSE[i]:.4f})"
        plt.plot(t, losses[i], label=label, linestyle='dotted')
    plot_conf()


def true_func(t):
    return np.abs(t-np.floor(t+0.5))


def plot_conf():
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()


x = np.arange(0, 5.1, 0.25)
y = [0.0, 0.25, 0.5, 0.25, 0.0, 0.25, 0.5, 0.25, 0.0, 0.25, 0.5, 0.25, 0.0, 0.25, 0.5, 0.25, 0.0, 0.25, 0.5, 0.25, 0.0]

n = len(x) -1

t =  np.linspace(x[0], x[n], 1000)
f = np.abs(t-np.floor(t+0.5))

c = polynom_interpolate(x, y)
y_p = evaluate_polynom(x,c,t)
loss_p = evaluate_loss(f, y_p)
MSE_p = polynom_MSE(x,c,t)

a, b, c, d = cubic_spline(x, y, 2)
S_1 = evaluate_spline(x, a, b, c, d, t)
MSE1 = evaluate_MSE(a, b, c, d, x, t)
loss = evaluate_loss(f, S_1)

a, b, c, d = cubic_spline(x, y, 3)
S_2 = evaluate_spline(x, a, b, c, d, t)
MSE2 = evaluate_MSE(a, b, c, d, x, t)
loss2 = evaluate_loss(f, S_2)

a, b, c, d = cubic_spline(x, y, 1)
S_3 = evaluate_spline(x, a, b, c, d, t)
MSE3 = evaluate_MSE(a, b, c, d, x, t)
loss3 = evaluate_loss(f, S_3)

labels = ["Natural", "Not-a-knot", "Clumped", "Polynomial"]
S_i = [S_1,S_2,S_3, y_p]
losses = [loss, loss2, loss3, loss_p]
MSE = [MSE1,MSE2,MSE3, MSE_p]
plt.figure(1)
plot_splines(t, S_i, labels, f, x, y)
plt.figure(2)
plot_losses(t, losses, labels, MSE)

plt.show()
print("good")