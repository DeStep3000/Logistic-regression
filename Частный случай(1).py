import numpy as np
import cvxpy as cp
import time

# 5 объектов, 10 признаков
X = np.random.rand(5, 10)
Y = np.random.randint(0, 2, 5)
X = np.hstack((np.ones((5, 1)), X))

f0_n = []
times_n = []

start_time = time.time()
for i in range(5):
    # текущий объект, целевая переменная и коэффициент
    x = X[i]
    y = Y[i] * 2 - 1
    k = cp.Variable(10 + 1)

    # Задаем коэффициент регуляризации
    alpha = 0.1

    # Определяем функцию потерь и оптимизационную задачу
    loss = cp.logistic(-y * cp.matmul(x, k)) + alpha * cp.sum_squares(x[1:])
    problem = cp.Problem(cp.Minimize(loss))

    # Решаем оптимизационную задачу и сохраняем результаты
    problem.solve(solver=cp.SCS)
end_time = time.time()

print(f'k.value: {k.value}')
print(f'loss.value: {loss.value}')
print(f'time: {end_time - start_time}')