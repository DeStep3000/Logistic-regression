import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt

# Задаем размерности n
n_values = [2, 5, 10]

# Задаем количество тестовых примеров
N = 100

# Задаем коэффициент регуляризации
alpha = 0.1

# Инициализируем переменные для сохранения результатов
x_star = []
f0_star = []
times = []

# Генерируем тестовые данные для каждой размерности n
for n in n_values:
    # Генерируем матрицу признаков X размера (N, n)
    X = np.random.rand(N, n)
    # Генерируем вектор меток Y размера (N, )
    Y = np.random.randint(0, 2, N)
    # Создаем единичный столбец в матрице X для учета свободного члена
    X = np.hstack((np.ones((N, 1)), X))
    # Инициализируем переменные для сохранения результатов текущей размерности
    x_star_n = []
    f0_star_n = []
    times_n = []
    # Решаем каждый тестовый пример с помощью CVX
    for i in range(N):
        # Определяем переменные и параметры оптимизации
        x = cp.Variable(n+1)
        y = Y[i] * 2 - 1
        l2_norm_squared = cp.sum_squares(x[1:])
        # Определяем функцию потерь и оптимизационную задачу
        loss = cp.logistic(-y * cp.matmul(X[i], x)) + alpha * l2_norm_squared
        problem = cp.Problem(cp.Minimize(loss))
        # Решаем оптимизационную задачу и сохраняем результаты
        start_time = time.time()
        problem.solve(solver=cp.SCS)
        end_time = time.time()
        x_star_n.append(x.value)
        f0_star_n.append(loss.value)
        times_n.append(end_time - start_time)
    # Сохраняем результаты текущей размерности
    x_star.append(x_star_n)
    f0_star.append(f0_star_n)
    times.append(times_n)
    # Выводим результаты
    print(f"Размерность вектора переменных: {n}")
    print(f"Среднее время решения: {np.mean(times_n)} секунд")
    print(f"Глобальный минимум x*: {x_star[n_values.index(n)][i]}")
    print(f"Оптимальное значение целевой функции f0(x*): {f0_star[n_values.index(n)][i]}")
    print("")

plt.plot(n_values, [np.mean(t) for t in times])
plt.xlabel("Размерность вектора переменных")
plt.ylabel("Среднее время решения, секунды")
plt.show()

