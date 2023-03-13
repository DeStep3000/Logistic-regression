import numpy as np
import cvxpy as cp
import time

# Задаем размерности n
n_values = [2, 5, 10]

# Задаем количество тестовых примеров
N = 100

# Задаем коэффициент регуляризации
alpha = 0.1

# Задаем списки для сохранения значений
solution_times = []
global_minima = []
optimal_values = []

# Генерируем тестовые данные для каждой размерности n
for n in n_values:
    # Генерируем матрицу признаков X размера (N, n)
    X = np.random.rand(N, n)
    # Генерируем вектор меток Y размера (N, )
    Y = np.random.randint(0, 2, N)
    # Создаем единичный столбец в матрице X для учета свободного члена
    X = np.hstack((np.ones((N, 1)), X))
    # Инициализируем веса W
    W = np.zeros(n + 1)
    # Обучаем модель логистической регрессии с регуляризацией
    for i in range(100):
        # Вычисляем предсказания модели
        y_pred = 1 / (1 + np.exp(-np.dot(X, W)))
        # Вычисляем градиент функции потерь
        grad = np.dot(X.T, (y_pred - Y)) + alpha * W
        # Обновляем веса
        W -= 0.01 * grad
    # Определяем переменные и функцию для оптимизации с помощью CVXPY
    x = cp.Variable(n + 1)
    f = cp.sum(cp.logistic(-cp.multiply(Y, X @ x))) + alpha * cp.sum_squares(x[1:])
    prob = cp.Problem(cp.Minimize(f))
    # Решаем задачу с помощью CVX
    start_time = time.time()
    prob.solve()
    end_time = time.time()
    # Сохраняем результаты
    solution_times.append(end_time - start_time)
    global_minima.append(x.value)
    optimal_values.append(prob.value)
    # Выводим результаты
    print(f"Размерность вектора переменных: {n}")
