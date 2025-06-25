#!/usr/bin/env python3


from collections import namedtuple # функция для создания подклассов кортежей с именованными полями  ну так написано в гугле
                                   # звучит сложно, но быстрый гуглеж показал что это импактно


import numpy as np


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""

# В семенаре мы вроде как реальзовывли метод градиентного спуска, ну основываясь на этом, реализуем метод Гаусса-Ньютона
def gauss_newton(y, f, j, x0, tol=1e-4, k=1, maxiter=10000):
    x = np.asarray(x0)
    nfev = 0
    cost = []

    while True:
        nfev += 1 #ограничение не кол-во итераций
        res = f(x) - y  
        cost.append(0.5 * np.dot(res, res))  
        J = j(x)  
        grad = J.T @ res  # вычисляем градиент функции стоимости
        grad_norm = np.linalg.norm(grad)  

        # Решаем систему уравнений
        delta = np.linalg.solve(J.T @ J, grad)

        x = x - k * delta

        #условия остановки
        if nfev > maxiter:
            break
       
        if len(cost) >= 2 and np.abs(cost[-1] - cost[-2]) <= tol * cost[-1]:
            #  len(cost) >= 2 хотябы 2 точки
            #  np.abs(cost[-1] - cost[-2]) <= tol * cost[-1] условие что мы приблизилсь к минимуму
            break

    return Result(nfev=nfev, cost=np.array(cost), gradnorm=grad_norm, x=x)

def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4, maxiter=10000): # добавим условие остановки + надо не забыть разобраться lmbd0=1e-2, nu=2 почему такие занчения
    x = np.asarray(x0)
    nfev = 0
    cost = []

    for i in range(maxiter):
        nfev += 1
        res = y - f(x)
        cost.append(0.5 * np.dot(res, res))
        J = j(x)
        A = J.T @ J
        g = J.T @ res
        grad_norm = np.linalg.norm(g)
        I = np.eye(len(x))

        # Решение модифицированной системы уравнений (J.T J + lmbd0I) Δx = J.T res
        dx = np.linalg.solve(A + lmbd0 * I, g)
        
        if np.linalg.norm(dx) / np.linalg.norm(x) < tol:
            break

        x_new = x + dx
        # проверяем уменьшилась ли ошибка
        if np.dot(y - f(x_new), y - f(x_new)) < np.dot(res, res):
            x = x_new
            lmbd0 /= nu
        else:
            lmbd0 *= nu

    return Result(nfev=nfev, cost=np.array(cost), gradnorm=grad_norm, x=x)


if __name__ == "__main__":
    pass
