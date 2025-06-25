import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/29324222/how-can-i-do-a-maximum-likelihood-regression-using-scipy-optimize-minimize, немного можно скомуниздить отсюда


def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-5): # с данным значением 1e-3 не проходился тест от @embudanov, так что поставил 1e-5, если будет слишком долго работать, то можно вернуть обратно. Как я уже понял, дописав остальное, так и в слеющем, так что там тоже поменял
    # логарифм правдоподобия
    def log_likelihood(params):

        tau, mu1, sigma1, mu2, sigma2 = params
        # Проверка на положительность дисперсий, ну на всякий случай
        if sigma1 <= 0 or sigma2 <= 0:
            return NameError("Дисперсии должны быть положительными, http://mathprofi.ru/dispersia_diskretnoi_sluchainoi_velichiny.html#:~:text=–%20определение%20дисперсии.,возьмите%20на%20заметку%20для%20практики!")
        

        likelihood1 = tau * norm.pdf(x, mu1, sigma1)
        likelihood2 = (1 - tau) * norm.pdf(x, mu2, sigma2)

        total_likelihood = np.sum(np.log(likelihood1 + likelihood2))

        return -total_likelihood

    # Начальные параметры для оптимизации
    initial_params = [tau, mu1, sigma1, mu2, sigma2]
    # Ограничения для параметров
    bounds = [(0, 1), (None, None), (rtol, None), (None, None), (rtol, None)]
    # Минимизация функции логарифма правдоподобия
    result = minimize(log_likelihood, initial_params, bounds=bounds, tol=rtol)
    
    if result.success:
        return tuple(result.x)
    else:
        raise ValueError(result.message)




def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    # Инициализация параметров
    params = np.array([tau, mu1, sigma1, mu2, sigma2])
    
    # E-шаг
    def e_step(x, params):
        tau, mu1, sigma1, mu2, sigma2 = params
        w1 = tau * norm.pdf(x, mu1, sigma1)
        w2 = (1 - tau) * norm.pdf(x, mu2, sigma2)
        return w1 / (w1 + w2)
    
    # M-шаг
    def m_step(x, w):
        tau = np.mean(w)
        mu1 = np.sum(w * x) / np.sum(w)
        sigma1 = np.sqrt(np.sum(w * (x - mu1)**2) / np.sum(w))
        mu2 = np.sum((1 - w) * x) / np.sum(1 - w)
        sigma2 = np.sqrt(np.sum((1 - w) * (x - mu2)**2) / np.sum(1 - w))
        return np.array([tau, mu1, sigma1, mu2, sigma2])
    
    # EM-алгоритм
    while True:
        params_old = params
        w = e_step(x, params)
        params = m_step(x, w)
        # Проверка сходимости
        if np.allclose(params, params_old, rtol=rtol):
            break
    return params





def em_double_2d_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-5):
    
    sigma1 = np.diag(sigma1**2)
    sigma2 = np.diag(sigma2**2)
    

    params = np.array([tau, *mu1, *sigma1.diagonal(), *mu2, *sigma2.diagonal()])
    
    # зачем мы писали прошлое...
    def e_step(x, params):
        tau, mu1, sigma1, mu2, sigma2 = params[0], params[1:3], np.diag(params[3:5]), params[5:7], np.diag(params[7:9])
        w1 = tau * multivariate_normal.pdf(x, mean=mu1, cov=sigma1)
        w2 = (1 - tau) * multivariate_normal.pdf(x, mean=mu2, cov=sigma2)
        return w1 / (w1 + w2)
    
    def m_step(x, w):
        tau = np.mean(w)
        mu1 = np.dot(w, x) / np.sum(w)
        sigma1 = np.dot(w, (x - mu1)**2) / np.sum(w)
        mu2 = np.dot((1 - w), x) / np.sum(1 - w)
        sigma2 = np.dot((1 - w), (x - mu2)**2) / np.sum(1 - w)
        return np.array([tau, *mu1, *sigma1, *mu2, *sigma2])
    
    while True:
        params_old = params
        w = e_step(x, params)
        params = m_step(x, w)
        if np.allclose(params, params_old, rtol=rtol):
            break
    
    tau, mu1, sigma1, mu2, sigma2 = params[0], params[1:3], np.diag(params[3:5]), params[5:7], np.diag(params[7:9])
    return tau, mu1, sigma1.diagonal(), mu2, sigma2.diagonal()



def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2, rtol=1e-5):
    # Может позже, пока лень
    pass


if __name__ == "__main__":
    pass
