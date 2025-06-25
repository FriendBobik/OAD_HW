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
    pass




def em_double_2d_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-5):
    pass



def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2, rtol=1e-5):
    # Может позже, пока лень
    pass


if __name__ == "__main__":
    pass
