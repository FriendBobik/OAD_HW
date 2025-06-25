import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from generate import generate_data

if __name__ == "__main__":

    seed = 42

    nrows = 500
    ncols = 20
    nrealizations = 10000
    variance = 0.01

    # Сгенерированные данные
    A, x, B = generate_data(nrows, ncols, nrealizations, variance, seed)
    
    # Массивы для оценок параметров и невязок
    estimated_x = np.zeros((nrealizations, ncols))
    residual_norms_squared = np.zeros(nrealizations)

    # Расчет оценок и невязок
    for i in range(nrealizations):
        b_i = B[i, :]
        x_estimated, residuals, rank, s = np.linalg.lstsq(A, b_i, rcond=None) # не похоже на реализованную в предыдущем упражнении
        estimated_x[i, :] = x_estimated
        residual_norms_squared[i] = residuals

    # Число степеней свободы
    df = nrows - ncols

    x = np.linspace(chi2.ppf(0.01, df),chi2.ppf(0.99, df), 100) #честно украл с сайта scipy, скопировал дословно

    # Визуализация гистограммы невязок
    plt.figure(figsize=(10, 6))
    plt.plot(x/100, chi2.pdf(x, df)*100, 'r-', lw=5, alpha=0.6, label='Теоретическое распределение Хи-квадрат')


    plt.hist(residual_norms_squared, bins=100, density=True, alpha=0.6, label='Невязка МНК')
    plt.xlabel('Значение')
    plt.ylabel('Плотность вероятности')
    plt.title('Сравнение распределения величин невязки с распределением Хи-квадрат')
    plt.legend() 
    plt.savefig('chi2.png')
    plt.show()
