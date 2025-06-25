from astropy.io import fits     # Какая-то астрологическая библиотека, вроде документация есть
from lsp import lstsq           # Импортируем прикольчик lstsq из модуля lsp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with fits.open('ccd_v2.fits') as file: # Открываем прикольчик
        data = file[0].data

    #print(data.shape)
        
    data = np.asarray(data, dtype=np.int32)
    #print(data)

    data_better = (data[:, 1, :, :] - data[:, 0, :, :]) 
    sigma_x = np.var(data_better, axis=(1, 2)) # Не забываем про саунтрек к этому коду https://www.youtube.com/watch?v=hRHoYTJV6q8
    x = np.mean(data, axis=(1, 2, 3))


    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    np.append(sigma_x, 1)

    # Подготовка матрицы A для линейной регрессии
    A = np.vstack((x, np.ones_like(x)))
    # Выполнение линейной регрессии с помощью метода наименьших квадратов
    args, cost, var = lstsq(A.T, sigma_x, 'svd', rcond=1e-6)

    # Отображение данных и линии линейной регрессии на графике
    datapoints, = plt.plot(x, sigma_x, 'o', markersize=3)
    linear, = plt.plot(x, x*args[0]+args[1], lw=2)
    ax.set_xlabel('общая засветка')
    ax.set_ylabel('выборочная дисперсия')
    datapoints.set_label('посчитанные точки')
    linear.set_label('линейное приближение')
    plt.legend(loc='best')
    plt.savefig('ccd.png')
    plt.show()

    # Вычисление коэффициента усилен ия и собственного шума на основе результатов регрессии
    g = 2/args[0]
    sigma_r = args[1]/g**2
    # Оценка ошибок для коэффициента усиления и собственного шума
    err_g = var[0, 0] / 2
    err_sigma = var[0, 0] - 2 * var[0][1] + 2 * var[1][1] 

    # Сохранение результатов в JSON-файл
    d = {
        "ron": np.round(sigma_r,2),
        "ron_err": np.round(err_sigma,2),
        "gain": np.round(g,2),
        "gain_err": np.round(err_g,6) 
        }
    import json
    with open('ccd.json', 'w') as f:
        json.dump(d, f, indent=2)

