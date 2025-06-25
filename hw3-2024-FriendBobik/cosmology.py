

# жесть самим файл надо создать, а то я его ищу в папке с проектом, а его нет


import numpy as np
import matplotlib.pyplot as plt
import json

from scipy.integrate import quad
from opt import gauss_newton , lm

from music.sound import sound      # моя веселая шизофрения, чтобы не было скучно кодить



music_ON=True 
music_ON=False # если хочеться проверять с кайфом, то надо закоментить эту строку музыка будет играть, до закрытия графиков

if music_ON:
    print("Выберете ваше настроение: веселое или грусное")

    nastroenie=input()
    sound(nastroenie)


z_values = []
mu_values = []

with open('jla_mub.txt', 'r') as file:
    # Пропуск заголовка
    next(file)
    
    for line in file:
        z, mu = line.split()
        z_values.append(float(z))
        mu_values.append(float(mu))


def calculation_mu(z, x): #просто формулки из readme
    H0, Omega = x  # Распаковка параметров H0 и Omega
    c = 3e5  # Скорость света в км/с
    pc_to_km = 3e13  # Конвертация из парсеков в километры
    z= np.array(z)

    def integrand(z_prime, Omega):
        return 1 / np.sqrt((1 - Omega) * (1 + z_prime)**3 + Omega)

    integral = np.array([quad(integrand, 0, z_val, args=(Omega))[0] for z_val in z])

    d = (c / H0) * (1 + z) * integral * pc_to_km

    mu = 5 * np.log10(d) - 5
    return mu




def jacobian(z, x, eps=1e-6):
    H0, Omega = x
    z = np.array(z)  
    mu_H0 = (calculation_mu(z, [H0 + eps, Omega]) - calculation_mu(z, [H0 - eps, Omega])) / (2 * eps)
    mu_Omega = (calculation_mu(z, [H0, Omega + eps]) - calculation_mu(z, [H0, Omega - eps])) / (2 * eps)
    
    J = np.vstack([mu_H0, mu_Omega]).T
    return J



x0 = [50, 0.5]  # [H_0, omega]

#подтягиваем функции из opt.py
result_gn = gauss_newton(y=mu_values, f=lambda x: calculation_mu(z_values, x), j=lambda x: jacobian(z_values, x), x0=x0) 
result_lm = lm(y=mu_values, f=lambda x: calculation_mu(z_values, x), j=lambda x: jacobian(z_values, x), x0=x0)

model_mu_values = calculation_mu(np.array(z_values), result_gn.x)

plt.figure(figsize=(10, 6))
plt.scatter(z_values, mu_values, color='blue', label='Наблюдаемые данные')  
plt.plot(z_values, model_mu_values, color='red', label='Подгоночкая кривая') 

plt.xlabel('Красное смещение, z')
plt.ylabel('Модуль расстояния, μ')
plt.title('Зависимость модуля расстояния от красного смещения')
plt.legend()
plt.grid(True)

plt.savefig('mu-z.png')
plt.show()





cost_gn = result_gn.cost
cost_lm = result_lm.cost

plt.figure(figsize=(10, 6))

plt.plot(cost_gn, label='Метод Гаусса-Ньютона', marker='.')

plt.plot(cost_lm, label='Метод Левенберга-Марквардта', marker='.')

plt.title('Зависимость функции потерь от номера итерации')
plt.xlabel('Номер итерации')
plt.ylabel('Функция потерь')
plt.legend()
plt.grid(True)

plt.savefig('cost.png')
plt.show()



results_data = {
    "Gauss-Newton": {
        "H0": result_gn.x[0],
        "Omega": result_gn.x[1],
        "nfev": result_gn.nfev
    },
    "Levenberg-Marquardt": {
        "H0": result_lm.x[0],
        "Omega": result_lm.x[1],
        "nfev": result_lm.nfev
    }
}

# Сохранение данных в файл JSON
with open('parameters.json', 'w') as json_file:
    json.dump(results_data, json_file, indent=4)

# Джисончик явно выдает какой-то бред 
    
#     {
#     "Gauss-Newton": {
#         "H0": 2103578871.5891337,
#         "Omega": 0.7251011621056246, ну это похоже на правду, а всё остальное какой-то бред
#         "nfev": 43
#     },
#     "Levenberg-Marquardt": {
#         "H0": 8504531.119108519,
#         "Omega": -748512.1381499475,
#         "nfev": 61
#     }
# }
    
# Я пока не очень понимаю в чем проблема, сейчас запушу так, мб потом разберусь