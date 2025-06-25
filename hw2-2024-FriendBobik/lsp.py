import numpy as np
#import time # тыкался в оптимизщацию


def lstsq_ne(A, b):
    #Преобразуем в масивы numpy
    #Да большая буква это обычно константа, но чтобы соответствовать конспекту сделаем так
    A = np.array(A)
    B = np.array(b) 
    
    AtA = np.dot(A.T, A) 
    AtB = np.dot(A.T, B)

    x = np.linalg.solve(AtA, AtB)
    
    r= B - np.dot(A, x) #из конспекта

    var= np.linalg.inv(AtA) # * r@r/(A.shape[0]-A.shape[1])

    return x, r, var
    # return x, r@r, var

def lstsq_svd(A, b, rcond=None):
    A = np.array(A)
    B = np.array(b) # "B" is not accessed

    U, sigma, VT = np.linalg.svd(A) #функция которая делает SVD разложение
    #Для того чтобы код с сигмой работал я его писал под этот ролик https://www.youtube.com/watch?v=hRHoYTJV6q8 всем рекомендую
    # принято!
    import os
    if os.name != 'posix':
        os.system('start "" "https://www.youtube.com/watch?v=hRHoYTJV6q8"')
    else:
        os.system('xdg-open "https://www.youtube.com/watch?v=hRHoYTJV6q8"')
    if rcond is None:
        rcond = np.finfo(sigma.dtype).eps * max(A.shape)
    #честно нагугленная конструкция как я понял, np.finfo возвращате самое маленькое число, при добавлении которого к единице
    #получается результат отличный от единицы, что является мерой точности
    #ну и max(A.shape) это просто максимальное число измерений в массиве

    # да, так функция и реализована в нумпае при rcond=None
        

    l_sigma = sigma > rcond * sigma[0] # "l_sigma" is not accessed 

    #обратная диагональная матрица  
    sigma_inv = np.zeros((VT.shape[0], U.shape[1]))
    indices = np.where(sigma > rcond * sigma[0])[0]

    for i in indices:
        sigma_inv[i, i] = 1.0 / sigma[i]
    # циклы это плохо
    # sigma_inv = np.eye(i) @ np.divide(1, sigma[i])

    A_pinv = np.dot(VT.T, np.dot(sigma_inv, U.T))

    x = np.dot(A_pinv, b)

    r = b - np.dot(A, x)

    return x, r, A_pinv
    # sigma0 = r@r / (b.shape[0] - x.shape[0])
    # var = VT.T @ np.diag(sigma**(-2)) @ VT * sigma0
    # return x, r@r, var


def lstsq(A, b, method, **kwargs):
    if method == 'ne':
        return lstsq_ne(A, b, **kwargs)
    elif method == 'svd':
        return lstsq_svd(A, b, **kwargs)
    else:
        raise ValueError("Неизвестный метод: {}".format(method))
