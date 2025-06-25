import numpy as np

def generate_data(nrows,ncols,nrealizations,variance,seed): #скармливаем параметры

    np.random.seed(seed) 
    
    A = np.random.rand(nrows, ncols)
    x = np.random.rand(ncols, 1)

    #print(x)
    mean_b = A @ x

    B = np.random.multivariate_normal(mean_b.flatten(), np.eye(nrows) * variance, nrealizations)

    return A, x, B