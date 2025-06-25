# # Установите библиотеку NetCDF4, если она еще не установлена
# pip install netCDF4

import netCDF4 as nc

# Откройте файл .nc
dataset = nc.Dataset('/Users/aboba/Desktop/OAD/seminars/numpy/tos_O1_2001-2002.nc')

# Выведите список переменных, содержащихся в файле
print(dataset.variables.keys())

# Доступ к переменной (например, 'time') и её данным
latitude = dataset.variables['tos'][:]
print(latitude)

# Закройте файл после завершения работы
dataset.close()

# Для доступа к переменным используйте ds.variables['имя_переменной']


# from scipy.io import netcdf
# import numpy as np

# with netcdf.netcdf_file('MSR-2.nc', mmap=False) as netcdf_file:
#     print('\n',"Dimension: {}".format(netcdf_file.dimensions, '\n'))
#     variables = netcdf_file.variables
#     for v in variables:
#         var = variables[v]
#         print("Variable {} dims {} shape {}".format(v, var.dimensions, var.data.shape))
# print('\n','Единица измерения времени: ', variables['time'].units, '\n')

# time_index = np.searchsorted(variables['time'].data, 200)
# lat_index = np.searchsorted(variables['longitude'].data, 75)
# lon_index = np.searchsorted(variables['latitude'].data, 75)
# print(variables['Average_O3_column'][time_index, lat_index, lon_index])