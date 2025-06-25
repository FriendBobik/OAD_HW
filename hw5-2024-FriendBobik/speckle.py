from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import scipy.ndimage
from scipy.ndimage import rotate


data = fits.open('speckledata.fits')[2].data
#print(data.shape)
'''(101, 200, 200) всего 101 кадр, каждый кадр 200x200 пикселей'''


# '''Посмотрим с чем мы вообще работаем'''
# plt.imshow(data[0])
# plt.colorbar()  
# plt.show()  


mean_image = np.mean(data, axis=0)
plt.imshow(mean_image,vmax=np.quantile(mean_image, 0.95))
plt.colorbar()  
plt.savefig('mean.png') 
plt.title('Среднее изображение')
plt.show()  
# '''Получилось изображение, на котором видно, что в центре яркое пятно, а по краям темнее, отлично совпадате с примером'''


# power_spectra = []
# for image in data:
#     fft_image = scipy.fft.fftshift(scipy.fft.fft2(image)) # Двумерное преобразование Фурье
#     power_spectra.append(np.abs(fft_image) ** 2)
#mean_power_spectrum = np.mean(power_spectra, axis=0)

fft_images = scipy.fft.fftshift(scipy.fft.fft2(data, axes=(-2, -1)), axes=(-2, -1)) # менее колхозный способ, без цикла, О-оптимизация

mean_power_spectrum = np.mean(np.abs(fft_images)**2, axis=0)


plt.imshow(mean_power_spectrum,vmax=np.quantile(mean_power_spectrum, 0.95))
plt.colorbar()
plt.savefig('fourier.png')
plt.title('Средняя мощность спектра')
plt.show() 


'''Создаем маску для выделения шума'''
y, x = np.indices(mean_power_spectrum.shape)
center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
radius = 50
distance_from_center = (x - center[0])**2 + (y - center[1])**2
mask = distance_from_center <= radius**2
masked_spectrum = np.ma.masked_array(mean_power_spectrum, mask)

'''Проверяем маску'''
# plt.imshow(masked_spectrum)
# plt.colorbar()
# plt.show()
'''Всё ок'''

noise_level = np.mean(masked_spectrum)
# print(noise_level)

fft_images_minus_noiz=fft_images-noise_level
fft_images_minus_noiz_mean = mean_power_spectrum - noise_level

plt.imshow(fft_images_minus_noiz_mean,vmax=np.quantile(fft_images_minus_noiz_mean, 0.95))
plt.colorbar()
plt.title('Средняя мощность спектра с вычетом шума')
plt.savefig('fourier-noise.png')
plt.show()


angles = np.linspace(0, 360, num=360) # Сетка углов
averaged_rotated_spectrum = np.zeros_like(fft_images_minus_noiz_mean)

# Поворот изображения на каждый угол и усреднение
for angle in angles:
    rotated_image = rotate(fft_images_minus_noiz_mean, angle, reshape=False, order=1)
    averaged_rotated_spectrum += rotated_image

averaged_rotated_spectrum /= len(angles)


plt.imshow(averaged_rotated_spectrum,vmax=np.quantile(averaged_rotated_spectrum, 0.95))
plt.colorbar()
plt.title('Усредненный Фурье-образ по углам') # тут мы уже работаем с вычемом шума, из прошлого графика
plt.savefig('rotaver.png')
plt.show()



mask_high_freq = distance_from_center > 75**2
data_3=np.divide(averaged_rotated_spectrum,mean_power_spectrum)
# plt.imshow(data_3)
# plt.colorbar()
# plt.show()
# Применение маски
filtered_fft_images = np.where(mask_high_freq, 0, data_3)
'''Проверяем маску'''
# plt.imshow(np.abs(filtered_fft_images))
# plt.colorbar()
# plt.show()
'''Опять же, всё ок'''

# Обратное Фурье-преобразование для получения очищенного изображения

filtered_images = scipy.fft.ifftshift(scipy.fft.ifft2(filtered_fft_images, axes=(-2, -1)))
filtered_images_real = np.abs(filtered_images)


plt.imshow(filtered_images_real, vmax= 0.07)
plt.colorbar()
plt.title('Востановленое изображение')
plt.savefig('binary.png')
plt.show()

# Можно конечно накинуть зум, но мне лень