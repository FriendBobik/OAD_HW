#!/usr/bin/env python3

import argparse
import netCDF4 as nc 
'''Тут надо пояснить на семинаре мы использовами from scipy.io import netcdf
Однако я поуглил и нашел, сильно более понятную документацию по netCDF4 https://unidata.github.io/netcdf4-python/'''
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('location', nargs='+') # Парсинг аргументов из командной строки на вход принимает или 1 или 2 значения 2 значения это широта и долгота 1 это название места



"""Потыкав данные я понял что данные по широте и долготе идут с шагом в 0.5
Так как на вход могут подавться не целые числа, то надо будет найти ближайшие значения к введенным
Напишем функцию для этого. По хорошему функции надо было бы вынести в отдельный файл, но это не подразумеваеться ТЗ"""
def find_nearest_index(value, value_array):
    index = (np.abs(value_array - value)).argmin()
    return index



# Функция для расчета мин макс сред
def calculate_min_max_mean(ozone_data):
    return {
        "min": float(np.min(ozone_data)),
        "max": float(np.max(ozone_data)),
        "mean": float(np.mean(ozone_data))
    }


#Чтобы не обрабатывать перескоки через года в ручную, воспользуемся первой же нагугленной библиотекой
#Даааа, мозможно не самое оптимальное решение, но выглядит не сложно и работает

from datetime import datetime
from dateutil.relativedelta import relativedelta
def add_months_to_base_date(months):
    base_date = datetime(1960, 1, 15)  # Базовая дата полученная из файла запросом print(data.variables['time'])
    new_date = base_date + relativedelta(months=+months)
    return new_date

if __name__ == "__main__":
    args = parser.parse_args() # Парсим аргументы из командной строки
    if len(args.location) == 2:  # Предполагаем, что введены широта и долгота
        try:
            longitude = float(args.location[0])
            latitude = float(args.location[1])
        except ValueError:
            raise ValueError("Некорректный формат координат. Укажите два числа для широты и долготы.")
    else:  # Предполагаем, что введено название места
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="earth")
        location = geolocator.geocode(args.location)
        latitude = location.latitude
        longitude = location.longitude
    
    # Проверка введённых координат на соответствие допустимому диапазону
    if not (-90 <= longitude <= 90) or not (-180 <= latitude <= 180):
        print("Введены некорректные координаты. Широта должна быть в диапазоне от -90 до +90, долгота от -180 до +180.\n  Обязательно к ознакомлению https://www.youtube.com/watch?v=dUNR_25o6RA Не банте пжжпжпжп, это шутка")

    else:
        data = nc.Dataset('MSR-2.nc', mode='r') 

        latitudes = data.variables['latitude'][:]
        longitudes = data.variables['longitude'][:]
        times = data.variables['time'][:]
        ozone = data.variables['Average_O3_column'][:]

        dates = [add_months_to_base_date(int(month)) for month in times]  # Преобразование временных значений в объекты datetime

        #for date in dates:
        #    print(date)

        # Нахождение ближайших индексов
        lat_idx = find_nearest_index(latitude, latitudes)
        lon_idx = find_nearest_index(longitude, longitudes)

        #print(lat_idx, lat_idx)
        
        ozone_data = ozone[:, lat_idx, lon_idx] # Извлечение данных о содержании озона для координат
        #print(ozone_data)

        jan_ozone = np.array([ozone_data[i] for i, date in enumerate(dates) if date.month == 1])
        jul_ozone = np.array([ozone_data[i] for i, date in enumerate(dates) if date.month == 7])
        #print(jan_ozone)

        # Расчет статистик для всех данных, января и июля
        metrics_all = calculate_min_max_mean(ozone_data)
        metrics_jan = calculate_min_max_mean(jan_ozone)
        metrics_jul = calculate_min_max_mean(jul_ozone)

        #Чтобы график был нагляднее, я решил усреднить данные по годам
        #Просто если не усреднять то это просто случайней точки
        average_ozone = []
        years = []

        for i in range(0, len(ozone_data), 12):
            if i + 11 < len(ozone_data):  
                avg = sum(ozone_data[i:i+12]) / 12
                average_ozone.append(avg)
        
                years.append(1960 + i // 12)

        datetime_years = []
        #Так как у нас получиолся просто год, то переведем это в datetime, чтобы график построился нормально
        # Преобразование каждого года в datetime с фиксированным месяцем и днём (15 июня)
        for year in years:
            date = datetime(year, 6, 15)  # 15 июня в теории это где-то середина года
            datetime_years.append(date)

        # Вот тут я понял, что переходить в datetime было не лучшим решением, но уже сделал, а переписывать не хочу
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(datetime_years, average_ozone, label='Все данные', color='blue', linewidth=1, linestyle='-')
        
        jan_dates = [dates[i] for i in range(len(dates)) if dates[i].month == 1]
        plt.plot(jan_dates, jan_ozone, label='Январь', color='green', linewidth=1, linestyle='-')

        jul_dates = [dates[i] for i in range(len(dates)) if dates[i].month == 7]
        plt.plot(jul_dates, jul_ozone, label='Июль', color='red', linewidth=1, linestyle='-')

        plt.title('Зависимость содержания озона от времени')
        plt.xlabel('Время')
        plt.ylabel('Содержание озона')
        plt.legend()
        plt.grid(True)

        plt.gcf().autofmt_xdate() #Формат для даты

        plt.savefig('ozon.png')
        plt.close()

        # Формирование для сохранения в JSON
        results = {
            "coordinates": [longitude, latitude],
            "jan": metrics_jan,
            "jul": metrics_jul,
            "all": metrics_all
        }

        with open('ozon.json', 'w') as json_file:
            json.dump(results, json_file, indent=4)
        data.close()

    
