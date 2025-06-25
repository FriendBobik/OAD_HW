import serial
import time

# Замените 'COM_PORT' на соответствующий порт вашего Arduino
ser = serial.Serial('/dev/tty.usbserial-1110', 9600)

while True:
    current_time = int(time.time() * 10)  # Умножаем, чтобы получить десятые доли секунды
    ser.write((str(current_time) + '\n').encode())  # Отправка времени на Arduino
    time.sleep(1)
