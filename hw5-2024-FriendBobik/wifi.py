import argparse
import numpy as np


def read(filename):
    with open(filename, "r") as reader:
        data = list(map(float, reader.readlines()))
    return data


# +1 +1 +1 −1 −1 −1 +1 −1 −1 +1 −1 = 1
# -1 -1 -1 +1 +1 +1 -1 +1 +1 -1 +1 = 0
def decode_bit(b_arr: list):
    original_b_1 = [+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1]
    original_b_0 = [-1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1]

    b_1 = []
    b_0 = []

    for i in range(11):
        for _ in range(5):
            b_1.append(original_b_1[i])
            b_0.append(original_b_0[i])

    corr_1, corr_0 = 0, 0
    for i in range(55):
        corr_1 += int(b_arr[i] == b_1[i])
        corr_0 += int(b_arr[i] == b_0[i])
    #выбираем бит, который больше всего похож на оригинальный
    if corr_1 > corr_0:
        return 1
    return 0


# Функция декодирования отдельной части данных
def decode_part(part: list):
    assert len(part) == 55
    decoded = [0] * 55
    for i in range(55):
        if part[i] > 0:
            decoded[i] = 1
        else:
            decoded[i] = 0
    
    return decode_bit(decoded)


# Функция декодирования всего массива данных
def decode(data: list):
    decoded = []
    for i in range(0, len(data), 55):
        decoded.append(decode_part(data[i:i+55]))
    return decoded


# Функция преобразования битов в строку
def bits_to_str(bits):
    data = ''
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        data += chr(int(''.join(map(str, byte)), 2))
    return data

# Функция сохранения данных в json
def save_json(data):
    import json
    with open('wifi.json', 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', metavar='FILENAME', type=str, help="Path to the input file with encoded data")
    args = parser.parse_args()
    data = read(args.input_file)
    decoded_data = bits_to_str(decode(data))
    #print(decoded_data)
    save_json({'message': decoded_data})

