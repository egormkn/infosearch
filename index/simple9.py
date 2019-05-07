# coding: utf-8


def encode(int_index):
    byte_index = bytearray()
    for (i, number) in enumerate(int_index):
        if i > 0:
            number -= int_index[i - 1]
        number_bytes = []
        if number == 0:
            number_bytes.append(0)
        while number != 0:
            number_bytes.append(number & 0b01111111)
            number >>= 7
        number_bytes[0] |= 0b10000000
        byte_index.extend(reversed(number_bytes))
    return bytes(byte_index)


def decode(byte_index):
    int_index = []
    number = 0
    for byte in byte_index:
        byte = ord(byte)
        number <<= 7
        number |= byte & 0b01111111
        if byte & 0b10000000:
            int_index.append(number)
            number = 0
    for (i, number) in enumerate(int_index):
        if i > 0:
            int_index[i] += int_index[i - 1]
    return int_index
