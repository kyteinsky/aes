from textwrap import wrap

def hex_2_bin(x):
    return str(bin(int(str(x), 16)))[2:]

def int_2_bin(x):
    x = ['{0:08b}'.format(i) for i in x]
    return [list(map(int, list(i))) for i in x]

