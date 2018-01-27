import random
from tqdm import tqdm
import numpy as np
import os, subprocess

input, output = [], []

#subprocess.Popen('backup.bat')

for i in range(200):
    num = str(i)
    if os.path.isfile('train{}.in'.format(num)) and os.path.isfile('train{}.out'.format(num)):
        print("Reading input")
        with open('train{}.in'.format(num), "r") as f:
            train_in = f.read().split('\n')
            train_in = [eval(i) for i in tqdm(train_in[:-1])]
            print("done train in")

        print("Reading output")
        with open('train{}.out'.format(num), "r") as f:
            train_out = f.read().split('\n')
            train_out = [np.fromstring(i.lstrip('[]').replace(',',''),sep=' ') for i in tqdm(train_out[:-1])]
            print("done train out")

        if len(train_in) != len(train_out):
            print('train{}.in'.format(num), len(train_in))
            print('train{}.out'.format(num), len(train_out))
            raise('lengths do not match')
            
        print('adding train{}.in'.format(num))
        for line in tqdm(train_in):
            input.append(line)
        print('adding train{}.out'.format(num))
        for line in tqdm(train_out):
            output.append(line)

#if os.path.isfile('train_in.npy') and os.path.isfile('train_out.npy'):
#    train_in = np.load('train_in.npy')
#    train_out = np.load('train_out.npy')
    
#    if len(train_in) != len(train_out):
#        print('train_in.npy:', len(train_in))
#        print('train_out.npy:', len(train_out))
#        raise('lengths do not match')
    
#    print('adding train_in.npy')
#    for line in tqdm(train_in):
#        input.append(line)
#    print('adding train_out.npy')
#    for line in tqdm(train_out):
#        output.append(line)

if input and output:
    print("shuffling data")
    data = list(zip(input, output))
    random.shuffle(data)
    train_in, train_out = [],[]
    print("combining data")
    for line in tqdm(data):
        train_in.append(line[0])
        train_out.append(line[1])

    print("saving data")
    np.save("train_in.npy", train_in)
    np.save("train_out.npy", train_out)
