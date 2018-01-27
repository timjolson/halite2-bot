import random
from tqdm import tqdm
import numpy as np
import os, subprocess
import time

input, output = [], []

subprocess.Popen('backup.bat').wait()

time.sleep(1)

if os.path.isfile('train.in') and os.path.isfile('train.out'):
    print("Reading input")
    with open('train.in', "r") as f:
        train_in = f.read().split('\n')
        train_in = [eval(i) for i in tqdm(train_in[:-1])]
        print("done train in")

    print("Reading output")
    with open('train.out', "r") as f:
        train_out = f.read().split('\n')
        train_out = [np.fromstring(i.lstrip('[]').replace(',',''),sep=' ') for i in tqdm(train_out[:-1])]
        print("done train out")

    if len(train_in) != len(train_out):
        print('train.in', len(train_in))
        print('train.out', len(train_out))
        raise('lengths do not match')
        
    print('adding train.in')
    for line in tqdm(train_in):
        input.append(line)
    print('adding train.out')
    for line in tqdm(train_out):
        output.append(line)

if os.path.isfile('train_in.npy') and os.path.isfile('train_out.npy'):
    print('Loading previous data')
    train_in = np.load('train_in.npy')
    train_out = np.load('train_out.npy')
    
    if len(train_in) != len(train_out):
        print('train_in.npy:', len(train_in))
        print('train_out.npy:', len(train_out))
        raise('lengths do not match')
   
    print('adding train_in.npy')
    for line in tqdm(train_in):
        input.append(line)
    print('adding train_out.npy')
    for line in tqdm(train_out):
        output.append(line)

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

num = os.listdir('.')
num.remove('add_new_data.py')
num.remove('backup.bat')
num.remove('combine_all_data.py')
num.remove('train_in.npy')
num.remove('train_out.npy')
num = int(len(num)/2)

os.rename('train.in', 'train{}.in'.format(num))
os.rename('train.out', 'train{}.out'.format(num))