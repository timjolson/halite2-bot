
# coding: utf-8

# Notebook to manage keras models (loading, saving, training, etc.)

# In[11]:

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import load_model
import random
from tqdm import tqdm  # progress bars
import numpy as np
import os, subprocess, sys
import copy


# **********
# Add logcosh loss function if using Keras 1

# In[13]:

def logcosh(y_true, y_pred):
    """Sourced from keras version 2
    Logarithm of the hyperbolic cosine of the prediction error.
    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.
    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
    # Returns
        Tensor with one scalar loss entry per sample.
    """
    def _logcosh(x):
        return x + keras.backend.softplus(-2. * x) - keras.backend.log(2.)
    return keras.backend.mean(_logcosh(y_pred - y_true), axis=-1)

try:
    keras.objectives.logcosh = logcosh
except AttributeError:
    pass


# ***
# Generate Models

# In[14]:

general_model_structure = '''
model = Sequential()
model.add(Dense(1024, input_shape=(len(input),)))
model.add(Activation('tanh')) # relu, elu, sigmoid
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(len(output)))
model.add(Activation('softmax'))
'''


# ***
# Load models and their configs

# In[17]:

def load_models(path, extension='.h5'):
    model_names = os.listdir(path)
    model_names = [n for n in model_names if n.endswith(extension)]
    models = []
    model_configs = []
    
    for load_version, model_name in enumerate(tqdm(model_names)):
        models.append(load_model('{}/'.format(path)+model_name))
        model_configs.append(models[-1].get_config())
        
    print('\nLoaded {} models from {}'.format(len(models),path))
    return models, model_names, model_configs


# ***
# Save out models

# In[18]:

def save_models(path, models, model_names=None):
    if not os.path.exists(path):
        os.mkdir(path)
    
    print('Saving {} models to {}'.format(len(models), path))
    if model_names:
        if len(model_names)==len(models):
            for m, n in tqdm(zip(models, model_names)):
                m.save('{}/{}.h5'.format(path, n))
        else:
            raise('Supply a name for each model {} vs {}'.format(len(model_names), len(models)))
    else:
        for idx, m in enumerate(tqdm(models)):
            m.save('{}/{}.h5'.format(path, idx))


# ***
# Filter duplicate configs

# In[19]:

def get_unique(model_configs):
    unique = []
    configs = copy.copy(model_configs)
    for config in tqdm(configs):
        configs.remove(config)
        compare = [config == c for c in configs]
        if not np.sum(compare):
            unique.append(config)
        
    print('\n{} unique model configs'.format(len(unique)))
    return unique


# *******************
# Compile models from configs

# In[20]:

def compile_from_configs(model_configs, savepath=None, losses=['logcosh'], optimizers=['adagrad'], metrics=['accuracy']):
    models = []
    
    for loss in losses:
        for opt in optimizers:
            for idx, config in enumerate(tqdm(model_configs)):
                model = keras.Sequential.from_config(config)
                model.compile(loss=loss, optimizer=opt, metrics=metrics)
                models.append(model)
                if savepath:
                    if not os.path.exists(savepath):
                        os.mkdir(savepath)
                    model.save('{}/{}-{}-{}.h5'.format(path, loss, optimizer, idx))
                    
    return models


# ***
# Compile models

# In[21]:

def compile_models(models, savepath=None, losses=['logcosh'], optimizers=['adagrad'], metrics=['accuracy']):
    _models = []
    
    for loss in losses:
        for opt in optimizers:
            for idx, config in enumerate(tqdm(models)):
                model = keras.Sequential.from_config(config)
                model.compile(loss=loss, optimizer=opt, metrics=metrics)
                _models.append(model)
                if savepath:
                    if not os.path.exists(savepath):
                        os.mkdir(savepath)
                    model.save('{}/{}-{}-{}.h5'.format(path, loss, optimizer, idx))
                    
    return _models


# ***
# Load in training data from 'trainN.in' and 'trainN.out' (data formatted as lists or numpy arrays), for N in nums

# In[22]:

def load_data_from_path(path, nums=[]):
    if not isinstance(nums, list):
        nums = list(nums)
    
    input, output = [], []
    
    for num in nums:
        print("Reading train{}.in".format(num))
        with open('{}/train{}.in'.format(path, num), "r") as f:
            train_in = f.read().split('\n')
            train_in = [np.fromstring(i.lstrip('[]').replace(',',' '),sep=' ') for i in tqdm(train_in[:-1])]

        print("Reading train{}.out".format(num))
        with open('{}/train{}.out'.format(path, num), "r") as f:
            train_out = f.read().split('\n')
            train_out = [np.fromstring(i.lstrip('[]').replace(',',' '),sep=' ') for i in tqdm(train_out[:-1])]
        
        if len(train_in) != len(train_out):
            print('train{}.in'.format(num), len(train_in))
            print('train{}.out'.format(num), len(train_out))
            raise('lengths do not match')
        
        print('Adding train{}.in'.format(num))
        for line in tqdm(train_in):
            input.append(line)
        print('Adding train{}.out'.format(num))
        for line in tqdm(train_out):
            output.append(line)
            
    return np.array(input), np.array(output)


# ***
# Load in training data from numpy saved arrays

# In[23]:

def load_data_numpy(path, input_file='train_in.npy', output_file='train_out.npy'):
    print('Loading from {}'.format(path))
    train_in = np.load('{}/{}'.format(path, input_file))
    train_out = np.load('{}/{}'.format(path, output_file))
    
    print('input {}, length {}'.format(input_file), len(train_in))
    print('output {}, length {}'.format(output_file), len(train_out))
    if len(train_in) != len(train_out):
        raise('lengths do not match')
    
    return input, output


# ***
# Shuffle data

# In[24]:

def shuffle(input, output):
    data = list(zip(input, output))
    random.shuffle(data)
    train_in, train_out = [],[]
    
    for line in tqdm(data):
        train_in.append(line[0])
        train_out.append(line[1])
        
    return np.array(train_in), np.array(train_out)


# ***
# Split data into training and testing groups

# In[25]:

def split_data(train_in, train_out, test_ratio):
    test_size = int(test_ratio * len(train_in))

    x_train = train_in[:-test_size]
    y_train = train_out[:-test_size]

    x_test = train_in[-test_size:]
    y_test = train_out[-test_size:]

    print('Training on {} samples, testing on {} samples'.format(len(train_in), test_size))
    return x_train, y_train, x_test, y_test


# ***
# Train models

# In[26]:

def train_models(models, x_train, y_train, epochs=5, batch_size=2**15, verbose=0, validation_split=0.08):
    histories = []
    
    print('Training on batches of {} for {} epochs'.format(batch_size, epochs))
    
    for idx, model in enumerate(tqdm(models)):
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            validation_split=validation_split)

        histories.append(history)

    return models


# Evaluate models

# In[27]:

def eval_models(models, x_test, y_test, batch_size=2**15, verbose=0):
    scores = []
    
    for idx, model in enumerate(tqdm(models)):
        score = model.evaluate(x_test, y_test,
                               batch_size=batch_size, verbose=verbose)
        
        scores.append(score)
        print('Model {} Score: {}'.format(idx, score))
        
    for s in scores:
        yield s


# Train and eval

# In[28]:

def train_and_eval(models, x_train, y_train, x_test, y_test, epochs=5, batch_size=2**15, verbose=0, validation_split=0.08):
    print('Training and evaluating models')
    _models, scores = [], []
    for model in tqdm(models):
        _models.append(train_models(model, x_train, y_train, epochs, batch_size, verbose, validation_split))
        scores.append(eval_models(model, x_test, y_test, batch_size, verbose))
    
    return _models, scores


# In[ ]:

