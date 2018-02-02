import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
from tqdm import tqdm  # progress bars
from numpy import concatenate

# Generate Models
model_structure = '''
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

# Add logcosh to Keras v1
if hasattr(keras, 'objective'):
    # Add logcosh loss function if using Keras 1
    def logcosh(y_true, y_pred):
        """Taken from Keras v2
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
    
    keras.objectives.logcosh = logcosh
    

# Load models and their configs
def load_models(path, extension='.h5'):
    from keras.models import load_model
    from os import listdir
    model_names = listdir(path)
    model_names = [n.replace(extension,'') for n in model_names if n.endswith(extension)]
    models = []
    model_configs = []
    
    tqdm.write('Loading models from {}'.format(path))
    
    for load_version, model_name in enumerate(tqdm(model_names)):
        models.append(load_model('{}/{}{}'.format(path,model_name,extension)))
        model_configs.append(models[-1].get_config())
        
    return models, model_names, model_configs


# Save out models
def save_models(path, models, model_names=None, extension='.h5'):
    import os
    from keras import Model
    if not os.path.exists(path):
        os.mkdir(path)
    
    tqdm.write('Saving {} models to {}'.format(len(models),path))
    
    if model_names:
        if len(model_names)!=len(models):
            raise('Supply a name for each model {} vs {}'.format(len(model_names), len(models)))
    else:
        model_names = [*range(len(models))]
    
    for m, n in tqdm(zip(models, model_names)):
        if not isinstance(m, Model):
            m = Sequential.from_config(m)
        m.save('{}/{}{}'.format(path, n, extension))
    
    return models, model_names


# Filter duplicate configs
def get_unique_configs(models):
    from copy import copy
    from keras.models import Sequential
    from keras import Model
    unique = []
    
    tqdm.write('Comparing {} configs'.format(len(models)))
    
    model_configs = [Sequential.from_config(model.get_config()) \
                     if isinstance(model, Model) else copy(model) for model in models]
    
    configs = copy(model_configs)
    for config in tqdm(model_configs):
        while True:
            try:
                configs.remove(config)
            except ValueError:
                break
        unique.append(config)
    
    tqdm.write('{} unique configs'.format(len(unique)))
    return unique


# Compile models
def compile_models(models, savepath=None, losses=['logcosh'], optimizers=['adagrad'], metrics=['accuracy']):
    import os
    from copy import copy
    _models = []
    
    tqdm.write('Compiling {} models'.format(len(models)))
    
    model_configs = [model.get_config() if isinstance(model, keras.Model) else copy(model) for model in models]
    
    for loss in tqdm(losses):
        for opt in tqdm(optimizers):
            for idx, config in enumerate(tqdm(model_configs)):
                model = keras.Sequential.from_config(config)
                model.compile(loss=loss, optimizer=opt, metrics=metrics)
                _models.append(model)
                if savepath:
                    if not os.path.exists(savepath):
                        os.mkdir(savepath)
                    model.save('{}/{}-{}-{}.h5'.format(savepath, loss, opt, idx))
                    
    return _models


# Load in training data from 'trainN.in' and 'trainN.out' (data formatted as lists or numpy arrays), for N in nums
def load_data_from_path(path, nums=[]):
    from numpy import fromstring, array
    if not isinstance(nums, list):
        nums = list(nums)
    
    input, output = [], []
    
    tqdm.write('Loading {} data sets from {}'.format(nums, path))
    
    for num in tqdm(nums):
        tqdm.write("Reading train{}.in".format(num))
        with open('{}/train{}.in'.format(path, num), "r") as f:
            train_in = f.read().split('\n')
            train_in = [fromstring(i.lstrip('[]').replace(',',' '),sep=' ') for i in tqdm(train_in[:-1])]

        tqdm.write("Reading train{}.out".format(num))
        with open('{}/train{}.out'.format(path, num), "r") as f:
            train_out = f.read().split('\n')
            train_out = [fromstring(i.lstrip('[]').replace(',',' '),sep=' ') for i in tqdm(train_out[:-1])]
        
        if len(train_in) != len(train_out):
            tqdm.write('train{}.in'.format(num), len(train_in))
            tqdm.write('train{}.out'.format(num), len(train_out))
            raise('lengths do not match')
        
        tqdm.write('Adding train{}.in'.format(num))
        for line in tqdm(train_in):
            input.append(line)
        tqdm.write('Adding train{}.out'.format(num))
        for line in tqdm(train_out):
            output.append(line)
            
    return array(input), array(output)


# Load generic data
def load_data(path, inputfile, outputfile):
    from numpy import fromstring, array
    input, output = [], []
    
    tqdm.write('Loading data set from {} // {}'.format(inputfile, outputfile))
    
    tqdm.write("Reading {}".format(inputfile))
    with open('{}/{}'.format(path, inputfile), "r") as f:
        train_in = f.read().split('\n')
        train_in = [fromstring(i.lstrip('[]').replace(',',' '),sep=' ') for i in tqdm(train_in[:-1])]

    tqdm.write("Reading {}".format(outputfile))
    with open('{}/{}'.format(path, outputfile), "r") as f:
        train_out = f.read().split('\n')
        train_out = [fromstring(i.lstrip('[]').replace(',',' '),sep=' ') for i in tqdm(train_out[:-1])]

    if len(train_in) != len(train_out):
        raise('lengths do not match')

    tqdm.write('Adding input')
    for line in tqdm(train_in):
        input.append(line)
    tqdm.write('Adding output')
    for line in tqdm(train_out):
        output.append(line)
    
    return array(input), array(output)


# Load in training data from numpy saved arrays
def load_data_numpy(path, inputfile='train_in.npy', outputfile='train_out.npy'):
    from numpy import load
    tqdm.write('Loading numpy arrays {} // {} from {}'.format(inputfile, outputfile, path))
    train_in = load('{}/{}'.format(path, inputfile))
    train_out = load('{}/{}'.format(path, outputfile))
    
    tqdm.write('input {}, length {}'.format(inputfile, len(train_in)))
    tqdm.write('output {}, length {}'.format(outputfile, len(train_out)))
    if len(train_in) != len(train_out):
        raise('lengths do not match')
    
    return train_in, train_out


# Shuffle data, keeping element association between input and output
def shuffle(input, output):
    import random
    from numpy import array
    
    tqdm.write('Shuffling data')
    
    data = list(zip(input, output))
    random.shuffle(data)
    train_in, train_out = [],[]
    
    for line in tqdm(data):
        train_in.append(line[0])
        train_out.append(line[1])
        
    return array(train_in), array(train_out)


# Save out data
def save_data(train_in, train_out, inputfilename='train_in', outputfilename='train_out'):
    from numpy import save
    
    tqdm.write('Saving out samples')
    
    if len(train_in)==len(train_out):
        save(inputfilename, train_in)
        save(outputfilename, train_out)
    else:
        raise('Data lengths do not match')


# Split data into training and testing groups
def split_data(train_in, train_out, test_ratio=0.05):
    test_size = int(test_ratio * len(train_in))
    
    tqdm.write('Splitting data')
    
    x_train = train_in[:-test_size]
    y_train = train_out[:-test_size]

    x_test = train_in[-test_size:]
    y_test = train_out[-test_size:]

    tqdm.write('Training on {} samples, testing on {} samples'.format(len(train_in), test_size))
    return x_train, y_train, x_test, y_test


# Train models
def train_models(models, x_train, y_train, epochs=5, batch_size=2**15, stats=False, verbose=0, validation_split=0.05):
    histories = []
    
    if stats:
        tqdm.write('Training on batches of {} for {} epochs'.format(batch_size, epochs))
    
    for idx, model in enumerate(tqdm(models)):
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            validation_split=validation_split)

        histories.append(history)

    return models


# Evaluate models
def eval_models(models, x_test, y_test, batch_size=2**15, stats=False, verbose=0):
    scores = []
    
    if stats:
        tqdm.write('Evaluating models')
    
    for idx, model in enumerate(tqdm(models)):
        score = model.evaluate(x_test, y_test,
                               batch_size=batch_size, verbose=verbose)
        
        scores.append(score)
        if stats:
            tqdm.write('Model {} Score: {}'.format(idx, score))
        
    return scores


# Train and eval
def train_and_eval(models, x_train, y_train, x_test, y_test, epochs=5, batch_size=2**15, stats=False, verbose=0, validation_split=0.08):
    _models, scores = [], []
    tqdm.write('Training and evaluating models')
    
    _models.append(train_models(models, x_train, y_train, epochs, batch_size, \
                                stats, verbose, validation_split))
    
    scores.append(eval_models(models, x_test, y_test, batch_size, stats, verbose))
    
    return _models, scores

