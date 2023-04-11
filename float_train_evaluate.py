import os
import random

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Activation, Dense

from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning

def train_model(X_train_val, y_train_val):
    model = Sequential()
    model.add(Dense(64, input_shape=(16,), name='fc1', use_bias=False,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu1'))
    model.add(Dense(32, name='fc2', use_bias=False,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu2'))
    model.add(Dense(32, name='fc3', use_bias=False,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu3'))
    model.add(Dense(5, name='output', use_bias=False,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='softmax', name='softmax'))

    pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)}
    model = prune.prune_low_magnitude(model, **pruning_params)
    
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    model.fit(X_train_val, y_train_val, batch_size=1024,
              epochs=30, validation_split=0.25, shuffle=True,
              callbacks = [pruning_callbacks.UpdatePruningStep()])
    
    model = strip_pruning(model)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    return model


def main():
    # Setting the random seeds for reproducibility
    # see: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    file_path = os.path.realpath(__file__)                                                                                                                                                               
    dir_path = os.path.dirname(file_path)

    X_train_val = np.load('X_train_val.npy')
    X_test = np.load('X_test.npy')
    y_train_val = np.load('y_train_val.npy')
    y_test = np.load('y_test.npy')
    # classes = np.load('classes.npy', allow_pickle=True)
     
    model = train_model(X_train_val, y_train_val)

    print("-----------------EVALUATING-----------------")
    loss, acc = model.evaluate(X_test, y_test, verbose=False)
    print(f"Software model accuracy is: {acc}")
    print("--------------------------------------------")
    
    with open(os.path.join(dir_path, 'float', 'results.txt'), 'w') as f:                                                                                                                 
        f.write(f"Float acc:{acc}")

if __name__ == '__main__':
    main()
