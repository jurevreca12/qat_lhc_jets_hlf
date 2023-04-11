import os
import random
import hls4ml

import tensorflow as tf
import numpy as np
import qkeras
from tensorflow.keras import backend as K
from qkeras.utils import print_model_sparsity

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Activation
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning

def train_model(X_train_val, y_train_val, bits):
    model = Sequential()
    model.add(QDense(64, input_shape=(16,), name='fc1',
                     kernel_quantizer=quantized_bits(bits,0,alpha='auto_po2'), use_bias=False,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(QActivation(activation=quantized_relu(bits), name='relu1'))
    model.add(QDense(32, name='fc2',
                     kernel_quantizer=quantized_bits(bits,0,alpha='auto_po2'), use_bias=False,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(QActivation(activation=quantized_relu(bits), name='relu2'))
    model.add(QDense(32, name='fc3',
                     kernel_quantizer=quantized_bits(bits,0,alpha='auto_po2'), use_bias=False,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(QActivation(activation=quantized_relu(bits), name='relu3'))
    model.add(QDense(5, name='output',
                     kernel_quantizer=quantized_bits(bits,0,alpha=1), use_bias=False,
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

    X_train_val = np.load('X_train_val.npy')
    X_test = np.load('X_test.npy')
    y_train_val = np.load('y_train_val.npy')
    y_test = np.load('y_test.npy')
    # classes = np.load('classes.npy', allow_pickle=True)
     

    for bits in [2, 3, 4, 5, 6, 7, 8]:
        model = train_model(X_train_val, y_train_val, bits=bits)
        print(f"-----------------EVALUATING bits:{bits}-----------------")
        loss, acc = model.evaluate(X_test, y_test, verbose=False)
        print(f"Software model accuracy is: {acc}")
        print("---------------------------------------------------------")

        for reuse in [1, 2, 4]:
            print("-----------------GENERATING HLS4ML HARDWARE-----------------")
            hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
            hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
            hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
            config = hls4ml.utils.config_from_keras_model(model, granularity='name')
            config['Model']['ReuseFactor'] = reuse
            config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
            config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
            hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                                   hls_config=config,
                                                                   output_dir=f'hls4ml_reuse{reuse}_bits{bits}',
                                                                   part='xcvu9p-flga2104-2L-e')
            hls_model.compile()

            print("------------------EVALUATING HLS4ML HARDWARE-------------------------")
            y_hls = hls_model.predict(np.ascontiguousarray(X_test))
            from sklearn.metrics import accuracy_score 
            print("Accuracy hls4ml: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))
            hls_model.build(csim=False)
            hls4ml.report.read_vivado_report('hls4ml')

if __name__ == '__main__':
    main()
