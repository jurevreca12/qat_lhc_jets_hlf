import os
import sys
import time
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


def main(bits=None):
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
     
    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)

    model = train_model(X_train_val, y_train_val, bits=bits)
    print(f"-----------------EVALUATING bits:{bits}-----------------")
    loss, acc = model.evaluate(X_test, y_test, verbose=False)
    print(f"Software model accuracy is: {acc}")
    print("---------------------------------------------------------")

    print("-----------------GENERATING HLS4ML HARDWARE-----------------")
    stime = time.time()
    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
    config = hls4ml.utils.config_from_keras_model(model, 
                                                  granularity='name', 
                                                  default_reuse_factor=1,
                                                  default_precision='ap_fixed<11,6>')
    config['Model']['ReuseFactor'] = 1
    config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
    config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           hls_config=config,
                                                           output_dir=f'hls4ml_bits{bits}',
                                                           part='xcvu9p-flga2104-2L-e')
    hls_model.compile()
    hls_model.build(csim=False) #, vsynth=True)
    hls4ml.report.read_vivado_report('hls4ml')
    etime = time.time()

    print("------------------EVALUATING HLS4ML HARDWARE-------------------------")
    y_hls = hls_model.predict(np.ascontiguousarray(X_test))
    from sklearn.metrics import accuracy_score 
    hls_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))
    print("Accuracy hls4ml: {}".format(hls_acc))
    with open(os.path.join(dir_path, f'hls4ml_bits{bits}', 'results.txt'), 'w') as f:
        f.write(f"Qkeras acc:{acc}\n"
                f"hls4ml acc:{hls_acc}\n"
                f"gen time: {etime-stime} seconds.")


if __name__ == '__main__':
    assert len(sys.argv) == 2                                                                                                                                                                                                                                                       
    main(bits=int(sys.argv[1]))
