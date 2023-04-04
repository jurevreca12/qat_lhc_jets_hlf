import os
import tensorflow as tf
import numpy as np
import qkeras

from tensorflow.keras.datasets import mnist
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

def main():
    X_train_val = np.load('X_train_val.npy')
    X_test = np.load('X_test.npy')
    y_train_val = np.load('y_train_val.npy')
    y_test = np.load('y_test.npy')
    classes = np.load('classes.npy', allow_pickle=True)
    
    # We rescale the inputs so that we can quantize them as signed integers.
    X_train_val = X_train_val * (2**6)
    X_test = X_test * (2**6)
    X_train_val = np.round(X_train_val)
    X_test = np.round(X_test)
    
    
    alpha = 'auto_po2'
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(16,)))
    model.add(qkeras.QActivation(qkeras.quantized_bits(bits=11, integer=10, keep_negative=True, alpha=1)))
    model.add(QDense(64, input_shape=(16,), name='fc1',
                     kernel_quantizer=quantized_bits(6,5,alpha=alpha, keep_negative=True), use_bias=False, 
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(QActivation(activation=quantized_relu(5,5), name='relu1'))
    model.add(QDense(32, name='fc2',
                     kernel_quantizer=quantized_bits(6,5,alpha=alpha, keep_negative=True), use_bias=False,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(QActivation(activation=quantized_relu(5,5), name='relu2'))
    model.add(QDense(32, name='fc3',
                     kernel_quantizer=quantized_bits(6,5,alpha=alpha, keep_negative=True), use_bias=False,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(QActivation(activation=quantized_relu(5,5), name='relu3'))
    model.add(QDense(5, name='output',
                     kernel_quantizer=quantized_bits(6,5,alpha=alpha, keep_negative=True), use_bias=False,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), activation='softmax'))
    
    pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)}
    model = prune.prune_low_magnitude(model, **pruning_params)
    
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    model.fit(X_train_val, y_train_val, batch_size=1024,
                  epochs=30, validation_split=0.25, shuffle=True,
                  callbacks = [pruning_callbacks.UpdatePruningStep()])
    #model.save_weights('cern_test.h5')
    #model.load_weights('cern_test.h5')
    
    model = strip_pruning(model)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    
    print("\n")
    print("-----------------EVALUATING-----------------")
    loss, acc = model.evaluate(X_test, y_test, verbose=False)
    print(f"Software model accuracy is: {acc}")
    print("--------------------------------------------")
    
    
    print("\n")
    print("-----------------GENERATING VERILOG WITH CHISEL4ML-----------------")
    from chisel4ml import optimize, generate
    opt_model = optimize.qkeras_model(model)
    circuit = generate.circuit(opt_model, is_simple=True, use_verilator=True)
    file_path = os.path.realpath(__file__)                                                                                  
    dir_path = os.path.dirname(file_path)  
    circuit.package(directory=os.path.join(dir_path, 'chisel4ml'), name='ProcessingPipeline')
    print("-------------------------------------------------------------------")
    
    print("\n")
    print("-----------------EVALUATING CIRCUIT VIA SIMULATION-----------------")
    correct = 0
    wrong = 0
    cnt = 0
    for sample, res in zip(X_test, y_test):
        if cnt % 1000 == 0:
            print(f"Finnished batch {cnt/1000}. So far we have {correct} correct vals and {wrong} wrong values.")
        cnt=cnt+1
        cres = circuit(sample)
        if np.argmax(res) == np.argmax(cres):
            correct = correct + 1
        else:
            wrong = wrong + 1
    print(f"The circuit model has an accuracy of: {correct/(correct+wrong)}. That is {correct} values and {wrong} wrong values.")
    
    with open(os.path.join(dir_path, 'chisel4ml', 'results.txt'), 'w') as f:
        f.write(f"QKeras acc:{acc}\n"
                f"chisel4ml acc: {correct/(correct+wrong)}")


if __name__ == '__main__':
    main()