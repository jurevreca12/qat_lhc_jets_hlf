.PHONY: all dataset float chisel4ml

dataset: X_train_val.npy y_train_val.npy X_test.npy y_test.npy

float: float/results.txt

C4ML += \
	./chisel4ml_bits2/ProcessingPipelineSimple.sv \
	./chisel4ml_bits3/ProcessingPipelineSimple.sv \
	./chisel4ml_bits4/ProcessingPipelineSimple.sv \
	./chisel4ml_bits5/ProcessingPipelineSimple.sv \
	./chisel4ml_bits6/ProcessingPipelineSimple.sv \
	./chisel4ml_bits7/ProcessingPipelineSimple.sv \
	./chisel4ml_bits8/ProcessingPipelineSimple.sv  
C4ML_IMPL = $(patsubst %ProcessingPipelineSimple.sv, %output/impl_netlist.v, $(C4ML))
chisel4ml: $(C4ML) $(C4ML_IMPL)

all: dataset float chisel4ml #hls4ml/vivado_hls.log

float/results.txt: dataset
	python float_train_evaluate.py

X_train_val.npy y_train_val.npy X_test.npy y_test.npy classes.npy:
	python get_float_dataset.py

$(C4ML): X_train_val.npy y_train_val.npy X_test.npy y_test.npy
	python chisel4ml_train_evaluate.py

chisel4ml_bits%/output/impl_netlist.v: chisel4ml_bits%/ProcessingPipelineSimple.sv
	mkdir chisel4ml_bits$*/output
	vivado -mode batch -source chisel4ml_bits$*/synth.tcl

hls4ml.tar.gz hls4ml/vivado_hls.log: X_train_val.npy y_train_val.npy X_test.npy y_test.npy
	python hls4ml_train_evaluate.py

.PHONY: clean
clean:
	rm -f X_train_val.npy y_train_val.npy X_test.npy y_test.npy classes.npy 
	rm -f $(wildcard chisel4ml_bits*/results.txt) $(wildcard chisel4ml_bits*/ProcessingPipelineSimple.sv)
	rm -rf $(wildcard chisel4ml_bits*/output)
	rm -rf hls4ml.tar.gz hls4ml/
	rm -rf xsim.dir/
	rm -f vivado* xvlog*
