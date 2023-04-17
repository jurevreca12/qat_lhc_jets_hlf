.PHONY: all dataset float chisel4ml hls4ml

C4ML += \
	chisel4ml_bits2/ProcessingPipelineSimple.sv \
	chisel4ml_bits3/ProcessingPipelineSimple.sv \
	chisel4ml_bits4/ProcessingPipelineSimple.sv \
	chisel4ml_bits5/ProcessingPipelineSimple.sv \
	chisel4ml_bits6/ProcessingPipelineSimple.sv \
	chisel4ml_bits7/ProcessingPipelineSimple.sv \
	chisel4ml_bits8/ProcessingPipelineSimple.sv  
C4ML_IMPL = $(patsubst %ProcessingPipelineSimple.sv, %utilization.rpt, $(C4ML))

HLS4ML += \
	hls4ml_bits2/vivado_hls.log \
	hls4ml_bits3/vivado_hls.log \
	hls4ml_bits4/vivado_hls.log \
	hls4ml_bits5/vivado_hls.log \
	hls4ml_bits6/vivado_hls.log \
	hls4ml_bits7/vivado_hls.log \
	hls4ml_bits8/vivado_hls.log
HLS4ML_IMPL = $(patsubst %vivado_hls.log, %utilization.rpt, $(HLS4ML))

all: dataset float chisel4ml hls4ml
hls4ml: $(HLS4ML) $(HLS4ML_IMPL)
chisel4ml: $(C4ML) $(C4ML_IMPL)
dataset: X_train_val.npy y_train_val.npy X_test.npy y_test.npy
float: float/results.txt


float/results.txt: dataset
	python float_train_evaluate.py

X_train_val.npy y_train_val.npy X_test.npy y_test.npy classes.npy:
	python get_float_dataset.py

chisel4ml_bits%/ProcessingPipelineSimple.sv: X_train_val.npy y_train_val.npy X_test.npy y_test.npy
	python chisel4ml_train_evaluate.py $*

chisel4ml_bits%/utilization.rpt: chisel4ml_bits%/ProcessingPipelineSimple.sv
	vivado -mode batch -source synth.tcl -tclargs $(PWD)/chisel4ml_bits$*

hls4ml_bits%/vivado_hls.log: X_train_val.npy y_train_val.npy X_test.npy y_test.npy
	python hls4ml_train_evaluate.py $*

hls4ml_bits%/utilization.rpt: hls4ml_bits%/vivado_hls.log
	vivado -mode batch -source hlssynth.tcl -tclargs $(PWD)/hls4ml_bits$*

.PHONY: clean
clean:
	rm -f X_train_val.npy y_train_val.npy X_test.npy y_test.npy classes.npy 
	rm -rf $(wildcard chisel4ml_bits*)
	rm -rf $(wildcard hls4ml_bits*)
	rm -rf float/
	rm -rf xsim.dir/
	rm -f vivado* xvlog*
