.PHONY: all
all: chisel4ml/results.txt chisel4ml/ProcessingPipelineSimple.sv chisel4ml/output/impl_netlist.v hls4ml/vivado_hls.log

X_train_val.npy y_train_val.npy X_test.npy y_test.npy classes.npy:
	python get_float_dataset.py

chisel4ml/results.txt chisel4ml/ProcessingPipelineSimple.sv: X_train_val.npy y_train_val.npy X_test.npy y_test.npy
	python chisel4ml_train_evaluate.py

chisel4ml/output/impl_netlist.v chisel4ml/output/impl_netlist.edif chiselml/output/utilization.rpt: chisel4ml/ProcessingPipelineSimple.sv
	mkdir chisel4ml/output
	vivado -mode batch -source chisel4ml/synth.tcl

hls4ml.tar.gz hls4ml/vivado_hls.log: X_train_val.npy y_train_val.npy X_test.npy y_test.npy
	python hls4ml_train_evaluate.py

.PHONY: clean
clean:
	rm -f X_train_val.npy y_train_val.npy X_test.npy y_test.npy classes.npy 
	rm -f chisel4ml/results.txt chisel4ml/ProcessingPipeline.sv
	rm -rf hls4ml.tar.gz hls4ml/
	rm -rf chisel4ml/output
	rm -rf xsim.dir/
	rm -f vivado* xvlog*
