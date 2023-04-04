.PHONY: all
all: chisel4ml/results.txt chisel4ml/ProcessingPipeline.sv

X_train_val.npy y_train_val.npy X_test.npy y_test.npy classes.npy:
	python get_float_dataset.py

chisel4ml/results.txt chisel4ml/ProcessingPipeline.sv: X_train_val.npy y_train_val.npy X_test.npy y_test.npy classes.npy
	python chisel4ml_train_evaluate.py

.PHONY: clean
clean:
	rm X_train_val.npy y_train_val.npy X_test.npy y_test.npy classes.npy chisel4ml/results.txt chisel4ml/ProcessingPipeline.sv
