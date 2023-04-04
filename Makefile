X_train_val.npy y_train_val.npy X_test.npy y_test.npy classes.npy:
	python get_float_dataset.py

.PHONY: clean
clean:
	rm X_train_val.npy y_train_val.npy X_test.npy y_test.npy classes.npy
