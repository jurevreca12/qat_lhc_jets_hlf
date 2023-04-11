# qat\_lhc\_jets\_hlf
Quantization-Aware Training of neural networks for the LHC-Jets-Hlf dataset and comparison of
chisel4ml and hls4ml.

To rebuild the project install verilator and the java run time.
You will also need Xilinx Vivado 2019.2 on the path, as well
as a licence for the xcvu9p-flga2104-2L-e fpga part.
(If you don't have the license you can just change it to
 a part that has a free license, e.g. zedboard)

For debian systems:
	`apt-get install verilator default-jre`

Next its best to create a python enviroment:
	`python -m venv venv/`

and then install the requirements.
	`pip install -r requirements.txt`


Finally, running `make all` will train the models and run chisel4ml and hls4ml tools.
