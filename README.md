# qat\_lhc\_jets\_hlf
Quantization-Aware Training of neural networks for the LHC-Jets-Hlf dataset and comparison of
chisel4ml, hls4ml and FINN.

To rebuild the project install verilator and the java run time.
You will also need Xilinx Vivado 2019.2 in the path, as well
as a licence for the xcvu9p-flga2104-2L-e fpga part.

For debian systems:
	`apt-get install verilator default-jre`

Next its best to create a python enviroment:
	`python -m venv venv/`

and then install the requirements.
	`pip install -r requirements.txt`


