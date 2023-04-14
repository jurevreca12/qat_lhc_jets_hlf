# synth.tcl is a synthesis script for Vivado
if { $argc != 1 } {
	break
}

set work_dir [lindex $argv 0]

set_part xcvu9p-flga2104-2L-e

read_verilog $work_dir/ProcessingPipelineSimple.sv 
read_xdc     constr.xdc

# Run synthesis
synth_design -top ProcessingPipelineSimple -retiming -flatten_hierarchy full
opt_design
report_timing_summary	-file $work_dir/post_synth_timing_summary.rpt
report_utilization -file $work_dir/utilization.rpt
