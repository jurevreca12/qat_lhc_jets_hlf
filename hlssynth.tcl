if { $argc != 1 } {
	break
}

set work_dir [lindex $argv 0]

set_part xcvu9p-flga2104-2L-e

add_files $work_dir/myproject_prj/solution1/syn/verilog
read_xdc hlsconstr.xdc
 
synth_design -top myproject -retiming -flatten_hierarchy full
opt_design

report_timing_summary -file $work_dir/post_synth_timing_summary.rpt
report_utilization -file $work_dir/utilization.rpt

