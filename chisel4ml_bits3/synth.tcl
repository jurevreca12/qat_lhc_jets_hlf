# synth.tcl is a synthesis script for Vivado
# 
# run "vivado -mode batch -source synth.tcl" to get a compiled vivado design
#
set script_path [ file dirname [ file normalize [ info script ] ] ]
set project_root_dir $script_path
set source_dir $project_root_dir/.
set output_dir $script_path/output/.

set_part xcvu9p-flga2104-2L-e
# Out-of-context synthesis

read_verilog $source_dir/ProcessingPipelineSimple.sv 
read_xdc     $source_dir/constr.xdc

# Run synthesis
synth_design -top ProcessingPipelineSimple \
	     -mode out_of_context
write_checkpoint -force $output_dir/post_synth
report_timing_summary 		-file $output_dir/post_synth_timing_summary.rpt
report_power			-file $output_dir/post_synth_power.rpt
report_clock_interaction	-file $output_dir/post_synth_clock_interaction.rpt \
				-delay_type min_max
report_high_fanout_nets		-file $output_dir/post_synth_high_fanout_nets.rpt  \
				-fanout_greater_than 200 \
				-max_nets 50
report_utilization -file $output_dir/utilization.rpt

write_verilog -force $output_dir/impl_netlist.v
write_edif    -force $output_dir/impl_netlist.edif
