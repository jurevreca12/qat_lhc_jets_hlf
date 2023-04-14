create_clock -period 10 -name clock [get_ports -filter { NAME =~  "*clk*" && DIRECTION == "IN" }]
set _xlnx_shared_i0 [get_ports -filter { NAME =~  "*in*" && DIRECTION == "IN" }]
set_input_delay -clock [get_clocks *clk*] 0.000 $_xlnx_shared_i0
set_output_delay -clock [get_clocks *clk*] 0.000 [get_ports -filter { NAME =~  "*out*" && DIRECTION == "OUT" }]
