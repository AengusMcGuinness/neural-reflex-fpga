# Build the XEM7310 FP4 bitstream (Vivado 2022.2, non-project flow):
#   vivado -mode batch -source build.tcl -tclargs <hls_ip_dir> <out_dir> [ok_library_dir]
#   hls_ip_dir : Vitis HLS export (fp4_hls/sol1/impl/ip) of xem7310/fp4/llama_fp4
#   ok_library : Opal Kelly FrontPanel HDL for the XEM7310 (default: the CS1410 copy on cs1410vnc)
set src [file dirname [file normalize [info script]]]
set hls_ip [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set ok_dir [expr {[llength $argv] > 2 ? [lindex $argv 2] : "/home/amcguinness/cs1410_lab/lab2/lab2_server/OK_library"}]
set part xc7a75tfgg484-1
set clk_div [expr {[info exists ::env(FP4_CLK_DIV)] ? $::env(FP4_CLK_DIV) : 7.5}]
set clk_hz [expr {int(1000000000.0 / $clk_div)}]
file mkdir $out
file mkdir $out/ip

create_project -in_memory -part $part
set_property ip_repo_paths $hls_ip [current_project]
update_ip_catalog
create_ip -vlnv neo:llama:llama_fp4:1.0 -module_name llama_fp4_0 -dir $out/ip
generate_target all [get_ips llama_fp4_0]
synth_ip [get_ips llama_fp4_0]

read_verilog -sv [list $src/rtl/llama_xem7310_top.sv $src/rtl/fp_cmd_proc.sv]
read_verilog [file join $ok_dir okLibrary.v]
foreach m {okCoreHarness okWireIn okWireOut okPipeIn okPipeOut} {
    read_verilog [file join $ok_dir $m.v]
}
read_xdc $src/constraints/llama_xem7310.xdc

synth_design -top llama_xem7310_top -part $part \
    -generic CLKOUT_DIVIDE=$clk_div -generic CLK_HZ=$clk_hz
write_checkpoint -force $out/post_synth.dcp
report_utilization -file $out/utilization_synth.rpt
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force $out/post_route.dcp
report_timing_summary -max_paths 20 -file $out/timing.rpt
report_utilization -file $out/utilization.rpt
report_drc -file $out/drc.rpt
report_clock_interaction -file $out/clock_interaction.rpt

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "LLAMA_TIMING WNS=$wns WHS=$whs CLK_HZ=$clk_hz"
if {$wns < 0 || $whs < 0} {
    puts "LLAMA_BUILD_FAILED timing not met"
    exit 1
}
write_bitstream -force $out/llama_xem7310_top.bit
puts "LLAMA_BUILD_OK $out/llama_xem7310_top.bit"
