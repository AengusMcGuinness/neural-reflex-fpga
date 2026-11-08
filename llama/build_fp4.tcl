# Build the FP4 Urbana bitstream:  vivado -mode batch -source build.tcl -tclargs <hls_ip_dir> <out_dir>
#   hls_ip_dir : Vitis HLS export (…/llama_hls/sol1/impl/ip) of llama_fp4
set src [file dirname [file normalize [info script]]]
set hls_ip [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set part xc7s50csga324-1
file mkdir $out
file mkdir $out/ip

# non-project flow (synth_ip is only supported there)
create_project -in_memory -part $part
set_property ip_repo_paths $hls_ip [current_project]
update_ip_catalog
create_ip -vlnv neo:llama:llama_fp4:1.0 -module_name llama_fp4_0 -dir $out/ip
generate_target all [get_ips llama_fp4_0]
synth_ip [get_ips llama_fp4_0]

read_verilog -sv [list $src/rtl/llama_fp4_top.sv $src/rtl/uart_axi_bridge.sv]
read_xdc $src/constraints/llama_fp4_top.xdc

set ctrl_aw [expr {[info exists ::env(FP4_CTRL_AW)] ? $::env(FP4_CTRL_AW) : 11}]
synth_design -top llama_fp4_top -part $part -generic CTRL_AW=$ctrl_aw
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

# refuse to emit a bitstream that does not meet timing
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "LLAMA_TIMING WNS=$wns WHS=$whs"
if {$wns < 0 || $whs < 0} {
    puts "LLAMA_BUILD_FAILED timing not met"
    exit 1
}
write_bitstream -force $out/llama_fp4_top.bit
puts "LLAMA_BUILD_OK $out/llama_fp4_top.bit"
