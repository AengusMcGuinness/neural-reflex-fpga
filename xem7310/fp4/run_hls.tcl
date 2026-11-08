# vitis_hls -f run_hls.tcl   env: FP4_STEPS="csim csynth cosim export syn", FP4_REF, FP4_OUT,
#                                  FP4_CLK (ns, default 8.0 = 125 MHz), FP4_COSIM_STEPS
set here [file dirname [file normalize [info script]]]
set steps [expr {[info exists ::env(FP4_STEPS)] ? $::env(FP4_STEPS) : "csim csynth"}]
set clk [expr {[info exists ::env(FP4_CLK)] ? $::env(FP4_CLK) : 8.0}]
set ref $::env(FP4_REF)
set out $::env(FP4_OUT)
set cosim_steps [expr {[info exists ::env(FP4_COSIM_STEPS)] ? $::env(FP4_COSIM_STEPS) : 8}]

open_project -reset fp4_hls
set_top llama_fp4
add_files $here/llama_fp4.cpp -cflags "-I$here -ffp-contract=off"
add_files -tb $here/tb_fp4.cpp -cflags "-I$here -ffp-contract=off"
open_solution -reset sol1 -flow_target vivado
set_part xc7a75tfgg484-1
create_clock -period $clk -name default
config_compile -pipeline_loops 0

if {[lsearch $steps csim] >= 0}   { csim_design -argv "$ref $out 256" }
if {[lsearch $steps csynth] >= 0 || [lsearch $steps cosim] >= 0 || [lsearch $steps export] >= 0 || [lsearch $steps syn] >= 0} { csynth_design }
if {[lsearch $steps cosim] >= 0}  { cosim_design -argv "$ref $out $cosim_steps quick" -rtl verilog }
if {[lsearch $steps export] >= 0} { export_design -format ip_catalog -vendor neo -library llama -version 1.0 }
if {[lsearch $steps syn] >= 0}    { export_design -flow syn -format ip_catalog -vendor neo -library llama -version 1.0 }
exit
