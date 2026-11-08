# vitis_hls -f run_hls.tcl  [env: LLAMA_STEPS=csim|csynth|cosim|export (space separated)]
#   LLAMA_CKPT / LLAMA_REF : checkpoint and ref_dump prefix used by the testbench
set here [file dirname [file normalize [info script]]]
set steps [expr {[info exists ::env(LLAMA_STEPS)] ? $::env(LLAMA_STEPS) : "csim csynth"}]
set ckpt $::env(LLAMA_CKPT)
set ref  $::env(LLAMA_REF)
set cosim_pos [expr {[info exists ::env(LLAMA_COSIM_POS)] ? $::env(LLAMA_COSIM_POS) : 3}]

open_project -reset llama_hls
set_top llama_forward
add_files $here/llama_kernel.cpp -cflags "-I$here"
add_files -tb $here/tb_llama.cpp -cflags "-I$here"
open_solution -reset sol1 -flow_target vivado
set_part xc7s50csga324-1
create_clock -period 10 -name default
config_interface -m_axi_addr64=false
config_compile -pipeline_loops 0

if {[lsearch $steps csim] >= 0}   { csim_design -argv "$ckpt $ref" }
if {[lsearch $steps csynth] >= 0 || [lsearch $steps cosim] >= 0 || [lsearch $steps export] >= 0 || [lsearch $steps syn] >= 0} { csynth_design }
if {[lsearch $steps cosim] >= 0}  { cosim_design -argv "$ckpt $ref $cosim_pos" -rtl verilog }
if {[lsearch $steps export] >= 0} { export_design -format ip_catalog -vendor neo -library llama -version 1.0 }
if {[lsearch $steps syn] >= 0}    { export_design -flow syn -format ip_catalog -vendor neo -library llama -version 1.0 }
exit
