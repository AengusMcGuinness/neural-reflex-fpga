set project_dir [file normalize [file dirname [info script]]]
set vendor_dir /home/amcguinness/cs1410_lab/lab2/lab2_server/OK_library
set output_dir [file join $project_dir artifacts]
file mkdir $output_dir

read_verilog -sv [file join $project_dir rtl neo_packet_top.sv]
read_verilog [file join $vendor_dir okLibrary.v]
foreach module {okCoreHarness okWireIn okWireOut okPipeIn} {
    read_verilog [file join $vendor_dir ${module}.v]
}
read_xdc [file join $project_dir constraints xem7310_host.xdc]

synth_design -top neo_packet_top -part xc7a75tfgg484-1
opt_design
place_design
route_design
report_timing_summary -file [file join $output_dir timing.rpt]
report_utilization -file [file join $output_dir utilization.rpt]
report_drc -file [file join $output_dir drc.rpt]
write_checkpoint -force [file join $output_dir neo_packet_top_routed.dcp]
write_bitstream -force [file join $output_dir neo_packet_top.bit]
