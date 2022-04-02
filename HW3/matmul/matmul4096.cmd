############################################
##
## Matrix Multiplication Condor command file
##
############################################

executable	 = matmul
output		 = result/matmul.out
error		 = result/matmul.err
log		     = result/matmul.log
request_cpus = 16
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
arguments	            = /nfs/home/mgp2022_data/input_4096_8192.txt /nfs/home/mgp2022_data/output_4096_8192.txt
queue