############################################
##
## Matrix Multiplication Condor command file
##
############################################

executable	 = bin/matmul
output		 = result/matmul.out
error		 = result/matmul.err
log		     = result/matmul.log
environment = "LD_LIBRARY_PATH=/usr/local/cuda/lib64"
request_cpus = 16
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
arguments	            = /nfs/home/mgp2022_data/hw5/input_1024.txt /nfs/home/mgp2022_data/hw5/output_1024.txt
queue