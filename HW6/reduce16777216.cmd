############################################
##
## Sum Reduction Condor command file
##
############################################

executable	 = bin/reduce
output		 = result/reduce.out
error		 = result/reduce.err
log		     = result/reduce.log
environment = "LD_LIBRARY_PATH=/usr/local/cuda/lib64"
request_cpus = 16
should_transfer_files   = YES 
when_to_transfer_output = ON_EXIT
arguments	            = /nfs/home/mgp2022_data/hw6/input_16777216.txt /nfs/home/mgp2022_data/hw6/output_16777216.txt
queue