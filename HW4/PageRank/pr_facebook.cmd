############################################
##
## PageRank Condor command file
##
############################################

executable	 = bin/pr
output		 = result/pr.out
error		 = result/pr.err
log		     = result/pr.log
environment = "LD_LIBRARY_PATH=/usr/local/cuda/lib64"
request_cpus = 16
should_transfer_files   = YES 
when_to_transfer_output = ON_EXIT
arguments	            = -f /nfs/home/mgp2022_data/pagerank/facebook.el -c /nfs/home/mgp2022_data/pagerank/facebook_answer.txt -k 10
queue
