####################
##
## Test Condor command file
##
####################

executable	= HTtest
output		= better.out
error		= better.err
request_cpus = 16
log		= better.log
arguments	= 10000000 4000000 4000000 9 16 1
queue
