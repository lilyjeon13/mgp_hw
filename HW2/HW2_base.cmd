####################
##
## Test Condor command file
##
####################

executable	= HTtest
 output		= base.out
 error		= base.err
 request_cpus = 16
 log		= base.log
 arguments	= 10000000 4000000 4000000 9 16 0
 queue
