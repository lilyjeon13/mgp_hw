####################
##
## Test Condor command file
##
####################

executable	= sw
 output		= smith_waterman.out
 error		= smith_waterman.err
 request_cpus = 16
 log		= smith_waterman.log
 arguments	= 10000 10000
 queue

