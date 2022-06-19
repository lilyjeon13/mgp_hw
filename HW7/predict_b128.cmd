############################################
##
## Join Tables Condor command file
##
############################################

executable	 = predict
output		 = result/vgg16_b128.out
error		 = result/vgg16_b128.err
log		     = result/vgg16_b128.log
request_cpus = 1
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_output_files   = tmp
arguments	              = /nfs/home/mgp2022_data/hw7/cifar10/test_batch.bin 0 128 tmp/cifar10_test_%d_%s.bmp /nfs/home/mgp2022_data/hw7/vgg_weight/values_vgg.txt
queue
