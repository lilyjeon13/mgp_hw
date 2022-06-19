#!/bin/bash

vessl experiment create \
  --organization "yonsei-acsys" \
  --project "cuda-vgg" \
  --image-url "public.ecr.aws/vessl/kernels:yonsei-mgp-vgg-cuda-10" \
  --cluster "aws-apne2-prod1" \
  --resource "v1.k80-1.mem-52.spot" \
  --command "mv /home/vessl/local/vgg16_cuda.cu /cuda-vgg/src && mv /home/vessl/local/vgg16_cuda.h /cuda-vgg/src && cd /cuda-vgg && make clean && make && /cuda-vgg/predict /cuda-vgg/test_batch.bin 0 128 tmp/cifar10_test_%d_%s.bmp /cuda-vgg/values_vgg.txt >> /output/result.txt && cat /output/result.txt" \
  --working-dir /home/vessl/local/ \
  --root-volume-size "20Gi" \
  --output-dir "/output/" \
  --upload-local-file "src:/home/vessl/local/"
