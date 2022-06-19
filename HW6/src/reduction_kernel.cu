#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "reduction.h"

#define MAX_THREADS 512

void allocateDeviceMemory(void** M, int size)
{
    cudaError_t err = cudaMalloc(M, size);
    assert(err==cudaSuccess);
}

void deallocateDeviceMemory(void* M)
{
    cudaError_t err = cudaFree(M);
    assert(err==cudaSuccess);
}

void cudaMemcpyToDevice(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyHostToDevice);
    assert(err==cudaSuccess);
}

void cudaMemcpyToHost(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyDeviceToHost);
    assert(err==cudaSuccess);
}

void reduce_ref(const int* const g_idata, int* const g_odata, const int n) {
    for (int i = 0; i < n; i++)
        g_odata[0] += g_idata[i];
}

__device__ void warpReduce(volatile int* sdata, int tid){
    sdata[tid] += sdata[tid+32];
    sdata[tid] += sdata[tid+16];
    sdata[tid] += sdata[tid+8];
    sdata[tid] += sdata[tid+4];
    sdata[tid] += sdata[tid+2];
    sdata[tid] += sdata[tid+1];
}

template <unsigned int blockSize>
__device__ void warpReduce2(volatile int* sdata, int tid){
    if (blockSize >= 64) sdata[tid] += sdata[tid+32];
    if (blockSize >= 32) sdata[tid] += sdata[tid+16];
    if (blockSize >= 16) sdata[tid] += sdata[tid+8];
    if (blockSize >= 8) sdata[tid] += sdata[tid+4];
    if (blockSize >= 4) sdata[tid] += sdata[tid+2];
    if (blockSize >= 2) sdata[tid] += sdata[tid+1];
}

__global__ void reduce1 (const int* const d_idata, int* d_odata, int n){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * MAX_THREADS + tid;
    sdata[tid] = d_idata[i];
    __syncthreads();

    for (unsigned int s=1; s< MAX_THREADS; s*=2){
        if (tid % (2*s) == 0)   sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}
__global__ void reduce2 (const int* const d_idata, int* d_odata, int n){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * MAX_THREADS + tid;
    sdata[tid] = d_idata[i];
    __syncthreads();

    for (unsigned int s=1; s< MAX_THREADS; s*=2){
        int idx = 2* s * tid;
        if (idx < MAX_THREADS)  sdata[idx] += sdata[idx+s];
        __syncthreads();
    }
    if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}
__global__ void reduce3 (const int* const d_idata, int* d_odata, int n){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * MAX_THREADS + tid;
    sdata[tid] = d_idata[i];
    __syncthreads();

    for (unsigned int s=MAX_THREADS/2; s>0; s>>=1){
        if (tid <s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}
__global__ void reduce4 (const int* const d_idata, int* d_odata, int n){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (MAX_THREADS * 2) + tid;
    int temp = (i<n) ? d_idata[i] : 0;
    temp += (i+MAX_THREADS<n) ? d_idata[i+MAX_THREADS] : 0;
    sdata[tid] = temp;
    __syncthreads();

    for (unsigned int s=MAX_THREADS/2; s>0; s>>=1){
        if (tid <s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}
__global__ void reduce5 (const int* const d_idata, int* d_odata, int n){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (MAX_THREADS * 2) + tid;
    int temp = (i<n) ? d_idata[i] : 0;
    temp += (i+MAX_THREADS<n) ? d_idata[i+MAX_THREADS] : 0;
    sdata[tid] = temp;
    __syncthreads();

    for (unsigned int s=MAX_THREADS/2; s>32; s>>=1){
        if (tid <s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid <32) warpReduce(sdata, tid);
    if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}

// FIXED BLOCKSIZE
__global__ void reduce6 (const int* const d_idata, int* d_odata, int n){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (MAX_THREADS * 2) + tid;
    int temp = (i<n) ? d_idata[i] : 0;
    temp += (i+MAX_THREADS<n) ? d_idata[i+MAX_THREADS] : 0;
    sdata[tid] = temp;
    __syncthreads();
    // if (MAX_THREADS >= 512){
    //     if (tid <256) sdata[tid] += sdata[tid+256]; 
    //     __syncthreads();
    // }
    // if (MAX_THREADS >= 256){
    //     if (tid <128) sdata[tid] += sdata[tid+128];
    //     __syncthreads();
    // }
    // if (MAX_THREADS >= 128){
    //     if (tid <64) sdata[tid] += sdata[tid+64]; 
    //     __syncthreads();
    // }

    if (tid <256) sdata[tid] += sdata[tid+256]; 
    __syncthreads();
    if (tid <128) sdata[tid] += sdata[tid+128];
    __syncthreads();
    if (tid <64) sdata[tid] += sdata[tid+64]; 
    __syncthreads();
    
    if (tid <32) warpReduce(sdata, tid);
    if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}
__global__ void reduce7 (const int* const d_idata, int* d_odata, int n){
    extern __shared__ int sdata[];
    unsigned int gridSize = MAX_THREADS*2*gridDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (MAX_THREADS * 2) + tid;
    
    int tempSum = 0;
    while(i < n){
        tempSum += d_idata[i] + d_idata[i+MAX_THREADS];
        i += gridSize;
    }
    sdata[tid] = tempSum;
    __syncthreads();

    // sdata[tid] = 0;
    // while(i < n){
    //     sdata[tid] += d_idata[i] + d_idata[i+MAX_THREADS];
    //     i += gridSize;
    // }
    // __syncthreads();  
    
    // if (MAX_THREADS >= 512){
    //     if (tid <256) sdata[tid] += sdata[tid+256]; 
    //     __syncthreads();
    // }
    // if (MAX_THREADS >= 256){
    //     if (tid <128) sdata[tid] += sdata[tid+128];
    //     __syncthreads();
    // }
    // if (MAX_THREADS >= 128){
    //     if (tid <64) sdata[tid] += sdata[tid+64]; 
    //     __syncthreads();
    // }
    // if (tid <512) sdata[tid] += sdata[tid+512]; 
    // __syncthreads();
    if (tid <256) sdata[tid] += sdata[tid+256]; 
    __syncthreads();
    if (tid <128) sdata[tid] += sdata[tid+128];
    __syncthreads();
    if (tid <64) sdata[tid] += sdata[tid+64]; 
    __syncthreads();

    if (tid <32) warpReduce(sdata, tid);
    if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}

void reduce_optimize(const int* const g_idata, int* const g_odata, const int* const d_idata, 
                        int* const d_odata, const int n) {
    // TODO: Implement your CUDA code
    // Reduction result must be stored in d_odata[0] 
    // You should run the best kernel in here but you must remain other kernels as evidence.

    // # FUNCTION ACCORDING TO CHOICE 
    // int kernel_choice = 7;
    // int block_num = (kernel_choice < 4) ? (n + MAX_THREADS -1) / MAX_THREADS : (n + 2*MAX_THREADS -1) / (MAX_THREADS*2);
    // reduce<<< block_num, MAX_THREADS ,MAX_THREADS*sizeof(int)>>>(d_idata, d_odata, n);

    // # FUNCTION FIXED & thread number fixed to MAX_THREADS

    // kernel 1~3
    // int block_num = (n + MAX_THREADS -1) / MAX_THREADS;
    // kernel 4~7
    int block_num = (n + 16*MAX_THREADS -1) / (MAX_THREADS*16);
    reduce7<<< block_num, MAX_THREADS ,MAX_THREADS*sizeof(int)>>>(d_idata, d_odata, n);
    
    int array_size = n;
    while (true)
    {
        // if (array_size< thread_num){
        if (array_size<MAX_THREADS){
            // # FUNCTION CHOICE
            // reduce<<<1, MAX_THREADS, MAX_THREADS*4>>>(d_odata, d_odata, array_size);
            
            // # FUNCTION FIXED & thread number fixed to MAX_THREADS
            reduce7<<<1, MAX_THREADS, MAX_THREADS*4>>>(d_odata, d_odata, array_size);
            break;

        }else{
            // // # FUNCTION CHOICE 
            // block_num = (kernel_choice < 4) ? (array_size + MAX_THREADS -1)/ MAX_THREADS : (array_size+2*MAX_THREADS -1)/ (MAX_THREADS*2);
            // reduce<<<block_num, MAX_THREADS, MAX_THREADS*4>>>(d_odata, d_odata, array_size);
            // array_size = (kernel_choice < 4) ? (array_size + MAX_THREADS -1) / MAX_THREADS: (array_size + 2*MAX_THREADS -1) / (MAX_THREADS*2);

            // # FUNCTION FIXED & thread number fixed to MAX_THREADS
            // // kernel 1~3
            // block_num = (array_size + MAX_THREADS -1) / MAX_THREADS;
            // reduce1<<<block_num, MAX_THREADS, MAX_THREADS*4>>>(d_odata, d_odata, array_size);
            // array_size = (array_size + MAX_THREADS -1) / MAX_THREADS;
            // kernel 4~7
            block_num = (array_size + 16*MAX_THREADS -1) / (MAX_THREADS*16);
            reduce7<<<block_num, MAX_THREADS, MAX_THREADS*4>>>(d_odata, d_odata, array_size);
            array_size = (array_size + 16*MAX_THREADS -1) / (MAX_THREADS*16);
        }
    }
}

