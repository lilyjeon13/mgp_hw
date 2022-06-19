#include <stdio.h>
#include <iostream>
#include <chrono>
#include <assert.h>
#include <cmath>
#include "matmul.h"
#include <memory.h>

#define TILE_WIDTH 32
using namespace std;

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

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

__global__ void matmul_kernel (const int* d_A, const int* d_B, int* d_C, int n){

  __shared__ int subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ int subTileB[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  int Pvalue = 0;

  for (int m=0; m<n/TILE_WIDTH; ++m){
    subTileA [ty][tx] = d_A[Row*n + m*TILE_WIDTH+tx];
    subTileB [ty][tx] = d_B[(m*TILE_WIDTH+ty)*n + Col];
    __syncthreads();

    for(int k=0; k<TILE_WIDTH; ++k){
      Pvalue += subTileA[ty][k] * subTileB[k][tx];
    }
    __syncthreads();
  }
  d_C[Row*n+Col] = Pvalue;
}

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int* d_A, const int* d_B,  int* const d_C, const int n) {

  int new_n;
  if (n % 32 != 0){
    int quot = (n/32)+1;
    new_n = quot * 32;
  }else{
    new_n = n;
  }
  
  int* new_matrixA;
  int* new_matrixB;
  int* new_matrixC; 
  
  int* new_d_A;
  int* new_d_B;
  int* new_d_C;

  if (new_n != n){  // n is not multiple of 32
    int new_size = sizeof(int)*new_n*new_n;
    int new_total = new_size/sizeof(int);
    new_matrixA = new int[new_total];
    new_matrixB = new int[new_total];
    new_matrixC = new int[new_total];

    memset(new_matrixA, 0, new_total*sizeof(int));
    memset(new_matrixB, 0, new_total*sizeof(int));

    for (int i=0; i<n; i++){
      for (int j=0; j<n; j++){
        new_matrixA[i*new_n+j] = matrixA[i*n+j];
        new_matrixB[i*new_n+j] = matrixB[i*n+j];
      }
    }
    
    allocateDeviceMemory((void**)&new_d_A, new_size);
    allocateDeviceMemory((void**)&new_d_B, new_size);
    allocateDeviceMemory((void**)&new_d_C, new_size);
    
    cudaMemset((void*)new_d_A, 0, new_size);
    cudaMemset((void*)new_d_B, 0, new_size);
    
    // transfer matrix A,B from host to device
    cudaMemcpy((void*)new_d_A, (void*)new_matrixA, new_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)new_d_B, (void*)new_matrixB, new_size, cudaMemcpyHostToDevice);
  }else{ // n is multiple of 32
    // transfer matrix A,B from host to device
    cudaMemcpy((void*)d_A, (void*)matrixA, 4*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_B, (void*)matrixB, 4*n*n, cudaMemcpyHostToDevice);
  }
  
  //kernel invocation code
  dim3 dimGrid(new_n/TILE_WIDTH, new_n/TILE_WIDTH, 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  if (new_n != n){
    matmul_kernel<<<dimGrid, dimBlock>>> (new_d_A, new_d_B, new_d_C, new_n);
    // transfer matrix C from device to host
    cudaMemcpy(new_matrixC, new_d_C, 4*new_n*new_n, cudaMemcpyDeviceToHost);
    for (int i=0; i<n; i++){
      for (int j=0; j<n; j++){
        matrixC[i*n+j] = new_matrixC[i*new_n+j];
      }
    }
    deallocateDeviceMemory((void*)new_d_A);
    deallocateDeviceMemory((void*)new_d_B);
    deallocateDeviceMemory((void*)new_d_C);
    free (new_matrixA);
    free (new_matrixB);
    free (new_matrixC);

  }else{
    matmul_kernel<<<dimGrid, dimBlock>>> (d_A, d_B, d_C, n);
    // transfer matrix C from device to host
    cudaMemcpy(matrixC, d_C, 4*n*n, cudaMemcpyDeviceToHost);
  }
}