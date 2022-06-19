#include "matmul.h"
#include <thread>
#include <iostream>
#include <vector>
#include <cmath>

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n, const int m) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < m; k++)
        matrixC[i * n + j] += matrixA[i * m + k] * matrixB[k * n + j];
}

void run_matmul(int* const matrixA, int* const matrixB_trans, int* const matrixC, 
                const int n, int m, int a, int b, int block_size){
  //calculate C_a,b
  int start_i = 2*block_size*a;
  int start_j = 2*block_size*b;

  for(int ii=start_i; ii<start_i+2*block_size; ii++){
    for(int jj=start_j;jj<start_j+2*block_size; jj++){
      int sum=0; 
      for(int kk=0; kk<m; kk+=2){
        sum += matrixA[ii*m+kk]*matrixB_trans[jj*m+kk];
        sum += matrixA[ii*m+kk+1]*matrixB_trans[jj*m+kk+1];
      }
      matrixC[ii*n+jj] = sum;
    }
  } 
}
void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n, const int m) {
  int block_size = 64; //16,32,64(1.8sec),128 64(2.5sec)
  int blocks; 
  bool needtoEnd = false;
  while(!needtoEnd){
    blocks = (n/block_size)*(n/block_size);
    if(blocks > 1){
      needtoEnd = true;
      break;
    }else{
      block_size /= 2;
    }
  }

  int new_m;
  float log_res = std::log(m)/std::log(2);
  int log_res_i = int(log_res);
  if (log_res != log_res_i){
    new_m = int(pow(2,++log_res_i)); // 바뀌면 matrixA, matrixB 둘다 바뀌어야. 
  }else{
    new_m = m;
  }

  int* const matrixB_trans = new int[n*new_m]();
  int* const matrixA_new= new int[n*new_m]();
  #pragma omp parallel for collapse(2)
  for (int j=0; j<n; j++){
    for(int i=0; i<new_m; i+=2){
      if(i<m){
        matrixB_trans[j*new_m + i] = matrixB[i*n + j];
        matrixA_new[j*new_m+i] = matrixA[j*m+i];
        matrixB_trans[j*new_m + i+1] = matrixB[(i+1)*n + j];
        matrixA_new[j*new_m+i+1] = matrixA[j*m+i+1];
      }
    }
  }

  int block_len = n/block_size;
  #pragma omp parallel for
  for(int i=0; i<blocks/4; i++){
    int a = i/(block_len/2);
    int b = i%(block_len/2);
    run_matmul(matrixA_new, matrixB_trans, matrixC, n, new_m, a,b, block_size);
  }

}