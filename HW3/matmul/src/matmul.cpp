#include "matmul.h"
#include <thread>
#include <iostream>
#include <vector>

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n, const int m) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < m; k++)
        matrixC[i * n + j] += matrixA[i * m + k] * matrixB[k * n + j];
}

void run_matmul(const int* const matrixA, const int* const matrixB, int* matrixB_trans, int* const matrixC, 
                const int n, const int m, int a, int b, const int block_size, bool large){
  int B = n/block_size;
  int four_or_two = (large) ? 4 : 2;
  // C_4b,4a ~ C_4b+B/4-1, 4a+B/4-1
  #pragma omp parallel for collapse(2)
  for (int i=(B/four_or_two)*b; i<(B/four_or_two)*(b+1); i++){
    for (int j=(B/four_or_two)*a; j<(B/four_or_two)*(a+1); j++){
      //C_i,j = A_i,0 * B_0,j + A_i,1 * B_1,j + ... A_i,k * B_k,j + ... A_i,m/b-1 * B_m/b-1,j
      for (int k=0; k<m/block_size; k++){ // matrix-wise multiplication
        //A_i,k * B_k,j -> A_i,k * B'_j,k matrix multiplication
        for (int ii=0; ii<block_size; ii++){
          for(int jj=0; jj<block_size; jj++){
            for (int kk=0; kk<block_size; kk++){
              int index_a_i = i * block_size + ii;
              int index_a_j = k * block_size + kk;
              int index_b_i = k * block_size + kk;
              int index_b_j = j * block_size + jj;
              int index_bt_i = j * block_size + jj;
              int index_bt_j = k * block_size + kk;

              // matrixC[index_a_i * n + index_b_j] += \
              //   matrixA[index_a_i*m + index_a_j] * matrixB[index_b_i * n + index_b_j];
              matrixC[index_a_i * n + index_b_j] += \
                matrixA[index_a_i*m + index_a_j] * matrixB_trans[index_bt_i * m + index_bt_j];

            }
          }
        }
      }
    }
  }
}


void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n, const int m) {
  const int block_size = 128; 
  
  std::vector<std::thread> threads;
  int cores = std::thread::hardware_concurrency();
  int B = n/block_size;
  int* matrixB_trans = new int[n*m];
  #pragma omp parallel for collapse(2)
  for (int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      // B_i,j *n -> B'_j,i *m
      matrixB_trans[j*m + i] = matrixB[i*n + j];
    }
  }

  if (B*B >= cores){
    for (int i=0; i<cores; i++){
      int a = i/4;
      int b = i%4;
      threads.push_back(std::thread(run_matmul, matrixA, matrixB, matrixB_trans, matrixC, n, m, a, b, block_size, true));
    }
  }else{
    for(int i=0; i<B*B; i++){
      int a = i/2;
      int b = i%2;
      threads.push_back(std::thread(run_matmul, matrixA, matrixB, matrixB_trans, matrixC, n, m, a, b, block_size, false));
    }
  }

  for(auto& thread: threads){
    thread.join();
  }

}