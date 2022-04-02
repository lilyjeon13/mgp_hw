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
                const int n, int m, int a, int b, const int block_size, int four_or_two){
  // #pragma omp parallel for collapse(3)
  for(int i=n/four_or_two*b ; i<n/four_or_two*(b+1) ; i += block_size){
    for(int k=0; k<m; k += block_size){
      for(int j=n/four_or_two*a; j<n/four_or_two*(a+1); j+= block_size){
        for(int ii=i; ii<std::min(i+block_size, n); ii++){
          for(int jj=j; jj<std::min(j+block_size,n); jj++){
            int sum = 0;
            for(int kk=k; kk<std::min(k+block_size, m); kk++){
              sum += matrixA[ii*m + kk] * matrixB_trans[jj*m+kk];
            }
            matrixC[ii*n + jj] += sum;
          }
        }
      }
    }
  }
}


void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n, const int m) {
  const int block_size = 64; //16,32,64,128 64(2.5sec)
  
  std::vector<std::thread> threads;
  int cores = std::thread::hardware_concurrency();
  int B = n/block_size;
  int new_m;

  float log_res = std::log(m)/std::log(2);
  int log_res_i = int(log_res);
  if (log_res != log_res_i){
    new_m = int(pow(2,++log_res_i)); // 바뀌면 matrixA, matrixB 둘다 바뀌어야. 
  }else{
    new_m = m;
  }

  int* const matrixB_trans = new int[n*new_m];
  int* const matrixA_new= new int[n*new_m];
  #pragma omp parallel for collapse(2)
  for(int i=0; i<new_m; i++){
    for (int j=0; j<n; j++){
      if(i<m){
        matrixB_trans[j*new_m + i] = matrixB[i*n + j];
        matrixA_new[j*new_m+i] = matrixA[j*m+i];
      }else{
        matrixB_trans[j*new_m + i] = 0;
        matrixA_new[j*new_m+i] = 0;
      }
    }
  }

  int four_or_two;
  if (B*B >= cores){
    for (int i=0; i<cores; i++){
      int a = i/4;
      int b = i%4;
      four_or_two = 4;
      threads.push_back(std::thread(run_matmul, matrixA_new, matrixB_trans, matrixC, \
            n, new_m, a, b, block_size, four_or_two));
    }
  }else{
    for(int i=0; i<B*B; i++){
      int a = i/2;
      int b = i%2;
      four_or_two = 2;
      threads.push_back(std::thread(run_matmul, matrixA_new, matrixB_trans, matrixC, \
            n, new_m, a, b, block_size, four_or_two));
    }
  }

  for(auto& thread: threads){
    thread.join();
  }

}