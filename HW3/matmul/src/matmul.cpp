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
void strassen_matmul(int* A, int* B, int* C, int n){
  //matrix multiplication actually done
  if ((long long) n*n*n <= 33000){
    for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
        int sum=0;
        for(int k=0; k<n; k++){
          sum += A[i*n+k]*B[j*n+k];
        }
        C[i*n+j] += sum;
      }
    }
    return;
  }
  int half_n = n/2;
  int size = half_n * half_n;
  int** a = new int*[8];
  int** b = new int*[8]; 
  int** m = new int*[8];
  for(int i=0; i<8; i++){
    a[i] = new int[size];
    b[i] = new int[size];
    m[i] = new int[size];
  }

  for (int i=0; i<half_n; i++){
    for(int j=0; j<half_n; j++){
      int a11 = A[i*n+j];
      int a12 = A[(i*n+j+half_n)];
      int a21 = A[(i+half_n)*n+j];
      int a22 = A[(i+half_n)*n+j+half_n];
      int b11 = B[i*n+j];
      int b12 = B[i*n+j+half_n];
      int b21 = B[(i+half_n)*n+j];
      int b22 = B[(i+half_n)*n+j+half_n];

      a[1][i*half_n+j] = a11+a22;
      a[2][i*half_n+j] = a21+a22;
      a[3][i*half_n+j] = a11;
      a[4][i*half_n+j] = a22;
      a[5][i*half_n+j] = a11+a12;
      a[6][i*half_n+j] = a21-a11;
      a[7][i*half_n+j] = a12-a22;
      b[1][i*half_n+j] = b11+b22;
      b[2][i*half_n+j] = b11;
      b[3][i*half_n+j] = b12-b22;
      b[4][i*half_n+j] = b21-b11;
      b[5][i*half_n+j] = b22;
      b[6][i*half_n+j] = b11+b12;
      b[7][i*half_n+j] = b21+b22;
    }
  }
  for(int i=1; i<8; i++){
    strassen_matmul(a[i], b[i], m[i], half_n);
  }

  for(int i=0; i<half_n;i++){
    for(int j=0; j<half_n; j++){
      C[i*n+j] = m[1][i*half_n+j] + m[4][i*half_n+j] - m[5][i*half_n+j] + m[7][i*half_n+j];
      C[i*n+j+half_n] = m[3][i*half_n+j]+ m[5][i*half_n+j];
      C[(i+half_n)*n+j] = m[2][i*half_n+j]+ m[4][i*half_n+j];
      C[(i+half_n)*n+j+half_n] = m[1][i*half_n+j]-m[2][i*half_n+j] +m[3][i*half_n+j]+m[6][i*half_n+j];
    }
  }

}

void run_matmul(int* const matrixA, int* const matrixB_trans, int* const matrixC, 
                const int n, int m, int a, int b, const int block_size, int four_or_two){
  for(int i=n/four_or_two*b ; i<n/four_or_two*(b+1) ; i += block_size){
    for(int k=0; k<m; k += block_size){
      for(int j=n/four_or_two*a; j<n/four_or_two*(a+1); j+= block_size){
        // C_i,j calculation 
        int half_b = block_size/2;
        
        for(int ii=i; ii<i+half_b; ii++){
          for(int jj=j; jj<j+half_b; jj++){
            int sum1(0), sum2(0), sum3(0), sum4(0), sum5(0), sum6(0), sum7(0);
            for(int kk=k; kk<k+half_b; kk++){
              int a11 = matrixA[ii*m + kk];
              int a12 = matrixA[ii*m + kk+half_b];
              int a21 = matrixA[(ii+half_b)*m+ kk];
              int a22 = matrixA[(ii+half_b)*m + kk+half_b];
              int b11 = matrixB_trans[jj*m + kk];
              int b12 = matrixB_trans[(jj+half_b)*m+kk];
              int b21 = matrixB_trans[jj*m + kk+half_b];
              int b22 = matrixB_trans[(jj+half_b)*m +kk+half_b];

              sum1 += (a11 + a22)*(b11+ b22);
              sum2 += (a21 + a22)*b11;
              sum3 += a11 * (b12 - b22);
              sum4 += a22 * (b21 - b11);
              sum5 += (a11 + a12) * b22;
              sum6 += (a21 - a11) * (b11 + b12);
              sum7 += (a12 - a22) * (b21 + b22);
            }
            matrixC[ii*n+jj] += sum1+sum4-sum5+sum7;
            matrixC[ii*n+jj+half_b] += sum3+sum5;
            matrixC[(ii+half_b)*n+jj] += sum2+sum4;
            matrixC[(ii+half_b)*n+jj+half_b] += sum1-sum2+sum3+sum6;
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
  // int B = n/block_size;
  int blocks = (n/block_size)*(n/block_size);
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
      }
      else{
        matrixB_trans[j*new_m + i] = 0;
        matrixA_new[j*new_m+i] = 0;
      }
    }
  }

  int four_or_two;
  // if (blocks >= cores){
  //   for (int i=0; i<cores; i++){
  //     int a = i/4;
  //     int b = i%4;
  //     four_or_two = 4;
  //     threads.push_back(std::thread(run_matmul, matrixA_new, matrixB_trans, matrixC, \
  //           n, new_m, a, b, block_size, four_or_two));
  //   }
  // }else{
  //   for(int i=0; i<blocks; i++){
  //     int a = i/2;
  //     int b = i%2;
  //     four_or_two = 2;
  //     threads.push_back(std::thread(run_matmul, matrixA_new, matrixB_trans, matrixC, \
  //           n, new_m, a, b, block_size, four_or_two));
  //   }
  // }
  if (blocks >= cores){
    #pragma omp parallel for
    for (int i=0; i<cores; i++){
      int a = i/4;
      int b = i%4;
      four_or_two = 4;
      run_matmul(matrixA_new, matrixB_trans, matrixC, n, new_m, a,b, block_size, four_or_two);
    }
  }else{
    #pragma omp parallel for
    for(int i=0; i<blocks; i++){
      int a = i/2;
      int b = i%2;
      four_or_two = 2;
      run_matmul(matrixA_new, matrixB_trans, matrixC, n, new_m, a,b, block_size, four_or_two);
    }
  }
  // for(auto& thread: threads){
  //   thread.join();
  // }

}