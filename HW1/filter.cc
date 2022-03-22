#include <stdlib.h>
// #include <cstdio>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <cmath>

void weightAdd(int tid, int num_threads, int N, const float* k, float* array_in, float* array_out_parallel){
  
  int assign = int(ceil((N-4)/float(num_threads)));
  int start_pnt = tid * assign;

  for (int p=0; p<assign; p++){
    int idx = p + start_pnt;
    float value = 0;  
    if (idx < (N-4)){
      value = value+ array_in[idx+0] * k[0];
      value = value+ array_in[idx+1] * k[1];
      value = value+ array_in[idx+2] * k[2];
      value = value+ array_in[idx+3] * k[3];
      value = value+ array_in[idx+4] * k[4];
    }
    array_out_parallel[idx] =  value;   
  }
}

int main(int argc, char** argv) 
{

  if(argc < 2) std::cout<<"Usage : ./filter num_items"<<std::endl;
  int N = atoi(argv[1]);
  int NT=16; //Default value. change it as you like. 32 16cores


  //0. Initialize

  const int FILTER_SIZE=5;
  const float k[FILTER_SIZE] = {0.125, 0.25, 0.25, 0.25, 0.125};
  alignas(64) float *array_in = new float[N];
  alignas(64) float *array_out_serial = new float[N];
  alignas(64) float *array_out_parallel = new float[N];
  {
    std::chrono::duration<float> diff;
    auto start = std::chrono::steady_clock::now();
    for(int i=0;i<N;i++) {
      array_in[i] = i;
    }
    auto end = std::chrono::steady_clock::now();
    diff = end-start;
    std::cout<<"init took "<<diff.count()<<" sec"<<std::endl;
  }

  {
    //1. Serial
    std::chrono::duration<float> diff;
    auto start = std::chrono::steady_clock::now();
    for(int i=0;i<N-4;i++) {
      for(int j=0;j<FILTER_SIZE;j++) {
        array_out_serial[i] += array_in[i+j] * k[j];
      }
    }
    auto end = std::chrono::steady_clock::now();
    diff = end-start;
    std::cout<<"serial 1D filter took "<<diff.count()<<" sec"<<std::endl;
  }

  {
    //2. parallel 1D filter
    std::chrono::duration<float> diff;
    auto start = std::chrono::steady_clock::now();
    /* TODO: put your own parallelized 1D filter here */
    /****************/

    std::vector<std::thread> threads;

    for(int t=0; t<NT; t++){
      threads.push_back(std::thread(weightAdd, t, NT, N, std::ref(k),std::ref(array_in), std::ref(array_out_parallel)));
    }
    
    for(auto& thread: threads){
      thread.join();
    }
    
    /****************/
    /* TODO: put your own parallelized 1D filter here */
    auto end = std::chrono::steady_clock::now();
    diff = end-start;
    std::cout<<"parallel 1D filter took "<<diff.count()<<" sec"<<std::endl;



    int error_counts=0;
    const float epsilon = 0.01;
    for(int i=0;i<N;i++) {
      float err= std::abs(array_out_serial[i] - array_out_parallel[i]);
      if(err > epsilon) {
        error_counts++;
        if(error_counts < 5) {
          std::cout<<"ERROR at "<<i<<": Serial["<<i<<"] = "<<array_out_serial[i]<<" Parallel["<<i<<"] = "<<array_out_parallel[i]<<std::endl;
          std::cout<<"err: "<<err<<std::endl;
        }
      }
    }


    if(error_counts==0) {
      std::cout<<"PASS"<<std::endl;
    } else {
      std::cout<<"There are "<<error_counts<<" errors"<<std::endl;
    }

  }
  return 0;
}


