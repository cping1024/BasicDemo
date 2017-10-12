// cuda规约求和

#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std::chrono;

int serial_reduce(int* in, int len) {
    int sum = 0;
    for (int i = 0; i < len; ++i) {
        sum += in[i];
    }

    return sum;
}

__global__ void reduce_kernerl_v1(int* in, int* out, int len) {

    int idx = threadIdx.x;
    for (int i = len / 2; i > 0; i = i/2) {
        if (idx < len) {
            in[idx] += in[i + idx];
            __syncthreads();
        }
    }

    if (idx == 0) {
        out[blockIdx.x] = in[idx];
    }
}

int main() {
    
    const int len = 1024;
    int *h_in = new int[len];
    for (int i = 0; i < len; ++i) {
        h_in[i] = i;
    }

    int *d_in, *d_out;
    cudaMalloc((void**)&d_out, sizeof(int));
    cudaMalloc((void**)&d_in, sizeof(int)*len);
    cudaMemcpy(d_in, h_in, sizeof(int)*len, cudaMemcpyHostToDevice);
    

steady_clock::time_point start = steady_clock::now();    
    dim3 grid(1,1);
    dim3 block(1024,1);
    reduce_kernerl_v1<<<grid, block>>>(d_in, d_out, len);
    cudaDeviceSynchronize();

    int result = 0;
    cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost);
steady_clock::time_point stop = steady_clock::now();\
milliseconds time = duration_cast<milliseconds>(stop - start);
    printf("cuda reduce sum [%d], time[%ld]ms.\n", result, time.count());

start = steady_clock::now();
    int sum = serial_reduce(h_in, len); 
stop = steady_clock::now();
time = duration_cast<milliseconds>(stop - start);
    printf("serail reduce sum [%d], time[%ld]ms.\n", sum, time.count());

    delete []h_in;
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
