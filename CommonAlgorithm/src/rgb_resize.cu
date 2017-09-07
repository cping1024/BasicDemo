
/// @brief 数字图像双线性插值

#include <stdio.h>

#include <string>
#include <chrono>

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std::chrono;

int divUp(int N, int M) { return (N - 1) / M + 1;}

__global__ void rgb_resize_kenerl(const unsigned char* in, unsigned char* out, int i_w, int i_h, int o_w, int o_h, float x_scale, float y_scale) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    if (idx >= o_w || idy >= o_h) {
        return;
    }

    float fx = idx * x_scale;
    float fy = idy * y_scale;
    int channel = idz;

    int px1 = int(fx);
    int py1 = int(fy);
    int px2 = px1 + 1;
    int py2 = py1;
    int px3 = px1;
    int py3 = py1 + 1;
    int px4 = px2;
    int py4 = py3;

    float pv1 = abs(fx - px1) * abs(fy - py1);
    float pv2 = abs(fx - px2) * abs(fy - py2);
    float pv3 = abs(fx - px3) * abs(fy - py3);
    float pv4 = abs(fx - px4) * abs(fy - py4);

    int offset = (idy * o_w + idx) * 3 + idz;
    int offset1 = (py1 * i_w + px1) * 3 + idz;
    int offset2 = (py2 * i_w + px2) * 3 + idz;
    int offset3 = (py3 * i_w + px3) * 3 + idz;
    int offset4 = (py4 * i_w + px4) * 3 + idz;
    
    float value = pv1 * in[offset1] + pv2 * in[offset2] + pv3 * in[offset3] + pv4 * in[offset4];
    //int value = 0.25 * in[offset1] + 0.25 * in[offset2] + 0.25 * in[offset3] + 0.25 * in[offset4];
    out[offset] = value;
}

void rgb_resize_cpu(unsigned char* src, unsigned char* des, int iw, int ih, int ow, int oh, float x, float y)
{
    if (!src || !des) {
        return; 
    }
    
    for (int i = 0; i < oh; ++i) {
        
        for (int j = 0; j < ow; ++j) {
            float fx = x * j;
            float fy = y * i;
            
            for (int n = 0; n < 3; ++n) {
                
                int offset1 = int(fy) * iw * 3 + int(fx) * 3 + n;
                int offset2 = int(fy) * iw * 3 + (int(fx) + 1) * 3 + n;
                int offset3 = (int(fy) + 1) * iw * 3 + int(fx) * 3 + n;
                int offset4 = (int(fy) + 1) * iw * 3 + (int(fx) + 1) * 3 + n;
                des[(i * ow  + j) * 3 + n] = 0.7 * src[offset1] + 0.1 * src[offset2] + 0.1 * src[offset3] + 0.1 * src[offset4];  
                //des[(i * ow + j) * 3 + n] = src[offset1];
            }
        }
    }

}


int main(int argc, char* argv[]) {

    if (argc != 2) { 
        printf("usage:./application <image filename>.\n");
        return -1;    
    }

    cv::Mat image = cv::imread(std::string(argv[1]));
    if (!image.data) {
        return -1;
    }

    const int N = 16;
    const int des_w = 1280;
    const int des_h = 720;
    const float x_scale = float(image.cols) / float(des_w);
    const float y_scale = float(image.rows) / float(des_h);

    void* src;
    const int len = image.cols * image.rows * image.channels();
    cudaMalloc(&src, len);
    cudaMemcpy(src, image.data, len, cudaMemcpyHostToDevice);

    void* rgb;
    cudaMalloc(&rgb, des_w * des_h * 3);

steady_clock::time_point start = steady_clock::now();
    dim3 block(N, N, 1);
    dim3 grid(divUp(des_w, N), divUp(des_h, N), 3);
    rgb_resize_kenerl<<<grid, block>>>((const unsigned char*)src, (unsigned char*)rgb, image.cols, image.rows, des_w, des_h, x_scale, y_scale);
    cudaDeviceSynchronize();
steady_clock::time_point stop = steady_clock::now();
milliseconds time = duration_cast<milliseconds>(stop - start);
printf("gpu resize time[%ld]ms.\n", time.count());

    cv::Mat resize_image;
    resize_image.create(cv::Size(des_w, des_h), image.type());
    cudaMemcpy(resize_image.data, rgb, des_w * des_h * 3, cudaMemcpyDeviceToHost);

    cv::imshow("src", image);
    cv::waitKey();

    cv::imshow("resize", resize_image);
    cv::waitKey();

    cv::Mat cpu_resize_image;
    cpu_resize_image.create(cv::Size(des_w, des_h), image.type());

start = steady_clock::now();
    rgb_resize_cpu(image.data, cpu_resize_image.data, image.cols, image.rows, des_w, des_h, x_scale, y_scale);
stop = steady_clock::now();
time = duration_cast<milliseconds>(stop - start);
printf("cpu resize time[%ld]ms.\n", time.count());    

    cv::imshow("cpu", cpu_resize_image);
    cv::waitKey();

    cudaFree(rgb);
    cudaFree(src);

    return 0;
}

