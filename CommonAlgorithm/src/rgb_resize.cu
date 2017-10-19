
/// @brief 数字图像双线性插值

#include <stdio.h>

#include <iostream>
#include <string>
#include <chrono>

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std::chrono;

typedef struct rect {
    int x;
    int y;
    int w;
    int h;
} sn_rect;

typedef struct size {
    int width;
    int height;
} sn_size;

typedef struct color{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} sn_color;

int divUp(int N, int M) { return (N - 1) / M + 1;}

__global__ void rgb_resize_kenerl(const unsigned char* in, unsigned char* out, int i_w, int i_h, int o_w, int o_h, float x_scale, float y_scale) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    if (idx >= o_w || idy >= o_h) {
        return;
    }

    float fx = (idx + 0.5) * i_w / o_w - 0.5;
    float fy = (idy + 0.5) * i_h / o_h - 0.5;

    int px1 = int(fx);
    int py1 = int(fy);
    int px2 = px1 + 1;
    if (px2 >= iw) {
        --px2;
    }
    int py2 = py1;
    int px3 = px1;
    int py3 = py1 + 1;
    if (py3 >= ih) {
        --py3;
    }
    int px4 = px2;
    int py4 = py3;

    float pv1 = fabs(px2 - fx) * fabs(py3 - fy);
    float pv2 = fabs(px1 - fx) * fabs(py3 - fy);
    float pv3 = fabs(px2 - fx) * fabs(fy - py1);
    float pv4 = fabs(px1 - fx) * fabs(fy - py1);
    int offset = (idy * o_w + idx) * 3 + idz;
    int offset1 = (py1 * i_w + px1) * 3 + idz;
    int offset2 = (py2 * i_w + px2) * 3 + idz;
    int offset3 = (py3 * i_w + px3) * 3 + idz;
    int offset4 = (py4 * i_w + px4) * 3 + idz;
    
    out[offset] = pv1 * in[offset1] + pv2 * in[offset2] + pv3 * in[offset3] + pv4 * in[offset4];
}

void sn_gpu_resize(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& des, const cv::Size& size) {

    if (!src.data) {
        return;
    }

    void *d_src, *d_des;
    int channel = src.channels();
    int owidth = size.width;
    int oheight = size.height;
    cudaMalloc(&d_src, src.cols * src.rows * channel);
    cudaMalloc(&d_des, owidth * oheight * channel);
    cudaMemcpy2D(d_src, src.cols * channel, src.data, src.step, src.cols * channel, src.rows, cudaMemcpyDefault);
    
    const int N = 16;
    dim3 block(N, N, 1);
    dim3 grid(divUp(owidth, N), divUp(oheight, N), 3);
    float scalex = float(src.cols) / float(size.width);
    float scaley = float(src.rows) / float(size.height);
    rgb_resize_kenerl<<<grid, block>>>((const unsigned char*)d_src, (unsigned char*)d_des, src.cols, src.rows, owidth, oheight, scalex, scaley);
    cudaDeviceSynchronize();
    
    cv::Size osize(owidth, oheight);
    if (des.size() != osize) {
        des.create(osize, src.type());
    }

    cudaMemcpy2D(des.data, des.step, d_des, owidth * channel, owidth * channel, oheight, cudaMemcpyDefault);

    cudaFree(d_des);
    cudaFree(d_src);
}

__global__ void copyMakeBorder_kernel(const unsigned char* in, unsigned char* out,
                                        int iw, int ih,
                                        int ow, int oh,
                                        int left, int top, int right, int bottom,
                                        unsigned char b, unsigned char g, unsigned char r )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= ow || idy >= oh) {
        return;
    }

    int offsetx = idx - left;
    int offsety = idy - top;
    int offset = (idy * ow + idx) * 3;
    int offset1 = (offsety * iw + offsetx) * 3;

    if ((0 <= offsetx) && (offsetx <= iw) && (0 <= offsety) && (offsety <= ih)) {
        out[offset] = in[offset1];
        out[offset + 1] = in[offset1 + 1];
        out[offset + 2] = in[offset1 + 2];
    } else {

        out[offset] = b;
        out[offset + 1] = g;
        out[offset + 2] = r;
    }
}

void copyMakeborder_cpu(const unsigned char* in, unsigned char* out, sn_size* isize, int left, int top, int right, int bottom, sn_size* osize, sn_color* color) {

    if (!in || !out) {
        return;
    }

    for (int i = 0; i < osize->height; ++i) {
        for (int j = 0; j < osize->width; ++j) {
           
           int offsetx = j - left;
           int offsety = i - top;
           if ((0 <= offsetx) && (offsetx <= isize->width) && (0 <= offsety) && (offsety <= isize->height)) {
                out[(i * osize->width + j) * 3] = in[(offsety * isize->width + offsetx) * 3];
                out[(i * osize->width + j) * 3 + 1] = in[(offsety * isize->width + offsetx) * 3 + 1];
                out[(i * osize->width + j) * 3 + 2] = in[(offsety * isize->width + offsetx) * 3 + 2];
           } else { 
                out[(i * osize->width + j) * 3] = color->b;
                out[(i * osize->width + j) * 3 + 1] = color->g;
                out[(i * osize->width + j) * 3 + 2] = color->r;
           }
        }
    }
}

void rgb_resize_cpu(const unsigned char* src, unsigned char* des, int iw, int ih, int ow, int oh, float x, float y)
{
    if (!src || !des) {
        return; 
    }
    
    for (int i = 0; i < oh; ++i) {        
        float fy = (i + 0.5) * ih / oh - 0.5;
        for (int j = 0; j < ow; ++j) {
            float fx = (j + 0.5) * iw / ow - 0.5;

            int px1 = int(fx);
            int py1 = int(fy);
            int px2 = px1 + 1;
            if (px2 >= iw) {
                --px2;
            }

            int py2 = py1;
            int px3 = px1;
            int py3 = py1 + 1;
            if (py3 >= ih) {
                --py3;
            }
            int px4 = px2;
            int py4 = py3;

            float pv1 = fabs(px2 - fx) * fabs(py3 - fy);
            float pv2 = fabs(px1 - fx) * fabs(py3 - fy);
            float pv3 = fabs(px2 - fx) * fabs(fy - py1);
            float pv4 = fabs(px1 - fx) * fabs(fy - py1);

            for (int n = 0; n < 3; ++n) {
                int offset1 = py1 * iw * 3 + px1 * 3 + n;
                int offset2 = py2 * iw * 3 + px2 * 3 + n;
                int offset3 = py3 * iw * 3 + px3 * 3 + n;
                int offset4 = py4 * iw * 3 + px4 * 3 + n;
                des[(i * ow  + j) * 3 + n] = pv1 * src[offset1] + pv2 * src[offset2] + pv3 * src[offset3] + pv4 * src[offset4];
            }
        }
    }

}

void sn_cpu_resize(const cv::Mat& src, cv::Mat& des, const cv::Size& size){
    if (!src.data) {
        return;
    }

    float scalex = float(src.cols) / float(size.width);
    float scaley = float(src.rows) / float(size.height);
    if (des.size() != size) {
        des.create(size, src.type());
    }

    rgb_resize_cpu(src.data, des.data, src.cols, src.rows, size.width, size.height, scalex, scaley);    
}

void test_resize_kernel(std::string& filename){
    cv::Mat image = cv::imread(filename);
    if (!image.data) {
        return;
    }

    const int N = 16;
    const int des_w = 320;
    const int des_h = 180;
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
}

void test_copymakeborder(std::string& filename){
    cv::Mat image = cv::imread(filename);
    if (!image.data) {
        printf("read image fail.\n");
        return;
    }

    const int N = 16;
    const int left = 10;
    const int top = 10;
    const int right = 10;
    const int bottom = 10;
    
    int owidth = image.cols + left + right;
    int oheight = image.rows + top + bottom;

    void *src;
    cudaMalloc(&src, image.cols * image.rows * image.channels());
    cudaMemcpy(src, image.data, image.cols * image.rows * image.channels(), cudaMemcpyHostToDevice);

    void *des;
    cudaMalloc(&des, owidth * oheight * image.channels());

    dim3 block(N, N);
    dim3 grid(divUp(owidth, N), divUp(oheight, N));
    copyMakeBorder_kernel<<<grid, block>>>((const unsigned char*)src, (unsigned char*)des, image.cols, image.rows, owidth, oheight, left, top, right, bottom, 0, 0, 255);
    cudaDeviceSynchronize();

    cv::Mat border_image;
    border_image.create(cv::Size(owidth, oheight), image.type());
    //copyMakeborder_cpu((const unsigned char*)image.data, (unsigned char*)border_image.data, &isize, left, top, right, bottom, &osize, &color);
    cudaMemcpy(border_image.data, des, owidth * oheight * 3, cudaMemcpyDeviceToHost);
    cv::imshow("border", border_image);
    cv::waitKey();

    cv::imshow("src", image);
    cv::waitKey();

    cudaFree(src);
    cudaFree(des);
}

void test_sn_gpu_resize(const std::string& filename, float scale, int row) {
    
    cv::Mat image = cv::imread(filename);
    if (!image.data) {
        return;
    }

    cv::gpu::GpuMat gpu_mat(image);
    
    printf("gpu mat width[%d], height[%d], step[%d].\n", gpu_mat.cols, gpu_mat.rows, gpu_mat.step);
    const int count = 1;    
    //float scale = 0.33f;
    int width = image.cols * scale;
    int height = image.rows * scale;

    steady_clock::time_point start = steady_clock::now();
    for (int i = 0; i < count; ++i) {
        cv::gpu::GpuMat resize_image;
        sn_gpu_resize(gpu_mat, resize_image, cv::Size(width, height));    
    }
    steady_clock::time_point stop = steady_clock::now();
    milliseconds time = duration_cast<milliseconds>(stop - start);
    printf("sn gpu resize avg time[%f]ms.\n", (time.count() * 1.0f) / count);

    cv::gpu::GpuMat resize_image;
    sn_gpu_resize(gpu_mat, resize_image, cv::Size(width, height));

    cv::Mat temp;
    //sn_cpu_resize(image, temp, cv::Size(width, height));
    resize_image.download(temp);
    cv::imshow("temp", temp);
    cv::waitKey();
    std::cout << "gpu resize image:" << temp.row(row) << std::endl;
    printf("resize image w[%d], h[%d], scale[%f], show row[%d].\n", temp.cols, temp.rows, scale, row);

    cv::imshow("src", image);
    cv::waitKey();
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        printf("usage:./application <image filename> scale row_num.\n");
        return -1;    
    }

    std::string filename(argv[1]);	
    //test_resize_kernel(filename);

    //test_copymakeborder(filename);

    float scale = atof(argv[2]);
    int row = atof(argv[3]);
    test_sn_gpu_resize(argv[1], scale, row);
    return 0;
}

