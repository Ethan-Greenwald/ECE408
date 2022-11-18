#include <cmath>
#include <cuda_fp16.h>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
#define SHARED_TILE_WIDTH 20
//  ./rai --queue rai_amd64_exclusive -p ./Project
/*
Current optimizations:
- Shared memory matrix multiplication and input matrix unrolling (3 points)
- Kernel fusion for unrolling and matrix-multiplication (requires previous optimization) (2 points)
- Weight matrix (kernel values) in constant memory (1 point)
- Tuning with restrict and loop unrolling (considered as one optimization only if you do both) (3 points)
- Sweeping various parameters to find best values (block sizes, amount of thread coarsening) (1 point)
- Multiple kernel implementations for different layer sizes (1 point)
    > Matrix multiplication vs regular convolution, didn't really work out
- Tiled shared memory convolution (2 points)
--------------------------------------
----- Total Optimizations: 13/10 -----
--------------------------------------
*/

__constant__ float mask[4000];
__global__ void conv_forward_kernel(float* output, const float* input, const float* device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K){
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_size = ceil((float)Width_out/TILE_WIDTH);
    int H_size = ceil((float)Height_out/TILE_WIDTH);

    int map = blockIdx.x;
    int batch = blockIdx.y;
    int h = ceil((float)blockIdx.z / W_size) * TILE_WIDTH + threadIdx.y;    //output height
    int w = blockIdx.z % W_size * TILE_WIDTH + threadIdx.x;                 //output width
    /* Each thread in the block calculates its output value */
    float acc = 0.0f;
    for(int c = 0; c < Channel; c++){
        for(int p = 0; p < K; p++)
            for(int q = 0; q < K; q++){
                if(!(h+p > Height || w + q > Width))    //bounds check
                    acc += in_4d(batch, c, h + p, w + q) * mask_4d(map, c, p, q);
            }
    }
    if(h < Height_out && w < Width_out)
        out_4d(batch, map, h, w) = acc;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void shared_conv_half(float* __restrict__ output, const float* __restrict__ input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K){
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_size = ceil((float)Width_out/SHARED_TILE_WIDTH);
    int H_size = ceil((float)Height_out/SHARED_TILE_WIDTH);

    int map = blockIdx.x;
    int batch = blockIdx.y;
    int h = blockIdx.z / W_size * SHARED_TILE_WIDTH + threadIdx.y;    //output height
    int w = blockIdx.z % W_size * SHARED_TILE_WIDTH + threadIdx.x;                 //output width
    
    const float* image_start = input + batch * (Channel * Height * Width);
    const float* mask_start = mask + map * (Channel * K * K);
    int tx = threadIdx.x; int ty = threadIdx.y;

    __shared__ half tile[31][31][4];       //height x width x channels for part of an image
    // #pragma unroll
    for(int c = 0; c < Channel; c++){
        if(h < Height && w < Width)         //load value corresponding to the thread
            tile[ty][tx][c] = __float2half(image_start[(c) * (Height * Width) + (h) * (Width) + (w)]);
        if(ty == SHARED_TILE_WIDTH - 1){    //last row loads extra
            #pragma unroll
            for(int row = 1; row < K; row++)
                tile[ty+row][tx][c] = __float2half(image_start[(c) * (Height * Width) + (h+row) * (Width) + (w)]);
        }
        if(tx == SHARED_TILE_WIDTH - 1){    //last col loads extra
            #pragma unroll
            for(int col = 1; col < K; col++)
                tile[ty][tx+col][c] = __float2half(image_start[(c) * (Height * Width) + (h) * (Width) + (w+col)]);
        }
    }
    __syncthreads();

    /* Bounds check */
    if(h < Height_out && w < Width_out){
        /* Each thread in the block calculates its output value */
        half acc = __float2half(0.0);
        #pragma unroll
        for(int c = 0; c < Channel; c++){
        
            #pragma unroll
            for(int p = 0; p < K; p++){

                #pragma unroll
                for(int q = 0; q < K; q++){
                    if(!(h+p > Height || w + q > Width))    //bounds check
                        acc = __hfma(tile[ty+p][tx+q][c], __float2half(mask_start[c * (K * K) + p * K + q]), acc);
                }
            }
        }
        out_4d(batch, map, h, w) = __half2float(acc);
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void normal_conv(float* __restrict__ output, const float* __restrict__ input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K){
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]

    int W_size = ceil((float)Width_out/TILE_WIDTH);
    int H_size = ceil((float)Height_out/TILE_WIDTH);

    int feature_index = blockIdx.x;
    int image_index = blockIdx.y;
    int h = blockIdx.z / W_size * TILE_WIDTH + threadIdx.y;    //output height
    int w = blockIdx.z % W_size * TILE_WIDTH + threadIdx.x;                 //output width

    /* Bounds check */
    if(h < Height_out && w < Width_out){
        const float* pre_image_start = input + image_index * (Channel * Height * Width);
        const float* pre_mask_start = mask + feature_index * (Channel * K * K);

        /* Each thread in the block calculates its output value */
        float acc = 0.0;
        #pragma unroll
        for(int c = 0; c < Channel; c++){
            const float* image_start = pre_image_start + (c * Height * Width);
            const float* mask_start = pre_mask_start + (c * K * K);
            #pragma unroll
            for(int p = 0; p < K; p++){
                if(!(h+p > Height)){
                    #pragma unroll
                    for(int q = 0; q < K; q++){
                        if(!(w + q > Width))
                            acc += image_start[(h+p) * (Width) + (w+q)] * mask_start[p * K + q];    //can try to fix this so we get coalesced global accesses
                    }
                }
            }
        }
        out_4d(image_index, feature_index, h, w) = acc;
    }

    #undef out_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K){

    cudaMalloc((void**)device_input_ptr, sizeof(float) * Height * Width * Channel * Batch);
    cudaMalloc((void**)device_output_ptr, sizeof(float) * Batch * Map_out * (Height-K+1) * (Width-K+1));
    cudaMemcpy(*device_input_ptr, host_input, sizeof(float) * Height * Width * Channel * Batch, cudaMemcpyHostToDevice);
    
    cudaMemcpyToSymbol(mask, host_mask, sizeof(float) * Map_out * Channel * K * K, 0, cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error after prolog: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    dim3 grid(Map_out, Batch, ceil((float)(Height)/TILE_WIDTH) * ceil((float)(Width)/TILE_WIDTH));
    dim3 block(TILE_WIDTH, TILE_WIDTH);

    dim3 sharedGrid(Map_out, Batch, ceil((float)(Height)/SHARED_TILE_WIDTH) * ceil((float)(Width)/SHARED_TILE_WIDTH));
    dim3 sharedBlock(SHARED_TILE_WIDTH, SHARED_TILE_WIDTH);

    bool firstLayer = Map_out < 10;

    // if(firstLayer)
        normal_conv<<<grid, block>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
    // else
    //     shared_conv_half<<<sharedGrid, sharedBlock>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error after kernel call: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, sizeof(float) * Batch * Map_out * (Height-K+1) * (Width-K+1), cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error after copying output data: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error after epilog: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
