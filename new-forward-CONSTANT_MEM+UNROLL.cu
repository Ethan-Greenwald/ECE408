#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 32

__constant__ float constant_mask[4000];
__global__ void conv_forward_kernel(float*__restrict__ output, const float* __restrict__ input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) constant_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_size = ceil((float)Width_out/TILE_WIDTH);
    int H_size = ceil((float)Height_out/TILE_WIDTH);

    int map = blockIdx.x;
    int batch = blockIdx.y;
    int h = ceil((float)blockIdx.z / W_size) * TILE_WIDTH + threadIdx.y;    //output height
    int w = blockIdx.z % W_size * TILE_WIDTH + threadIdx.x;                 //output width
    
    /* Each thread in the block calculates its output value */
    if(h < Height_out && w < Width_out){
        float acc = 0.0f;
        #pragma unroll
        for(int c = 0; c < Channel; c++){
            #pragma unroll
            for(int p = 0; p < K; p++){
                #pragma unroll
                for(int q = 0; q < K; q++){
                    if(!(h+p > Height || w + q > Width))    //bounds check
                        acc += in_4d(batch, c, h + p, w + q) * mask_4d(map, c, p, q);
                }
            }
        }

    /* Bounds check */
        out_4d(batch, map, h, w) = acc;
    }
    #undef out_4d
    #undef in_4d
    #undef mask_4d
    /*
    Conv-GPU==
    Layer Time: 10.223 ms
    Op Time: 0.262066 ms
    Conv-GPU==
    Layer Time: 7.39614 ms
    Op Time: 1.58265 ms

    Test Accuracy: 0.72
    */
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaMalloc((void**)device_input_ptr, sizeof(float) * Height * Width * Channel * Batch);
    cudaMalloc((void**)device_output_ptr, sizeof(float) * Batch * Map_out * (Height-K+1) * (Width-K+1));

    cudaMemcpy(*device_input_ptr, host_input, sizeof(float) * Height * Width * Channel * Batch, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constant_mask, host_mask, sizeof(float) * Map_out * Channel * K * K, 0, cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    // dim3 grid(Map_out, Batch, ceil((float)(Height)/TILE_WIDTH) * ceil((float)(Width)/TILE_WIDTH));
    dim3 grid(Map_out, Batch, ceil((float)(Height)/TILE_WIDTH) * ceil((float)(Width)/TILE_WIDTH));
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<grid, block>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, sizeof(float) * Batch * Map_out * (Height-K+1) * (Width-K+1), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
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
