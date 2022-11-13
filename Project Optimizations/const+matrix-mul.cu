#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 32

__constant__ float mask[5000];
__global__ void mat_mul_conv(const float *input, float *output, const int Channel, const int K, const int Map_out, const int Width, const int Height) {
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    __shared__ float maskTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float inputTile[TILE_WIDTH][TILE_WIDTH];

    /* General util variables */
    int batch = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int numACol = Channel*K*K;

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    
    float result = 0;

    int numTiles = ceil((float)numACol/TILE_WIDTH);
    
    for (int i = 0; i < numTiles; i++) {
        /* Unroll and load in mask/input tiles */
        int load_col = i * TILE_WIDTH + tx;
        int load_row = i * TILE_WIDTH + ty;

        int c = load_col / (K * K);           //channel = column / mask_size
        int h = (load_col % (K * K)) / K;     //row = (column % mask_size) / mask_size
        int w = (load_col % (K * K)) % K;     //col = (column % mask_size) % mask_size

        /* Load in mask tile value */
        if (load_col < numACol && row < Map_out) 
            maskTile[ty][tx] = mask_4d(row, c, h, w);
        else 
            maskTile[ty][tx] = 0;

        c = load_row / (K * K);
        h = col / Width_out;
        w = col % Width_out;
        
        int p = load_row % (K * K) / K;     //offset based on mask row
        int q = (load_row % (K * K)) % K;   //offset based on mask col

        /* Load in input tile value */
        if (load_row < numACol && col < Height_out*Width_out) 
            inputTile[ty][tx] = in_4d(batch, c, h+p, w+q);
        else 
            inputTile[ty][tx] = 0;
        __syncthreads();

        /* Calculate partial dot product */
        for(int i = 0; i < TILE_WIDTH; i++)
            result += maskTile[ty][i] * inputTile[i][tx];
        __syncthreads();
    }

    /* Store output (with bounds check) */
    if ((row < Map_out) && (col < Width_out*Height_out))
        out_4d(batch, row, col / Width_out, col % Width_out) = result;
}

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    /*  
    dim3 grid(Map_out, Batch, ceil((float)(Height)/TILE_WIDTH)) * ceil((float)(Width)/TILE_WIDTH);
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    */
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

    /* Bounds check */
    if(h < Height_out && w < Width_out)
        out_4d(batch, map, h, w) = acc;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaMalloc((void**)device_input_ptr, sizeof(float) * Height * Width * Channel * Batch);
    cudaMalloc((void**)device_output_ptr, sizeof(float) * Batch * Map_out * (Height-K+1) * (Width-K+1));
    cudaMalloc((void**)device_mask_ptr, sizeof(float) * Map_out * Channel * K * K);

    cudaMemcpy(*device_input_ptr, host_input, sizeof(float) * Height * Width * Channel * Batch, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Map_out * Channel * K * K, cudaMemcpyHostToDevice);
    
    cudaMemcpyToSymbol(mask, *device_mask_ptr, sizeof(float) * Map_out * Channel * K * K);

}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    dim3 grid(ceil((float)(Height_out * Width_out)/TILE_WIDTH), ceil((float)Map_out/TILE_WIDTH), Batch);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    mat_mul_conv<<<grid, block>>>(device_input, device_output, Channel, K, Map_out, Width, Height);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, sizeof(float) * Batch * Map_out * (Height-K+1) * (Width-K+1), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

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
