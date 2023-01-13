#include <cmath>
#include <cuda_fp16.h>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
/*
16 | 41.7008ms
13 | 38.2554ms

#define K 7:
Layer Time: 6.81766 ms
Op Time: 0.129979 ms
Conv-GPU==
Layer Time: 5.36077 ms
Op Time: 0.281237 ms

with starts:
Op Time: 11.9137 ms
Conv-GPU==
Layer Time: 459.165 ms
Op Time: 26.3417 ms
*/
__constant__ float MASK[4000];
__global__ void conv_forward_kernel(float* output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width)
{


    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) MASK[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define tile_3d(i2, i1, i0) inputTile[(i2) * (shared_width*shared_width) + (i1) * (shared_width) + (i0)]
    #define K 7

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    extern __shared__ float inputTile[];
    int shared_width = TILE_WIDTH + K - 1;

    int W_size = ceil((float)Width_out/TILE_WIDTH);
    
    int map = blockIdx.x;
    int batch = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;

    int h = blockIdx.z / W_size * TILE_WIDTH + ty;    //output height
    int w = blockIdx.z % W_size * TILE_WIDTH + tx;    //output width
    int h_base = h - ty;
    int w_base = w - tx;

    #pragma unroll 2
    for(int row = ty; row < shared_width; row += TILE_WIDTH)
        #pragma unroll 2
        for(int col = tx; col < shared_width; col += TILE_WIDTH)
            if((h_base + row) < Height && (w_base + col) < Width)
                 tile_3d(tz, row, col) = in_4d(batch, tz, h_base + row, w_base + col);
    __syncthreads();

    if(h < Height_out && w < Width_out){
        float acc = 0.0;
        // const float* tile_start = &tile_3d(tz, ty, tx);
        // const float* mask_start = &mask_4d(map, tz, 0, 0);
        #pragma unroll 7
        for(int p = 0; p < K; p++){
            
            #pragma unroll 7
            for(int q = 0; q < K; q++)
                acc += tile_3d(tz, ty + p, tx + q) * mask_4d(map, tz, p, q); //tile_start[p*shared_width + q] * mask_start[p*K + q];
        }
        // out_4d(batch, map, h, w) = acc;
        atomicAdd((&out_4d(batch, map, h, w)), acc);
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef tile_3d
    #undef K
}
	
__global__ void mat_mul_conv(float* output, const float* __restrict__ input, const float* __restrict__ mask, const int Map_out, const int Channel, const int Height, const int Width) {
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]//feature, channel, row, col
    #define K 7
    __shared__ float maskTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float inputTile[TILE_WIDTH][TILE_WIDTH];
    
    /* General util variables */
    int batch = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int numACol = Channel * K * K;

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    
    float result = 0.0;

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
                maskTile[ty][tx] = mask_4d(row, c, h, w);////////
            else 
                maskTile[ty][tx] = 0.0;

            c = load_row / (K * K);
            h = col / Width_out;
            w = col % Width_out;
            
            int p = load_row % (K * K) / K;     //offset based on mask row
            int q = (load_row % (K * K)) % K;   //offset based on mask col

            /* Load in input tile value */
            if (load_row < numACol && col < Height_out*Width_out) 
                inputTile[ty][tx] = in_4d(batch, c, h+p, w+q);/////////////
            else 
                inputTile[ty][tx] = 0.0;
            __syncthreads();

            /* Calculate partial dot product */
            /*No unroll:    Layer Time: 648.712 ms
                            Op Time: 12.6689 ms
                            Conv-GPU==
                            Layer Time: 477.777 ms
                            Op Time: 32.6175 ms
            */
            if((row < Map_out) && (col < Width_out*Height_out)){
                #pragma unroll 16
                for(int i = 0; i < TILE_WIDTH; i++)
                    result += maskTile[ty][i] * inputTile[i][tx];
            }
            __syncthreads();
        }

    /* Store output (with bounds check) */
    if ((row < Map_out) && (col < Width_out*Height_out))
        out_4d(batch, row, col / Width_out, col % Width_out) = result;//////////
    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef K
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
    cudaMemcpyToSymbol(MASK, host_mask, sizeof(float) * Map_out * Channel * K * K);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    // get_device_properties();
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    dim3 grid(Map_out, Batch, ceil((float)(Height_out)/TILE_WIDTH) * ceil((float)(Width_out)/TILE_WIDTH));
    dim3 reductionBlock(TILE_WIDTH, TILE_WIDTH, Channel);
    dim3 mat_mul_grid(ceil((float)(Height_out * Width_out)/TILE_WIDTH), ceil((float)Map_out/TILE_WIDTH), Batch);
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);

    bool isFirstLayer = Map_out < 10;
    if(isFirstLayer)
        conv_forward_kernel<<<grid, reductionBlock, sizeof(float) * Channel * (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1)>>>(device_output, device_input, NULL, Batch, Map_out, Channel, Height, Width);
    else
        mat_mul_conv<<<mat_mul_grid, block>>>(device_output, device_input, device_mask, Map_out, Channel, Height, Width);
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
