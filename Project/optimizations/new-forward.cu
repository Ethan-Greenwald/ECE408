#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

__constant__ float MASK[4000];
__global__ void conv_forward_kernel(float* output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) MASK[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_size = ceil((float)Width_out/TILE_WIDTH);
    int H_size = ceil((float)Height_out/TILE_WIDTH);

    int map = blockIdx.x;
    int batch = blockIdx.y;
    int h = blockIdx.z / W_size * TILE_WIDTH + threadIdx.y;    //output height
    int w = blockIdx.z % W_size * TILE_WIDTH + threadIdx.x;                 //output width
    
    /* Each thread in the block calculates its output value */
    if(h < Height_out && w < Width_out){
        float acc = 0.0f;
        #pragma unroll 2
        for(int c = 0; c < Channel; c++){
            const float* in_start = input + batch * (Channel * Height * Width) + c * (Height * Width);
            const float* mask_start = MASK + map * (Channel * K * K) + c * (K * K);

            #pragma unroll 7
            for(int p = 0; p < K; p++)
                if(!(h+p > Height))

                    #pragma unroll 7
                    for(int q = 0; q < K; q++){
                        if(!(w + q > Width))    //bounds check
                            acc += in_start[(h + p) * Width + (w + q)] * mask_start[p * K + q];
                    }
        }

        out_4d(batch, map, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}
	
__global__ void mat_mul_conv(float* output, const float* __restrict__ input, const float* __restrict__ mask, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]//feature, channel, row, col

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
                    result += maskTile[ty][i]*inputTile[i][tx];
            }
            __syncthreads();
        }

    /* Store output (with bounds check) */
    if ((row < Map_out) && (col < Width_out*Height_out))
        out_4d(batch, row, col / Width_out, col % Width_out) = result;//////////
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
    cudaMemcpyToSymbol(MASK, host_mask, sizeof(float) * Map_out * Channel * K * K);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    // get_device_properties();
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int SEG_SIZE = 100; 
    /*
    ORIGINAL: 12.59 + 32.61 = 45.20
    10:   15.18 + 32.73 = 47.91 ms
    100:  12.85 + 32.84 = 45.69 ms
    1000: 12.78 + 32.94 = 45.72 ms
    */

    dim3 grid(Map_out, SEG_SIZE, ceil((float)(Height)/TILE_WIDTH) * ceil((float)(Width)/TILE_WIDTH));
    dim3 mat_mul_grid(ceil((float)(Height_out * Width_out)/TILE_WIDTH), ceil((float)Map_out/TILE_WIDTH), SEG_SIZE);
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);

    bool isFirstLayer = Map_out < 10;
    int image_size = Channel * Height * Width;
    int output_size = Map_out * Height_out * Width_out;

    cudaStream_t stream0, stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8, stream9;
    cudaStreamCreate(&stream0); cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2); cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4); cudaStreamCreate(&stream5);
    cudaStreamCreate(&stream6); cudaStreamCreate(&stream7);
    cudaStreamCreate(&stream8); cudaStreamCreate(&stream9);
    if(isFirstLayer)
        for(int i = 0; i < Batch; i += 10 * SEG_SIZE){
            conv_forward_kernel<<<grid, block, 0, stream0>>>(device_output + (i + 0 * SEG_SIZE) * output_size, device_input + (i + 0 * SEG_SIZE) * image_size, device_mask, Batch, Map_out, Channel, Height, Width, K);
            conv_forward_kernel<<<grid, block, 0, stream1>>>(device_output + (i + 1 * SEG_SIZE) * output_size, device_input + (i + 1 * SEG_SIZE) * image_size, device_mask, Batch, Map_out, Channel, Height, Width, K);
            conv_forward_kernel<<<grid, block, 0, stream2>>>(device_output + (i + 2 * SEG_SIZE) * output_size, device_input + (i + 2 * SEG_SIZE) * image_size, device_mask, Batch, Map_out, Channel, Height, Width, K);
            conv_forward_kernel<<<grid, block, 0, stream3>>>(device_output + (i + 3 * SEG_SIZE) * output_size, device_input + (i + 3 * SEG_SIZE) * image_size, device_mask, Batch, Map_out, Channel, Height, Width, K);
            conv_forward_kernel<<<grid, block, 0, stream4>>>(device_output + (i + 4 * SEG_SIZE) * output_size, device_input + (i + 4 * SEG_SIZE) * image_size, device_mask, Batch, Map_out, Channel, Height, Width, K);
            conv_forward_kernel<<<grid, block, 0, stream5>>>(device_output + (i + 5 * SEG_SIZE) * output_size, device_input + (i + 5 * SEG_SIZE) * image_size, device_mask, Batch, Map_out, Channel, Height, Width, K);
            conv_forward_kernel<<<grid, block, 0, stream6>>>(device_output + (i + 6 * SEG_SIZE) * output_size, device_input + (i + 6 * SEG_SIZE) * image_size, device_mask, Batch, Map_out, Channel, Height, Width, K);
            conv_forward_kernel<<<grid, block, 0, stream7>>>(device_output + (i + 7 * SEG_SIZE) * output_size, device_input + (i + 7 * SEG_SIZE) * image_size, device_mask, Batch, Map_out, Channel, Height, Width, K);
            conv_forward_kernel<<<grid, block, 0, stream8>>>(device_output + (i + 8 * SEG_SIZE) * output_size, device_input + (i + 8 * SEG_SIZE) * image_size, device_mask, Batch, Map_out, Channel, Height, Width, K);
            conv_forward_kernel<<<grid, block, 0, stream9>>>(device_output + (i + 9 * SEG_SIZE) * output_size, device_input + (i + 9 * SEG_SIZE) * image_size, device_mask, Batch, Map_out, Channel, Height, Width, K);
        }
    else
        for(int i = 0; i < Batch; i += 10 * SEG_SIZE){
            mat_mul_conv<<<mat_mul_grid, block, 0, stream0>>>(device_output + (i + 0 * SEG_SIZE) * output_size, device_input + (i + 0 * SEG_SIZE) * image_size, device_mask, Map_out, Channel, Height, Width, K);
            mat_mul_conv<<<mat_mul_grid, block, 0, stream1>>>(device_output + (i + 1 * SEG_SIZE) * output_size, device_input + (i + 1 * SEG_SIZE) * image_size, device_mask, Map_out, Channel, Height, Width, K);
            mat_mul_conv<<<mat_mul_grid, block, 0, stream2>>>(device_output + (i + 2 * SEG_SIZE) * output_size, device_input + (i + 2 * SEG_SIZE) * image_size, device_mask, Map_out, Channel, Height, Width, K);
            mat_mul_conv<<<mat_mul_grid, block, 0, stream3>>>(device_output + (i + 3 * SEG_SIZE) * output_size, device_input + (i + 3 * SEG_SIZE) * image_size, device_mask, Map_out, Channel, Height, Width, K);
            mat_mul_conv<<<mat_mul_grid, block, 0, stream4>>>(device_output + (i + 4 * SEG_SIZE) * output_size, device_input + (i + 4 * SEG_SIZE) * image_size, device_mask, Map_out, Channel, Height, Width, K);
            mat_mul_conv<<<mat_mul_grid, block, 0, stream5>>>(device_output + (i + 5 * SEG_SIZE) * output_size, device_input + (i + 5 * SEG_SIZE) * image_size, device_mask, Map_out, Channel, Height, Width, K);
            mat_mul_conv<<<mat_mul_grid, block, 0, stream6>>>(device_output + (i + 6 * SEG_SIZE) * output_size, device_input + (i + 6 * SEG_SIZE) * image_size, device_mask, Map_out, Channel, Height, Width, K);
            mat_mul_conv<<<mat_mul_grid, block, 0, stream7>>>(device_output + (i + 7 * SEG_SIZE) * output_size, device_input + (i + 7 * SEG_SIZE) * image_size, device_mask, Map_out, Channel, Height, Width, K);
            mat_mul_conv<<<mat_mul_grid, block, 0, stream8>>>(device_output + (i + 8 * SEG_SIZE) * output_size, device_input + (i + 8 * SEG_SIZE) * image_size, device_mask, Map_out, Channel, Height, Width, K);
            mat_mul_conv<<<mat_mul_grid, block, 0, stream9>>>(device_output + (i + 9 * SEG_SIZE) * output_size, device_input + (i + 9 * SEG_SIZE) * image_size, device_mask, Map_out, Channel, Height, Width, K);
        }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, sizeof(float) * Batch * Map_out * (Height-K+1) * (Width-K+1), cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error after copying output back to host: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

    error = cudaGetLastError();
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
