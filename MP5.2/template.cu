// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

int ceil(int x, int y){
    return (x + y - 1) / y;
}

__global__ void add(float* input, float* output, float* sums, int len){
  int bx = blockIdx.x;
  int start = (bx*BLOCK_SIZE*2) + threadIdx.x;
  
  /* Calculate value to increment this block by*/
  __shared__ float incrementVal;
  if(threadIdx.x == 0)
    incrementVal = blockIdx.x == 0 ? 0 : sums[blockIdx.x - 1];
  __syncthreads();

  /* Each thread increments 2 elements */
  if(start < len)
    output[start] = input[start] + incrementVal;

  if(start + BLOCK_SIZE < len)
    output[start + BLOCK_SIZE] = input[start + BLOCK_SIZE] + incrementVal;
}

__global__ void scan(float *input, float *output, float* auxArray, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  __shared__ float T[2*BLOCK_SIZE];

  /* Load in T (each thread loads 2 elements) */
  int idx = (blockIdx.x * BLOCK_SIZE)*2 + threadIdx.x;
  if(idx < len)
    T[threadIdx.x] = input[idx];
  else
    T[threadIdx.x] = 0.0;
  
  if(idx + BLOCK_SIZE < len)
    T[threadIdx.x + BLOCK_SIZE] = input[idx + BLOCK_SIZE];
  else
    T[threadIdx.x + BLOCK_SIZE] = 0.0;


  /* Perform Scan */
  int stride = 1;
  while(stride < 2*BLOCK_SIZE){
    __syncthreads();
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0)
      T[index] += T[index-stride];
    stride *= 2;
  }

  /* Perform Post-Scan */
  stride = BLOCK_SIZE/2;
  while(stride > 0){
    __syncthreads();
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if((index+stride) < 2*BLOCK_SIZE)
      T[index + stride] += T[index];
    stride /= 2;
  }

  /* Each thread sets corresponding 2 output elements if not OOB */
  __syncthreads();
  if(idx < len)
    output[idx] = T[threadIdx.x];
  if(idx + BLOCK_SIZE < len)
    output[idx + BLOCK_SIZE] = T[threadIdx.x + BLOCK_SIZE];

  /* Save block reductions to auxillary array */
  if(auxArray != NULL && threadIdx.x == 0)
    auxArray[blockIdx.x] = T[2*BLOCK_SIZE-1];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float* deviceBuffer;
  float* deviceSums;
  float* deviceScannedSums;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  int numBlocks = ceil((float)numElements/(2*BLOCK_SIZE));

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));

  wbCheck(cudaMalloc((void **)&deviceBuffer, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceSums, 2 * BLOCK_SIZE * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScannedSums, 2 * BLOCK_SIZE * sizeof(float)));


  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 grid(numBlocks, 1, 1);
  dim3 single(1, 1, 1);
  dim3 block(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the device

  /* Perform initial scan to deviceBuffer. deviceSums contains the sum (reduction) of each block */
  scan<<<grid, block>>>(deviceInput, deviceBuffer, deviceSums, numElements);
  cudaDeviceSynchronize();

  /* Scan the deviceSums to get offsets for each block to correct them */
  scan<<<single, block>>>(deviceSums, deviceScannedSums, NULL, numBlocks);
  cudaDeviceSynchronize();

  /* Add offsets in deviceScannedSums to corresponding blocks in deviceBuffer. Final output stored in deviceOutput */
  add<<<grid, block>>>(deviceBuffer, deviceOutput, deviceScannedSums, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceSums);
  cudaFree(deviceBuffer);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}