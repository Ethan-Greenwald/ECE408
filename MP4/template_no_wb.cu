
//@@ Define any useful program-wide constants here
#define TILE_WIDTH 16
#define MASK_WIDTH 3

//@@ Define constant memory for device kernel here
__constant__ float* deviceKernel;

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  //Strategy 2
  __shared__ float tile[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int tz = threadIdx.z;

  int outputRow = blockIdx.y * TILE_WIDTH + ty;
  int outputCol = blockIdx.x * TILE_WIDTH + tx;
  int outputDepth = blockIdx.z * TILE_WIDTH + tz;

  int inputRow = outputRow - (MASK_WIDTH / 2);
  int inputCol = outputCol - (MASK_WIDTH / 2);
  int inputDepth = outputDepth - (MASK_WIDTH / 2);

  float outputValue = 0.0f;

  /* Load in 3D tile */
  if(inputRow >= 0 && inputRow < y_size && 
     inputCol >= 0 && inputCol < x_size && 
     inputDepth >= 0 && inputDepth < z_size)
      tile[ty][tx][tz] = input[inputDepth*(y_size*x_size) + inputRow*x_size + inputCol];
  else
    tile[ty][tx][tz] = 0.0f;

  __syncthreads();

  /* Calculate output value (if valid thread coordinates) */
  if(ty < TILE_WIDTH && tx < TILE_WIDTH && tz < TILE_WIDTH){
    for(int i = 0; i < MASK_WIDTH; i++)
      for(int j = 0; j < MASK_WIDTH; j++)
        for(int k = 0; k < MASK_WIDTH; k++)
          outputValue += deviceKernel[k*MASK_WIDTH*MASK_WIDTH + i*MASK_WIDTH + j] * tile[i+ty][j+tx][k+tz];
  }
  /* Set output value if valid output coordinates */
  if(outputRow < y_size && outputCol < x_size && outputDepth < z_size)
    output[outputDepth*(y_size*x_size) + outputRow*x_size + outputCol] = outputValue;
}

int main(int argc, char *argv[]) {
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;


  // Import data
  inputLength = 100;
  kernelLength = 27;
  hostInput = NULL;
  hostKernel = NULL;
  hostOutput = NULL;

  // First three elements are the input dimensions
  // z_size = hostInput[0];
  // y_size = hostInput[1];
  // x_size = hostInput[2];
  z_size = 10;
  y_size=10;
  x_size=10;
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**)&deviceInput, sizeof(float)*(inputLength-3));
  cudaMalloc((void**)&deviceOutput, sizeof(float)*(inputLength-3));

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, (hostInput+3), sizeof(float)*(inputLength-3), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, sizeof(float)*kernelLength);

  //@@ Initialize grid and block dimensions here

  dim3 grid(ceil(x_size/(float)TILE_WIDTH), ceil(y_size/(float)TILE_WIDTH), ceil(z_size/(float)TILE_WIDTH));
  dim3 block(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);  //block covers input including halo cells

  //@@ Launch the GPU kernel here
  conv3d<<<grid, block>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy((hostOutput+3), deviceOutput, sizeof(float)*(inputLength-3), cudaMemcpyDeviceToHost);

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
