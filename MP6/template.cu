// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32

//@@ insert code here
/*
Cast the image to unsigned char

Convert the image from RGB to Gray Scale. You will find one of the lectures and textbook chapters helpful.

Compute the histogram of the image

Compute the scan (prefix sum) of the histogram to arrive at the histogram equalization function

Apply the equalization function to the input image to get the color corrected image
*/

__global__ void floatToUChar(float* input, unsigned char* output, int width, int height){
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int idx = (blockIdx.z * width * height) + (row * width) + col;
  if(col < width && row < height)
    output[idx] = (unsigned char)(255 * input[idx]);
}

__global__ void RGBToGreyscale(unsigned char* input, unsigned char* output, int width, int height){
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int idx = (row * width) + col;

  if(col < width && row < height){
    unsigned char r = input[3*idx];
    unsigned char g = input[3*idx + 1];
    unsigned char b = input[3*idx + 2];
    output[idx] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void computeHistogram(unsigned char* image, int* output, int width, int height){

  /* Initialize histogram array in memory */
  __shared__ int histogram[HISTOGRAM_LENGTH];
  int idx = threadIdx.y * blockDim.x + threadIdx.x;
  if(idx < HISTOGRAM_LENGTH)
    histogram[idx] = 0;
  __syncthreads();

  /* Calculate histogram */
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if(col < width && row < height){
    int val = image[row*width + col];
    atomicAdd(&histogram[val], 1);
  }
  __syncthreads();

  /* Store histogram to output */
  if(idx < HISTOGRAM_LENGTH)
    atomicAdd(&(output[idx]), histogram[idx]);// output[idx] = histogram[idx];
}

__global__ void computeCDF(int* histogram, float* output, int width, int height){

  /* Initialize shared CDF array */
  __shared__ int CDF[HISTOGRAM_LENGTH];

  int idx = threadIdx.x;
  CDF[idx] = histogram[idx];

  /* Perform Scan */
  int stride = 1;
  while(stride <= HISTOGRAM_LENGTH/2){
    __syncthreads();
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if(index < HISTOGRAM_LENGTH)
      CDF[index] += CDF[index-stride];
    stride *= 2;
  }

  /* Perform Post-Scan */
  stride = HISTOGRAM_LENGTH/4;
  while(stride > 0){
    __syncthreads();
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if((index+stride) < HISTOGRAM_LENGTH)
      CDF[index + stride] += CDF[index];
    stride /= 2;
  }

  /* Each thread sets output element */
  __syncthreads();
  if(idx < HISTOGRAM_LENGTH)
    output[idx] = CDF[idx]/((float)(width*height));
}

__global__ void equalizeImage(unsigned char* image, float* CDF, int width, int height){
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int idx = (blockIdx.z * width * height) + (row * width) + col;
  float cdfMin = CDF[0];

  if(col < width && row < height)
    image[idx] = min(max(255*(CDF[image[idx]] - cdfMin)/(1.0-cdfMin), 0.0), 255.0);
}

__global__ void uCharToFloat(unsigned char* input, float* output, int width, int height){
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int idx = (blockIdx.z * width * height) + (row * width) + col;
  if(col < width && row < height)
    output[idx] = (float)(input[idx]/255.0);
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float* d_image;
  unsigned char* d_uchar_image;
  unsigned char* d_greyscale;
  int* d_histogram;
  float* d_CDF;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  
  /* Allocate device memory and copy input data */
  cudaMalloc((void**)&d_image, sizeof(float) * imageWidth * imageHeight * imageChannels);
  cudaMalloc((void**)&d_uchar_image, sizeof(unsigned char) * imageWidth * imageHeight * imageChannels);
  cudaMalloc((void**)&d_greyscale, sizeof(unsigned char) * imageWidth * imageHeight);

  cudaMalloc((void**)&d_histogram, sizeof(int)*HISTOGRAM_LENGTH);
  cudaMemset((void*)d_histogram, 0, HISTOGRAM_LENGTH*sizeof(int));//
  cudaMalloc((void**)&d_CDF, sizeof(float)*HISTOGRAM_LENGTH);

  cudaMemcpy(d_image, hostInputImageData, sizeof(float) * imageWidth * imageHeight * imageChannels, cudaMemcpyHostToDevice);
  
  /* Define dimensions */
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid3channel(ceil((float)imageWidth/BLOCK_SIZE), ceil((float)imageHeight/BLOCK_SIZE), imageChannels);
  dim3 grid1channel(ceil((float)imageWidth/BLOCK_SIZE), ceil((float)imageHeight/BLOCK_SIZE), 1);
  dim3 single(1,1,1);
  dim3 hist(HISTOGRAM_LENGTH, 1, 1);

  /* Run Kernels */
  floatToUChar<<<grid3channel, block>>>(d_image, d_uchar_image, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  RGBToGreyscale<<<grid1channel, block>>>(d_uchar_image, d_greyscale, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  computeHistogram<<<grid1channel, block>>>(d_greyscale, d_histogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  computeCDF<<<single, hist>>>(d_histogram, d_CDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  equalizeImage<<<grid3channel, block>>>(d_uchar_image, d_CDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  uCharToFloat<<<grid3channel, block>>>(d_uchar_image, d_image, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, d_image, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);
  
  return 0;
}