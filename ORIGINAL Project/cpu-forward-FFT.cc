#include "cpu-new-forward.h"
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;
using cf = complex<float>;
const float PI = acos(-1);

void fft(vector<cf> & a, bool invert) {
    int n = a.size();
    if (n == 1)
        return;

    vector<cf> a0(n / 2), a1(n / 2);
    for (int i = 0; 2 * i < n; i++) {
        a0[i] = a[2*i];
        a1[i] = a[2*i+1];
    }
    fft(a0, invert);
    fft(a1, invert);

    double ang = 2 * PI / n * (invert ? -1 : 1);
    cf w(1), wn(cos(ang), sin(ang));
    for (int i = 0; 2 * i < n; i++) {
        a[i] = a0[i] + w * a1[i];
        a[i + n/2] = a0[i] - w * a1[i];
        if (invert) {
            a[i] /= 2;
            a[i + n/2] /= 2;
        }
        w *= wn;
    }
}

void padZeros(vector<float>& input, int length){
  while(input.size() < length)
    input.push_back(0.0);
}

vector<float> fftConvolution(vector<float> input, vector<float> mask){
  int outLength = input.size() + mask.size() - 1;
  int pow2 = (int)log2(outLength - 1) + 1;
  cout << "pow2=" << pow2 << ", outLength=" << outLength << endl;
  padZeros(input, pow2);
  padZeros(mask, pow2);

  vector<cf> complexInput(pow2);
  vector<cf> complexMask(pow2)
  for(int i = 0; i < pow2; i++){
    complexInput[i] = input[i];
    complexMask[i] = mask[i];
  }

  cout << "   " << input[0] << " | " << complexInput[0] << endl; 
  fft(complexInput);
  fft(complexMask);

  vector<cf> complexOutput;
  for(int i = 0; i < (int)complexInput.size(); i++)
    complexOutput.push_back(complexInput[i]*complexMask[i]);
  fft(complexOutput, true);

  vector<float> realOutput;
  for(int i = 0; i < (int)complexOutput.size(); i++)
    realOutput[i] = real(complexOutput[i]);
  realOutput.resize(outLength);
  return realOutput;
}

void conv_forward_cpu(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
  /*
  Modify this function to implement the forward pass described in Chapter 16.
  The code in 16 is for a single image.
  We have added an additional dimension to the tensors to support an entire mini-batch
  The goal here is to be correct, not fast (this is the CPU implementation.)

  Function paramters:
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

  // Insert your CPU convolution kernel code here
  // vector<float> vectorMask {mask, Channel * K * K * Map_out };
  // vector<float> vectorInput {input,  Height * Width * Channel * Batch};
  vector<float> vectorMask, vectorInput;
  for(int i = 0; i < Channel * K * K * Map_out; i++)
    vectorMask[i] = mask[i];
  for(int i = 0; i < Height * Width * Channel * Batch; i++)
    vectorInput[i] = input[i];

  vector<float> vectorOutput = fftConvolution(vectorInput, vectorMask);
  for(int i = 0; i < (int)vectorOutput.size(); i++)
    output[i] = vectorOutput[i];
  #undef out_4d
  #undef in_4d
  #undef mask_4d

}