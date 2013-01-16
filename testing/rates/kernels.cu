#include "kernels.cuh"



extern "C" __global__
void ratesKernelVersion1(float* const input, float* output, dim3 domain)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < domain.x) output[tid] = input[tid];
}

extern "C" __global__
void ratesKernelVersion2(float* const input, float* output, dim3 domain)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < domain.x) output[tid] = input[tid];
}

void RatesVersion1(dim3 gridDim, dim3 blockDim, float* const input, float* output, dim3 domain)
{
	ratesKernelVersion1<<<gridDim, blockDim>>>(input, output, domain);
}
void RatesVersion2(dim3 gridDim, dim3 blockDim, float* const input, float* output, dim3 domain)
{
	ratesKernelVersion2<<<gridDim, blockDim>>>(input, output, domain);
}
