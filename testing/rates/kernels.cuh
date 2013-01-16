#ifndef _RATES_KERNEL_LIBRARY_CUH_
#define _RATES_KERNEL_LIBRARY_CUH_


void RatesVersion1(dim3 gridDim, dim3 blockDim, float* const input, float* output, dim3 domain);
void RatesVersion2(dim3 gridDim, dim3 blockDim, float* const input, float* output, dim3 domain);


template <int BLOCK_DIM>
__global__
void ratesKernelVersion3(float* const input, float* output, dim3 domain)
{
	int tid = blockIdx.x * BLOCK_DIM + threadIdx.x;
	if (tid < domain.x) output[tid] = input[tid];
}

template <int BLOCK_DIM>
void RatesVersion3(dim3 gridDim, dim3 blockDim, float* const input, float* output, dim3 domain)
{
	ratesKernelVersion3<BLOCK_DIM><<<gridDim, blockDim>>>(input, output, domain);
}

#endif // _RATES_KERNEL_LIBRARY_CUH_
