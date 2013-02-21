#include <cstdio>
#include <cstring>
#include <cassert>
#include "testproc.cuh"
#include "kernels.cuh"

#define CUDA_CHECK_RETURN(value)                                              \
{											                                  \
	cudaError_t _m_cudaStat_ = value;										  \
	if (_m_cudaStat_ != cudaSuccess) {										  \
		fprintf(stderr, "[%s] Error %s at line %d in file %s\n",		      \
				#value, cudaGetErrorString(_m_cudaStat_), __LINE__, __FILE__);\
		exit(1);															  \
	} }


static void produceDataset(float* const a, dim3 domain)
{
	// todo
	srand(time(NULL));
	for (int i = 0; i < domain.x; ++i)
	{
		a[i] = (rand() % 20) - 5;
	}
}

static void verifyCorrectness(float* const a, float* const b, dim3 domain)
{
	printf("\nVerify correctness: ");
	bool correct = false;

	// todo
	// allocate result array
	// compute reference solution, store in result array
	// compare res and b
	// free result array

	if (correct)
		printf(" Correct\n");
	else
		printf(" Incorrect!\n");
}

void runTest(Procedure procedure, dim3 domain, bool verify, const char* procName)
{
	printf("Run test case %s...\n", procName);

	size_t sizeA = sizeof(float) * domain.x * domain.y * domain.z;
	size_t sizeB = sizeof(float) * domain.x * domain.y * domain.z * 4;

	float* hostA = (float*)malloc(sizeA); assert(hostA != 0);
	float* hostB = (float*)malloc(sizeB); assert(hostB != 0);

	produceDataset(hostA, domain);

	float* a = 0;
	float* b = 0;

	CUDA_CHECK_RETURN(cudaMalloc((void**) &a, sizeA));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &b, sizeB));
	CUDA_CHECK_RETURN(cudaMemcpy(a, hostA, sizeA, cudaMemcpyHostToDevice));

	// todo
	dim3 blockDim(1, 1, 1);
	dim3 gridDim(1, 1, 1);

	printf("Grid  (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
	printf("Block (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);

	// warm up run
	procedure(gridDim, blockDim, a, b, domain);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	cudaEvent_t start, stop;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	const int numIterations = 100;

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	for (int i = 0; i < numIterations; ++i)
	{
		procedure(gridDim, blockDim, a, b, domain);
	}
    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

	float totalTimeMsec = 0.0f;
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&totalTimeMsec, start, stop));

	size_t loadedBytes = sizeA;
	size_t storedBytes = sizeB;
	float  meanTimeMsec = totalTimeMsec/numIterations;
    double flopsPerKenrnel = 0.0;
    double gigaFlopsPerKernel= (flopsPerKenrnel * 1.0e-9f) / (meanTimeMsec / 1000.0f);

	printf("Mean computation time:  %f [ms]\n",         meanTimeMsec);
	printf("Data transferred:       %.3f [KB]\n",       (loadedBytes + storedBytes) * 1.0e-3);
	printf("Effective bandwidth:    %.3f [GB/s]\n",     (loadedBytes) / meanTimeMsec / 1.0e6);
	printf("Performance:            %.3f [GFlop/s]\n",  gigaFlopsPerKernel);

	CUDA_CHECK_RETURN(cudaMemcpy(hostB, b, sizeB, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));
	CUDA_CHECK_RETURN(cudaFree((void*) a));
	CUDA_CHECK_RETURN(cudaFree((void*) b));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	if (verify)
	{
		verifyCorrectness(hostA, hostB, domain);
	}
    printf("\n");

	free(hostA);
	free(hostB);
}


static void programUsage(int argc, char** argv)
{
	printf("Usage: %s [-s <scenario>] [-x <dimx>] [-y <dimy>] [-z <dimz>] [-v <verificationFlag>]\n", argv[0]);
}

int parseCommandline(int argc, char** argv, Configuration& config)
{
	if (argc < 2)
	{
		programUsage(argc, argv);
		return 1;
	}

	int  s = 0;
	int  x = 0;
	int  y = 0;
	int  z = 0;
	bool v = false;

	int c = 1;
	while (c < argc)
	{
		if (strcmp(argv[c] , "-s") == 0)
		{
			s = (++c < argc) ? atoi(argv[c]) : 0;
		}
		else if (strcmp(argv[c] , "-x") == 0)
		{
			x = (++c < argc) ? atoi(argv[c]) : 0;
		}
		else if (strcmp(argv[c] , "-y") == 0)
		{
			y = (++c < argc) ? atoi(argv[c]) : 0;
		}
		else if (strcmp(argv[c] , "-z") == 0)
		{
			z = (++c < argc) ? atoi(argv[c]) : 0;
		}
		else if (strcmp(argv[c] , "-v") == 0)
		{
			v = (++c < argc) ? (atoi(argv[c]) > 0 ? true : false) : false;
		}
		else
		{
			programUsage(argc, argv);
			return 1;
		}
		c++;
	}

	printf("Run test with arguments: %d (%d, %d, %d) %s", s, x, y, z, v == true ? "True" : "False");

	if (x == 0)
	{
		printf(" - x-dimension doesn't make sense\n");
		return 1;
	}
	printf("\n");

	config.domain.x = x;
	config.domain.y = y;
	config.domain.z = z;
	config.scenario = s;
	config.verification = v;

	return 0;
}
