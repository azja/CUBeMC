#include <cstdlib>
#include <cstdio>
#include "testproc.cuh"
#include "kernels.cuh"

#define NAME(proc) #proc


static int runScenario(int scenario, dim3 domain, bool verify)
{
    printf("----------------------------------------------\n");
    printf(" Test scenario %d\n\n", scenario);

    switch (scenario)
    {
    case 0:
    {
    	runTest(RatesVersion1,      domain, verify, NAME(RatesVersion1));
    	runTest(RatesVersion2,      domain, verify, NAME(RatesVersion2));
    	runTest(RatesVersion3<256>, domain, verify, NAME(RatesVersion3<256>));
    }
    break;
    case 1: { runTest(RatesVersion1,      domain, verify, NAME(RatesVersion1)); } break;
    case 2: { runTest(RatesVersion2,      domain, verify, NAME(RatesVersion2)); } break;
    case 3: { runTest(RatesVersion3<256>, domain, verify, NAME(RatesVersion3<256>)); } break;
    default:
    {
    	printf(" Unknown test scenario! Exiting...\n");
    	return EXIT_FAILURE;
    }
    }
    return EXIT_SUCCESS;
}


int main(int argc, char** argv)
{
	Configuration config = {0};
	if (parseCommandline(argc, argv, config) > 0)
	{
		return EXIT_FAILURE;
	}

	int devID = 0;

    cudaError_t error;
    cudaDeviceProp deviceProp;

    error = cudaGetDevice(&devID);
    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);
    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("\nGPUDevice %d:  %s\nCompute cap:  %d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

	return runScenario(config.scenario, config.domain, config.verification);
}
