#ifndef _RATES_TEST_PROCEDURE_CUH_
#define _RATES_TEST_PROCEDURE_CUH_


typedef void (*Procedure)(dim3 gridDim, dim3 blockDim, float* const input, float* output, dim3 domain);

struct Configuration
{
	dim3 domain;
	int  scenario;
	bool verification;
};

int parseCommandline(int argc, char** argv, Configuration& args);
void runTest(Procedure proc, dim3 domain, bool verify, const char* procName);

#endif // _RATES_TEST_PROCEDURE_CUH_
