#include <iostream>
#include "cufft.h"
#include "cufftXt.h"



int main(int argc, char** argv) {

  int sz = 1024;
  if (argc > 1) sz = atoi(argv[1]);
  int nx=sz, ny=sz;
// Demonstrate how to use CUFFT to perform 3-d FFTs using 2 GPUs
//
// cufftCreate() - Create an empty plan
    cufftHandle plan_input; cufftResult result;
    result = cufftCreate(&plan_input);
    if (result != CUFFT_SUCCESS) { printf ("*Create failed\n"); return 0; }
//
// cufftXtSetGPUs() - Define which GPUs to use
    int nGPUs = 2, whichGPUs[2];
    whichGPUs[0] = 0; whichGPUs[1] = 1;
    result = cufftXtSetGPUs (plan_input, nGPUs, whichGPUs);
    if (result != CUFFT_SUCCESS) { printf ("*XtSetGPUs failed\n"); return 0; }
//
// Initialize FFT input data
    size_t worksize[2];
    cufftComplex *host_data_input, *host_data_output;
    int size_of_data = sizeof(cufftComplex) * nx * ny;
    host_data_input = (cufftComplex *)malloc(size_of_data);
    if (host_data_input == NULL) { printf ("malloc failed\n"); return 0; }
    host_data_output = (cufftComplex *)malloc(size_of_data);
    if (host_data_output == NULL) { printf ("malloc failed\n"); return 0; }
    //initialize_3d_data (nx, ny, host_data_input, host_data_output);
//
// cufftMakePlan3d() - Create the plan
    result = cufftMakePlan2d (plan_input, ny, nx, CUFFT_C2C, worksize);
    if (result != CUFFT_SUCCESS) { printf ("*MakePlan* failed\n"); return 0; }
//
// cufftXtMalloc() - Malloc data on multiple GPUs
    cudaLibXtDesc *device_data_input;
    result = cufftXtMalloc (plan_input, &device_data_input,
        CUFFT_XT_FORMAT_INPLACE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMalloc failed\n"); return 0; }
//
// cufftXtMemcpy() - Copy data from host to multiple GPUs
    result = cufftXtMemcpy (plan_input, device_data_input,
        host_data_input, CUFFT_COPY_HOST_TO_DEVICE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); return 0; }
//
// cufftXtExecDescriptorC2C() - Execute FFT on multiple GPUs
    result = cufftXtExecDescriptorC2C (plan_input, device_data_input,
        device_data_input, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS) { printf ("*XtExec* failed\n"); return 0; }
//
// cufftXtMemcpy() - Copy data from multiple GPUs to host
    result = cufftXtMemcpy (plan_input, host_data_output,
        device_data_input, CUFFT_COPY_DEVICE_TO_HOST);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); return 0; }
//
// Print output and check results
    //int output_return = output_2d_results (nx, ny,
   //     host_data_input, host_data_output);
    //if (output_return != 0) { return 0; }
//
// cufftXtFree() - Free GPU memory
    result = cufftXtFree(device_data_input);
    if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); return 0; }
//
// cufftDestroy() - Destroy FFT plan
    result = cufftDestroy(plan_input);
    if (result != CUFFT_SUCCESS) { printf ("*Destroy failed: code\n"); return 0; }
    free(host_data_input); free(host_data_output);
    return 0;
}
