#include <iostream>
#include "cufft.h"
#include "cufftXt.h"
#include <unistd.h>



int main(int argc, char** argv) {

  int sz = 1024;
  int nGPUs = 2, whichGPUs[16];
  whichGPUs[0] = 0; whichGPUs[1] = 1;
  if (argc > 1) {
    if ( ( strncmp(argv[1], "--help", 6) == 0) ||
         ( strncmp(argv[1], "-h", 2) == 0) ) {
      std::cout << "Usage:\n     FFT_test <field dimension> <num GPUs>" <<std::endl;
      return 0;
    } 
    sz = atoi(argv[1]);
  }
  int nx=sz, ny=sz;

  if (argc > 2) {
    nGPUs = atoi(argv[2]);
    if (nGPUs > 16) {
       std::cout << "More than 16 GPUs not supported" << std::endl;
       exit(1);
    }
    for (size_t qq=0;qq<nGPUs;qq++) whichGPUs[qq]=qq;
  }
  std::cout << "Transform " << ny << "x" << nx << " image "
            << "with " << nGPUs << " gpus." << std::endl;

// Timers
    cudaSetDevice(0);
    cudaEvent_t start, stop;
    cudaEvent_t mem_start, mem_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&mem_start);
    cudaEventCreate(&mem_stop);

// Demonstrate how to use CUFFT to perform 3-d FFTs using 2 GPUs
//
// cufftCreate() - Create an empty plan
    cufftHandle plan_input; cufftResult result; 
    result = cufftCreate(&plan_input);
    if (result != CUFFT_SUCCESS) { printf ("*Create failed\n"); return 0; }
//
// cufftXtSetGPUs() - Define which GPUs to use
    result = cufftXtSetGPUs (plan_input, nGPUs, whichGPUs);
    if (result != CUFFT_SUCCESS) { printf ("*XtSetGPUs failed\n"); return 0; }
//
// Initialize FFT input data
    size_t worksize[16];
    cufftComplex *host_data_input, *host_data_output;
    unsigned long size_of_data = sizeof(cufftComplex) * nx * ny;
    host_data_input = (cufftComplex *)malloc(size_of_data);
    if (host_data_input == NULL) { printf ("malloc (%ul) failed\n", size_of_data); return 0; }
    host_data_output = (cufftComplex *)malloc(size_of_data);
    if (host_data_output == NULL) { printf ("malloc (%ul, output) failed\n", size_of_data); return 0; }
    //initialize_3d_data (nx, ny, host_data_input, host_data_output);
//
// cufftMakePlan2d() - Create the plan
    result = cufftMakePlan2d (plan_input, ny, nx, CUFFT_C2C, worksize);
    if (result != CUFFT_SUCCESS) { printf ("*MakePlan* failed\n"); return 0; }
//
// cufftXtMalloc() - Malloc data on multiple GPUs
    cudaLibXtDesc *device_data_input;
    result = cufftXtMalloc (plan_input, &device_data_input,
        CUFFT_XT_FORMAT_INPLACE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMalloc failed\n"); return 0; }
    for(size_t dev=0;dev<nGPUs;dev++) { cudaSetDevice(whichGPUs[dev]); cudaDeviceSynchronize(); }
//
// cufftXtMemcpy() - Copy data from host to multiple GPUs
    cudaSetDevice(0); cudaEventRecord(mem_start);
    result = cufftXtMemcpy (plan_input, device_data_input,
        host_data_input, CUFFT_COPY_HOST_TO_DEVICE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); return 0; }
    for(size_t dev=0;dev<nGPUs;dev++) { cudaSetDevice(whichGPUs[dev]); cudaDeviceSynchronize(); }
//
// cufftXtExecDescriptorC2C() - Execute FFT on multiple GPUs
    cudaSetDevice(0); cudaEventRecord(start);
    result = cufftXtExecDescriptorC2C (plan_input, device_data_input,
        device_data_input, CUFFT_FORWARD);
    for(size_t dev=0;dev<nGPUs;dev++) { cudaSetDevice(whichGPUs[dev]); cudaDeviceSynchronize(); }
    cudaSetDevice(0); cudaEventRecord(stop);
    if (result != CUFFT_SUCCESS) { printf ("*XtExec* failed\n"); return 0; }
//
// cufftXtMemcpy() - Copy data from multiple GPUs to host
    result = cufftXtMemcpy (plan_input, host_data_output,
        device_data_input, CUFFT_COPY_DEVICE_TO_HOST);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); return 0; }
    cudaSetDevice(0); cudaEventRecord(mem_stop);
//
// Print output and check results
    //int output_return = output_2d_results (nx, ny,
   //     host_data_input, host_data_output);
    //if (output_return != 0) { return 0; }
//
// cufftXtFree() - Free GPU memory
    result = cufftXtFree(device_data_input);
    if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); return 0; }
    if (cudaGetLastError()) std::cout << "Uncaught CUDA error. Line " << __LINE__ << std::endl;
//

// report timing

    cudaSetDevice(0); 
    cudaEventSynchronize(stop);
    cudaEventSynchronize(mem_stop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << "cufft exec time: " << elapsed << " ms." << std::endl;
    cudaEventElapsedTime(&elapsed, mem_start, mem_stop);
    std::cout << "including memcpy: " << elapsed << " ms." << std::endl;
// cufftDestroy() - Destroy FFT plan
    result = cufftDestroy(plan_input);
    if (result != CUFFT_SUCCESS) { printf ("*Destroy failed: code\n"); return 0; }
    free(host_data_input); free(host_data_output);

// destroy timers
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(mem_start);
    cudaEventDestroy(mem_stop);

    return 0;
}
