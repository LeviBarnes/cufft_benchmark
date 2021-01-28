
FFT_test: FFT_test.cu
	nvcc -o FFT_test FFT_test.cu -L/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/math_libs/11.0/lib64/ -lcufft

clean:
	rm FFT_test

