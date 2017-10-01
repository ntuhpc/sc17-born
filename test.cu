#include <cuda_runtime.h>
#include <cstdio>
#include "base/gpu/wave_fkernel.3d8o.cu"

#define SIZE 24
#define RADIUS 4

float coeffs_cpu[13] = {1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};

void prop(float *p0, float *p1, float *vel) {
	int _nx = SIZE;
	int _n12 = SIZE*SIZE;
        for(int i3=4; i3 < SIZE-4; i3++){
          for (int i2 = 4; i2 < SIZE-4; i2++) {
            int ii = i2 * SIZE + 4 + SIZE*SIZE * i3;
            for (int i1 = 4; i1 < SIZE-4; i1++, ii++) {
              float x = p0[ii] =
                  vel[ii] *
                      ( coeffs_cpu[C0] * p1[ii] +
                        coeffs_cpu[CX1] * (p1[ii - 1] + p1[ii + 1]) +
                       +coeffs_cpu[CX2] * (p1[ii - 2] + p1[ii + 2]) +
                       +coeffs_cpu[CX3] * (p1[ii - 3] + p1[ii + 3]) +
                       +coeffs_cpu[CX4] * (p1[ii - 4] + p1[ii + 4]) +
                       +coeffs_cpu[CY1] * (p1[ii - _nx] + p1[ii + _nx]) +
                       +coeffs_cpu[CY2] * (p1[ii - 2 * _nx] + p1[ii + 2 * _nx]) +
                       +coeffs_cpu[CY3] * (p1[ii - 3 * _nx] + p1[ii + 3 * _nx]) +
                       +coeffs_cpu[CY4] * (p1[ii - 4 * _nx] + p1[ii + 4 * _nx]) +
                       +coeffs_cpu[CZ1] * (p1[ii - 1 * _n12] + p1[ii + 1 * _n12]) +
                       +coeffs_cpu[CZ2] * (p1[ii - 2 * _n12] + p1[ii + 2 * _n12]) +
                       +coeffs_cpu[CZ3] * (p1[ii - 3 * _n12] + p1[ii + 3 * _n12]) +
                       +coeffs_cpu[CZ4] * (p1[ii - 4 * _n12] + p1[ii + 4 * _n12])) +
                  p1[ii] + p1[ii] - p0[ii];
            }
          }
        }
}

int main()
{
	// init array
	float* ref_array_cpu = (float*)malloc(SIZE*SIZE*SIZE*sizeof(float));
	float* test_array_gpu, *gpu_p0;
	float* cpu_result = (float*)malloc(SIZE*SIZE*SIZE*sizeof(float));
	float* gpu_result = (float*)malloc(SIZE*SIZE*SIZE*sizeof(float));
	float* vel_cpu = (float*)malloc(SIZE*SIZE*SIZE*sizeof(float));
	float* vel_gpu;
	cudaMalloc(&test_array_gpu, SIZE*SIZE*SIZE*sizeof(float));
	cudaMalloc(&gpu_p0, SIZE*SIZE*SIZE*sizeof(float));
	cudaMalloc(&vel_gpu, SIZE*SIZE*SIZE*sizeof(float));
	for (int i = 0; i < SIZE*SIZE*SIZE; ++i)
	{
		ref_array_cpu[i] = i;
		cpu_result[i] = 100;
		vel_cpu[i] = i;
	}
	cudaMemcpy(test_array_gpu, ref_array_cpu, SIZE*SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(vel_gpu, vel_cpu, SIZE*SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_p0, cpu_result, SIZE*SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(coeffs, coeffs_cpu, sizeof(float)*13, 0);

	// run GPU kernel
	dim3 block(16, 16);
	dim3 grid(1, 1);
	int offset = SIZE * SIZE * RADIUS + SIZE * RADIUS + RADIUS;
	wave_kernel<<<grid, block>>>(gpu_p0+offset, test_array_gpu+offset, gpu_p0+offset, vel_gpu+offset, 0, 16);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("%s\n", cudaGetErrorString(err));
	cudaMemcpy(gpu_result, gpu_p0, SIZE*SIZE*SIZE*sizeof(float), cudaMemcpyDeviceToHost);

	// run CPU impl
	prop(cpu_result, ref_array_cpu, vel_cpu);

	// compare
	printf("cpu at %d: %f\n", offset, cpu_result[offset]);
	printf("gpu at %d: %f\n", offset, gpu_result[offset]);
}
