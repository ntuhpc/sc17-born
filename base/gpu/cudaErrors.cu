/* include this file to catch cuda runtime and cuda device API errors */
/* By: Abdullah AlTheyab (2009) */
#include <stdio.h>
#include <cuda.h>
#include <builtin_types.h>
#define kernel_exec(x,y) x,y; cuda_kernel_error(__FILE__, __LINE__)
inline void cuda_kernel_error(char* file, int linenum){
	cudaError_t errcode=cudaGetLastError();
	if(errcode!=cudaSuccess){
		printf("Kernel error in file %s line %d: %s\n", file, linenum, cudaGetErrorString(errcode));
		exit(-1);
	}
}

#define cuda_call(x) cuda_call_check(__FILE__, __LINE__, x)
inline void cuda_call_check(char* file, int linenum, cudaError_t errcode){
	if(errcode!=cudaSuccess){
		printf("CUDA error in file %s line %d: %s\n", file, linenum, cudaGetErrorString(errcode));
		exit(-1);
	}
}

#define cu_call(x) cu_call_check(__FILE__, __LINE__, x)
inline void cu_call_check(char* file, int linenum, CUresult status){
	if(status!=CUDA_SUCCESS){
		char * msg;
		
		switch(status){
			case CUDA_ERROR_DEINITIALIZED:
				msg="CUDA_ERROR_DEINITIALIZED"; break;
			case CUDA_ERROR_NOT_INITIALIZED:
				msg="CUDA_ERROR_NOT_INITIALIZED"; break;
			case CUDA_ERROR_INVALID_CONTEXT:
				msg="CUDA_ERROR_INVALID_CONTEXT"; break;
			case CUDA_ERROR_INVALID_HANDLE:
				msg="CUDA_ERROR_INVALID_HANDLE"; break;
			case CUDA_ERROR_INVALID_DEVICE:
				msg="CUDA_ERROR_INVALID_DEVICE"; break;
			case CUDA_ERROR_INVALID_VALUE:
				msg="CUDA_ERROR_INVALID_VALUE"; break;
			case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
				msg="CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"; break;
			case CUDA_ERROR_OUT_OF_MEMORY:
				msg="CUDA_ERROR_OUT_OF_MEMORY"; break;
			case CUDA_ERROR_NOT_FOUND:
				msg="CUDA_ERROR_NOT_FOUND"; break;
			case CUDA_ERROR_FILE_NOT_FOUND:
				msg="CUDA_ERROR_FILE_NOT_FOUND"; break;
			case CUDA_ERROR_NO_BINARY_FOR_GPU:
				msg="CUDA_ERROR_NO_BINARY_FOR_GPU"; break;
			case CUDA_ERROR_NOT_READY:
				msg="CUDA_ERROR_NOT_READY"; break;
			case CUDA_ERROR_NO_DEVICE:
				msg="CUDA_ERROR_NO_DEVICE"; break;
			case CUDA_ERROR_ARRAY_IS_MAPPED:
				msg="CUDA_ERROR_ARRAY_IS_MAPPED"; break;
			case CUDA_ERROR_MAP_FAILED:
				msg="CUDA_ERROR_MAP_FAILED"; break;
			case CUDA_ERROR_ALREADY_MAPPED:
				msg="CUDA_ERROR_ALREADY_MAPPED"; break;
			case CUDA_ERROR_NOT_MAPPED:
				msg="CUDA_ERROR_NOT_MAPPED"; break;
			case CUDA_ERROR_UNKNOWN:
				msg="CUDA_ERROR_UNKNOWN"; break;
			default:
				msg="I don't know!!"; break;
		}
		fprintf(stderr, "*Cuda driver API error code %d in %s:%d: %s.\n", status, file, linenum, msg );
		exit(-1);
	}
}
