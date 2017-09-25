#ifndef GPU_FINITE_3D_H
#define GPU_FINITE_3D_H 1
//#include <cutil.h>
//#include <cuda.h>
#include "gpu_funcs_3d.h"
//#define DEBUG
#if defined (DEBUG)
	#define _DBG printf("DBG: %d \n", __LINE__);
#else
	#define _DBG 
#endif

#define hloc(z,x,y) (z)+(x)*nz+(y)*nz*nx   /* host mem. z,x,y-coordinates to linear address converter */ //ok. add y to these location addresses
#define dloc(z,x,y) (z)+(x)*n1gpu+(y)*n1gpu*n2gpu  /* device mem. z,x,y-coordinate to linear address converter */
#define min2(v1,v2) (((v1)<(v2))?(v1):(v2)) /* minimum function */
#define dloc2(x,y) (x)+(y)*n2gpu


#define IMG_KERNEL 1

#define IK1_BLKSIZE 16 /* imaging kernel 1 & 2 variable */ //leave these for now....

#define IK4Z_ZBLKSIZE 128 
#define IK4Z_XBLKSIZE 1   /* sparce data */

#define IK4X_XBLKSIZE 16
#define IK4X_ZBLKSIZE 32

#define MAX_NUM_GPUS 8

int n_gpus,device[MAX_NUM_GPUS],lead_pad,offset,shot_gpu;
float *velocity[MAX_NUM_GPUS],*velocity2[MAX_NUM_GPUS];
//float *velocity,*velocity2;
double dd1,dd2,dd3;

__constant__ int n1gpu;
__constant__ int n2gpu;
__constant__ int n3gpu;

__constant__ float bc_agpu;
__constant__ float bc_bgpu;
__constant__ float bc_b_ygpu;

__constant__ int nsrc_gpu;
__constant__ int nt_gpu;
__constant__ int ntblock;

__constant__ int *datageom_gpu0;
__constant__ float *data_gpu0;
__constant__ int *srcgeom_gpu0;
__constant__ float *source_gpu0; 
__constant__ int *src_loc,*rec_loc;
__constant__ float *sinc_s_table;
__constant__ float *sinc_d_table;

__constant__ int nsinc_gpu;
__constant__ int nextract_gpu;


__constant__ int dir_gpu;
__constant__ int *extract_geom0;
__constant__ int npts_gpu;

__constant__ int rec_nx_gpu;
__constant__ int rec_ny_gpu;
__constant__ int rec_ny_total_gpu;
__constant__ int ntrace_d_gpu;

__constant__ int ntrace_gpu,ntblock_gpu;
__constant__ int hmax_gpu;
__constant__ int nh_gpu;
int *datageom_gpu;
int *srcgeom_gpu;
float *source_gpu;
float *data_gpu;
float *extract_gpu;
float *sincstable;
float *sincdtable;
void writeWavefield(char *tag, float **dat, int n3s,int ngpu, int n1, int n2, int n3,int edge);
int *extract_geom;
 cudaArray *vel_array1 ;
 cudaArray *vel_array2 ;

float *scale_s_gpu,*scale_g_gpu;
__device__ int min3(int v1,int v2,int v3){return min2(min2(v1,v2),v3);}
__device__ int min4(int v1,int v2,int v3,int v4){return min2(min2(v1,v2),min2(v3,v4));}
__device__ int min5(int v1,int v2,int v3, int v4, int v5){return min2(min2(min2(v1,v2),min2(v3,v4)),v5);}
texture <float, 3, cudaReadModeElementType> vel_ref1;
texture <float, 3, cudaReadModeElementType> vel_ref2;


#endif
