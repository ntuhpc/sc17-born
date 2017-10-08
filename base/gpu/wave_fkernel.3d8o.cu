/* 3D Time domain finite difference kernel
        By: Chris Leader, Abdullah AlTheyab
        Ref: similar to Paulius code but for 2D, then re-extended to 3D
 */
#include <stdio.h>
#include "gpu_finite_3d.h"
#define BLOCKZ_SIZE 16
#define BLOCKX_SIZE 16
#define BLOCKY_SIZE 16
#define FAT 4
#define COEFFS_SIZE 13
#define radius 4

#define C0 0
#define CZ1 1
#define CX1 2
#define CY1 3
#define CZ2 4
#define CX2 5
#define CY2 6
#define CZ3 7
#define CX3 8
#define CY3 9
#define CZ4 10
#define CX4 11
#define CY4 12

#define C_C00(d) (8.0 / (5.0 * (d) * (d)))
#define get_coeffs(d1, d2, d3)                                        \
  {                                                                   \
    -1025.0 / 576.0 * (C_C00(d1) + C_C00(d2) + C_C00(d3)), C_C00(d1), \
        C_C00(d2), C_C00(d3), -C_C00(d1) / 8.0, -C_C00(d2) / 8.0,     \
        -C_C00(d3) / 8.0, C_C00(d1) / 63.0, C_C00(d2) / 63.0,         \
        C_C00(d3) / 63.0, -C_C00(d1) / 896.0, -C_C00(d2) / 896.0,     \
        -C_C00(d3) / 896.0                                            \
  }

__constant__ float coeffs[COEFFS_SIZE];

extern "C" __global__ void wave_kernel(float *p0, float *p1, float *p2,
                                       float *vel, const int start3,
                                       const int end3, const int last_x_block, const int last_y_block) {
  __shared__ float p1s[BLOCKY_SIZE + 2 * FAT][BLOCKX_SIZE + 2 * FAT];

  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ix >= (n1gpu - 2 * radius) || iy >= (n2gpu - 2 * radius)) return; // boundary condition check

  int in_idx = iy * n1gpu + ix - radius * n1gpu * n2gpu;
  //int in_idx = iy * 24 + ix - radius * 24 * 24;
  int out_idx = 0;
  int stride = n1gpu * n2gpu;
  //int stride = 24 * 24;

  float infront1, infront2, infront3, infront4;
  float behind1, behind2, behind3, behind4;
  float current;

  int tx = threadIdx.x + radius;
  int ty = threadIdx.y + radius;

  // Assign slow axis values
  behind3 = p1[in_idx]; in_idx += stride;
  behind2 = p1[in_idx]; in_idx += stride;
  behind1 = p1[in_idx]; in_idx += stride;
  current = p1[in_idx]; out_idx = in_idx; in_idx += stride;
  infront1 = p1[in_idx]; in_idx += stride;
  infront2 = p1[in_idx]; in_idx += stride;
  infront3 = p1[in_idx]; in_idx += stride;
  infront4 = p1[in_idx]; in_idx += stride;

  for (int i = start3; i < end3; ++i) {
    // advance the slice
    behind4 = behind3;
    behind3 = behind2;
    behind2 = behind1;
    behind1 = current;
    current = infront1;
    infront1 = infront2;
    infront2 = infront3;
    infront3 = infront4;
    infront4 = p1[in_idx];

    in_idx += stride;
    out_idx += stride;
    __syncthreads();

    // also load bottom & top halo
    if (blockIdx.y == last_y_block && blockDim.y != BLOCKY_SIZE) {
    	if (threadIdx.y == 0) {
    	  int offset = n2gpu - 2 * radius - iy;

    	  p1s[0][tx] = p1[out_idx - radius * n1gpu];
    	  p1s[ty + offset][tx] = p1[out_idx + offset * n1gpu];

    	  p1s[1][tx] = p1[out_idx - 3 * n1gpu];
    	  p1s[ty + offset+1][tx] = p1[out_idx + (offset + 1) * n1gpu];

    	  p1s[2][tx] = p1[out_idx - 2 * n1gpu];
    	  p1s[ty + offset+2][tx] = p1[out_idx + (offset + 2) * n1gpu];

    	  p1s[3][tx] = p1[out_idx - 1 * n1gpu];
    	  p1s[ty + offset+3][tx] = p1[out_idx + (offset + 3) * n1gpu];
    	}
    } else {
    	if (threadIdx.y < radius) {
    	  p1s[threadIdx.y][tx] = p1[out_idx - radius * n1gpu];
    	  //p1s[threadIdx.y][tx] = p1[out_idx - radius * 24];
    	  p1s[ty + BLOCKY_SIZE][tx] = p1[out_idx + BLOCKY_SIZE * n1gpu];
    	  //p1s[ty + BLOCKY_SIZE][tx] = p1[out_idx + BLOCKY_SIZE * 24];
    	}
    }
    // also load left & right halo
    if (blockIdx.x == last_x_block && blockDim.x != BLOCKX_SIZE) {
    	if (threadIdx.x == 0) {
    	  int offset = n1gpu - 2 * radius - ix;

          p1s[ty][0] = p1[out_idx - radius];
          p1s[ty][tx + offset] = p1[out_idx + offset];

          p1s[ty][1] = p1[out_idx - 3];
          p1s[ty][tx + offset + 1] = p1[out_idx + offset + 1];

          p1s[ty][2] = p1[out_idx - 2];
          p1s[ty][tx + offset + 2] = p1[out_idx + offset + 2];

          p1s[ty][3] = p1[out_idx - 1];
          p1s[ty][tx + offset + 3] = p1[out_idx + offset + 3];
    	}
    } else {
      if (threadIdx.x < radius) {
        p1s[ty][threadIdx.x] = p1[out_idx - radius];
        p1s[ty][tx + BLOCKX_SIZE] = p1[out_idx + BLOCKX_SIZE];
      }
    }
    p1s[ty][tx] = current;
    __syncthreads();

//if (ix == 0 && iy == 0 && i == start3)
//{
//	printf("current: %f\n", p1s[tx][ty]);
//	printf("x axis: %f %f %f %f\n", p1s[ty][tx-1], p1s[ty][tx-2], p1s[ty][tx-3], p1s[ty][tx-4]);
//	printf("x axis: %f %f %f %f\n", p1s[ty][tx+1], p1s[ty][tx+2], p1s[ty][tx+3], p1s[ty][tx+4]);
//	printf("y axis negative: %f %f %f %f\n", p1s[ty - 1][tx], p1s[ty - 2][tx], p1s[ty - 3][tx], p1s[ty - 4][tx]);
//	printf("y pos: %f %f %f %f\n", p1s[ty + 1][tx], p1s[ty + 2][tx], p1s[ty + 3][tx], p1s[ty + 4][tx]);
//	printf("z axis: %f %f %f %f %f %f %f %f\n", behind4, behind3, behind2, behind1, infront1, infront2, infront3, infront4);
//	printf("p0: %f\n", p0[out_idx]);
//	printf("vel: %f\n", vel[out_idx]);
//}

    float div = coeffs[C0] * current +
                            coeffs[CX1] * p1s[ty][tx - 1] + coeffs[CX1] * p1s[ty][tx + 1] +
                            coeffs[CX2] * p1s[ty][tx - 2] + coeffs[CX2] * p1s[ty][tx + 2] +
                            coeffs[CX3] * p1s[ty][tx - 3] + coeffs[CX3] * p1s[ty][tx + 3] +
                            coeffs[CX4] * p1s[ty][tx - 4] + coeffs[CX4] * p1s[ty][tx + 4] +
                            coeffs[CY1] * p1s[ty - 1][tx] + coeffs[CY1] * p1s[ty + 1][tx] +
                            coeffs[CY2] * p1s[ty - 2][tx] + coeffs[CY2] * p1s[ty + 2][tx] +
                            coeffs[CY3] * p1s[ty - 3][tx] + coeffs[CY3] * p1s[ty + 3][tx] +
                            coeffs[CY4] * p1s[ty - 4][tx] + coeffs[CY4] * p1s[ty + 4][tx] +
                            coeffs[CZ1] * infront1 + coeffs[CZ1] * behind1 +
                            coeffs[CZ2] * infront2 + coeffs[CZ2] * behind2 +
                            coeffs[CZ3] * infront3 + coeffs[CZ3] * behind3 +
                            coeffs[CZ4] * infront4 + coeffs[CZ4] * behind4;
    p0[out_idx] = current + current - p0[out_idx] + div * vel[out_idx];
  }
}

extern "C" __global__ void wave_kernel_adj(float *p0, float *p1, float *p2,
                                           float *vel, const int start3,
                                           const int end3) {
  __shared__ float p1s[BLOCKX_SIZE + 2 * FAT][BLOCKY_SIZE + 2 * FAT];
  __shared__ float vls[BLOCKX_SIZE + 2 * FAT][BLOCKY_SIZE + 2 * FAT];

  int ig = blockIdx.x * blockDim.x +
           threadIdx.x;  // Global coordinates for the fastest two axes
  int jg = blockIdx.y * blockDim.y + threadIdx.y;

  int il = threadIdx.x + FAT;  // Local coordinates for the fastest two axes
  int jl = threadIdx.y + FAT;

  float p1y[2 * radius + 1];  // Array of elements to hold slow axis values
  float vly[2 * radius + 1];  // Array to hold out of plane velocity values

  int stride = n1gpu * n2gpu;  // Number of elements between wavefield slices
  int addr = ig + n1gpu * jg;  // Index of the central slow-axis element
  int addr_fwd =
      addr - radius * stride;  // Index of the first slow-axis element

  // Assign slow axis values
  p1y[1] = p1[addr_fwd];
  vly[1] = vel[addr_fwd];
  p1y[2] = p1[addr_fwd += stride];
  vly[2] = vel[addr_fwd];
  p1y[3] = p1[addr_fwd += stride];
  vly[3] = vel[addr_fwd];
  p1s[jl][il] = p1[addr_fwd += stride];
  vls[jl][il] = vel[addr_fwd];
  p1y[5] = p1[addr_fwd += stride];
  vly[5] = vel[addr_fwd];
  p1y[6] = p1[addr_fwd += stride];
  vly[6] = vel[addr_fwd];
  p1y[7] = p1[addr_fwd += stride];
  vly[7] = vel[addr_fwd];
  p1y[8] = p1[addr_fwd += stride];
  vly[8] = vel[addr_fwd];

  //#pragma unroll 9
  for (int yl = start3; yl < end3; yl++) {
    // Update slow axis values
    p1y[0] = p1y[1];
    vly[0] = vly[1];
    p1y[1] = p1y[2];
    vly[1] = vly[2];
    p1y[2] = p1y[3];
    vly[2] = vly[3];
    p1y[3] = p1s[jl][il];
    vly[3] = vls[jl][il];
    p1s[jl][il] = p1y[5];
    vls[jl][il] = vly[5];
    p1y[5] = p1y[6];
    vly[5] = vly[6];
    p1y[6] = p1y[7];
    vly[6] = vly[7];
    p1y[7] = p1y[8];
    vly[7] = vly[8];
    p1y[8] = p1[addr_fwd += stride];
    vly[8] = vel[addr_fwd];

    if (threadIdx.x < FAT) {
      p1s[jl][threadIdx.x] = p1[addr - FAT];
      p1s[jl][il + BLOCKZ_SIZE] = p1[addr + BLOCKZ_SIZE];
      vls[jl][threadIdx.x] = vel[addr - FAT];
      vls[jl][il + BLOCKZ_SIZE] = vel[addr + BLOCKZ_SIZE];
    }
    if (threadIdx.y < FAT) {
      p1s[threadIdx.y][il] = p1[addr - FAT * n1gpu];
      p1s[jl + BLOCKX_SIZE][il] = p1[addr + BLOCKX_SIZE * n1gpu];
      vls[threadIdx.y][il] = vel[addr - FAT * n1gpu];
      vls[jl + BLOCKX_SIZE][il] = vel[addr + BLOCKX_SIZE * n1gpu];
    }
    __syncthreads();

    p2[addr] = coeffs[C0] * p1s[jl][il] * vls[jl][il] +
               coeffs[CZ1] * (p1s[jl][il + 1] * vls[jl][il + 1] +
                              p1s[jl][il - 1] * vls[jl][il - 1]) +
               coeffs[CX1] * (p1s[jl + 1][il] * vls[jl + 1][il] +
                              p1s[jl - 1][il] * vls[jl - 1][il]) +
               coeffs[CY1] * (p1y[radius + 1] * vly[radius + 1] +
                              p1y[radius - 1] * vly[radius - 1]) +
               coeffs[CZ2] * (p1s[jl][il + 2] * vls[jl][il + 2] +
                              p1s[jl][il - 2] * vls[jl][il - 2]) +
               coeffs[CX2] * (p1s[jl + 2][il] * vls[jl + 2][il] +
                              p1s[jl - 2][il] * vls[jl - 2][il]) +
               coeffs[CY2] * (p1y[radius + 2] * vly[radius + 2] +
                              p1y[radius - 2] * vly[radius - 2]) +
               coeffs[CZ3] * (p1s[jl][il + 3] * vls[jl][il + 3] +
                              p1s[jl][il - 3] * vls[jl][il - 3]) +
               coeffs[CX3] * (p1s[jl + 3][il] * vls[jl + 3][il] +
                              p1s[jl - 3][il] * vls[jl - 3][il]) +
               coeffs[CY3] * (p1y[radius + 3] * vly[radius + 3] +
                              p1y[radius - 3] * vly[radius - 3]) +
               coeffs[CZ4] * (p1s[jl][il + 4] * vls[jl][il + 4] +
                              p1s[jl][il - 4] * vls[jl][il - 4]) +
               coeffs[CX4] * (p1s[jl + 4][il] * vls[jl + 4][il] +
                              p1s[jl - 4][il] * vls[jl - 4][il]) +
               coeffs[CY4] * (p1y[radius + 4] * vly[radius + 4] +
                              p1y[radius - 4] * vly[radius - 4]) +
               p1s[jl][il] + p1s[jl][il] - p0[addr];
    addr += stride;
  }
}

extern "C" __global__ void damp_kernel(float *p0, float *p1, const int start3,
                                       const int end3, const int gpu_id,
                                       const int n_gpus) {
  int ig = blockIdx.x * blockDim.x +
           threadIdx.x;  // Global coordinates for the fastest two axes
  int jg = blockIdx.y * blockDim.y + threadIdx.y;

  if (ig >= n1gpu - 2 * radius || jg >= n2gpu - 2 * radius) return;

  int stride = n1gpu * n2gpu;  // Number of elements between wavefield slices
  int addr = ig + n1gpu * jg;
  int edge = 0;
  // TODO: check
  int bc_agpu=40;
  float bc_bgpu=0.0005;

  // damp halo + internal region
  for (int zg = start3; zg < end3; zg++) {
	// damp all
    if (n_gpus == 1)
      edge = min2(
          ig,
          min2(zg - start3, min2(min2(n2gpu - 2 * radius - jg, n1gpu - 2 * radius - ig),
                              min2(end3 - zg, jg))));  // Damp all
    else if (gpu_id == 0)
      edge = min2(ig,
                  min2(zg - start3, min2(min2(n2gpu - 2 * radius - jg, n1gpu - 2 * radius - ig),
                                      jg)));  // Don't damp bottom
    else if (gpu_id == (n_gpus - 1))
      edge = min2(ig,
                  min2(min2(n2gpu - 2 * radius - jg, n1gpu - 2 * radius - ig),
                       min2(end3 - zg, jg)));  // Don't damp top
    else
      edge = min2(ig, min2(min2(n2gpu - 2 * radius - jg, n1gpu - 2 * radius - ig),
                                 jg));  // Don't damp top or bottom
    if (edge >= 0 && edge < 40) {
      float temp = expf(-bc_bgpu * (bc_agpu - edge));
      if (temp < 1.) {
        p1[addr] *= temp;
        p0[addr] *= temp;
      }
    }
	addr += stride;
  }
}

extern "C" __global__ void damp_top_kernel(int nblocksz, float *p0, float *p1) {
  int yblock = (blockIdx.x / nblocksz);
  int y0 = FAT + yblock * BLOCKY_SIZE;
  int zl = FAT + threadIdx.x;                                   // local  zdir
  int zg = (blockIdx.x - yblock * nblocksz) * blockDim.x + zl;  // global zdir
  int xl = FAT + threadIdx.y;                                   // local  xdir
  int xg = blockIdx.y * blockDim.y + xl;                        // global xdir

  int stride = n1gpu * n2gpu;
  int bc_agpu = 50;
  float bc_bgpu = 0.0005;

  for (int yl = 0; yl < BLOCKY_SIZE; ++yl) {
    int y = y0 + yl;
    int edge = n1gpu - FAT - zg;  // min2(zg-FAT ,n1gpu-FAT-zg);
    if (edge >= 0) {
      int addr = zg + n1gpu * xg + stride * y;
      float temp = expf(-bc_bgpu * (bc_agpu - edge));
      if (temp < 1.) {
        p1[addr] *= temp;
        p0[addr] *= temp;
      }
    }
  }
}
