//#include "cutil_inline.h"
//#include "cutil_math.h"
#include <vector>
#include <stdlib.h>
#include "gpu_finite_3d.h"
//#include "sep3d.h"
//#include "seplib.h"
#include "cudaErrors.cu"
#include "extra.h"
int ntblock_internal;
#include "assert.h"
#include "wave_fkernel.3d8o.cu"

static float *src_p0[MAX_NUM_GPUS], *src_p1[MAX_NUM_GPUS];
float *source_buf;
int _jt;
int npts_internal, source_blocked, ntsource_internal, ntreceiver_internal;
int rec_nx, rec_ny, _jtsS, _jtdD;

// void setup_cuda(int ngpus, int argc, char **argv){
void setup_cuda(int ngpus) {
  n_gpus = ngpus;
  fprintf(stderr, "Today, we are using %d GPUs; specifically: \n", n_gpus);
  int dr;

  for (int i = 0; i < n_gpus; i++) device[i] = i;

  for (int i = 0; i < n_gpus; i++) {
    cudaDeviceSynchronize();

    cudaSetDevice(device[i]);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device[i]);
    cudaDriverGetVersion(&dr);

    fprintf(stderr, "  GPU %s (%d),", properties.name, device[i]);
    if (properties.unifiedAddressing)
      fprintf(stderr, " UVA initiated,");
    else
      fprintf(stderr, " UVA not working ");
    fprintf(stderr, " driver %d\n", dr);

    // Enable P2P memcopies between GPUs
    if (n_gpus > 1) {
      for (int j = 0; j < n_gpus; j++) {
        if (i == j) continue;
        int peer_access_available = 0;
        cudaDeviceCanAccessPeer(&peer_access_available, device[i], device[j]);
        if (peer_access_available) {
          // fprintf(stderr,"Make the GPUs talk %d %d\n",device[i],device[j]);
          cudaDeviceEnablePeerAccess(device[j], 0);
        }
      }
    }
  }
}

void process_error(const cudaError_t &error, char *string = 0,
                   bool verbose = false) {
  if (error != cudaSuccess || verbose) {
    int current_gpu = -1;
    cudaGetDevice(&current_gpu);

    fprintf(stderr, "GPU %d: ", current_gpu);
    if (string) printf(string);
    fprintf(stderr, ": %s\n", cudaGetErrorString(error));
  }

  if (error != cudaSuccess) exit(-1);
}

extern "C" __global__ void new_src_inject_kernel(int it, int isinc, float *p) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("old source data: %f\n", p[srcgeom_gpu0[ix]]);
//  printf("source %d, dir_gpu: %f, sources: %08x %08x %08x %08x, index start %d\n", ix, dir_gpu,
//		  *((uint32_t*)&source_gpu0[ntrace_gpu * ix + it]),
//		  *((uint32_t*)&source_gpu0[ntrace_gpu * ix + it + 1]),
//		  *((uint32_t*)&source_gpu0[ntrace_gpu * ix + it + 2]),
//		  *((uint32_t*)&source_gpu0[ntrace_gpu * ix + it + 3]),
//		  ntrace_gpu * ix + it
//		  );
//  printf("sinc_s_table index: %d", isinc * nsinc_gpu);
//  printf("sinc_s_table: %08x %08x %08x %08x\n",
//		  *((uint32_t*)&sinc_s_table[isinc * nsinc_gpu]),
//		  *((uint32_t*)&sinc_s_table[isinc * nsinc_gpu + 1]),
//		  *((uint32_t*)&sinc_s_table[isinc * nsinc_gpu + 2]),
//		  *((uint32_t*)&sinc_s_table[isinc * nsinc_gpu + 3])
//		  );
  p[srcgeom_gpu0[ix]] += dir_gpu *
      (sinc_s_table[isinc * nsinc_gpu] * source_gpu0[ntrace_gpu * ix + it] +
       sinc_s_table[isinc * nsinc_gpu + 1] * source_gpu0[ntrace_gpu * ix + it + 1] +
       sinc_s_table[isinc * nsinc_gpu + 2] * source_gpu0[ntrace_gpu * ix + it + 2] +
       sinc_s_table[isinc * nsinc_gpu + 3] * source_gpu0[ntrace_gpu * ix + it + 3] +
       sinc_s_table[isinc * nsinc_gpu + 4] * source_gpu0[ntrace_gpu * ix + it + 4] +
       sinc_s_table[isinc * nsinc_gpu + 5] * source_gpu0[ntrace_gpu * ix + it + 5] +
       sinc_s_table[isinc * nsinc_gpu + 6] * source_gpu0[ntrace_gpu * ix + it + 6] +
       sinc_s_table[isinc * nsinc_gpu + 7] * source_gpu0[ntrace_gpu * ix + it + 7]);
  //printf("new source data: %f\n", p[srcgeom_gpu0[ix]]);
}

extern "C" __global__ void new_data_inject_kernel(int it, int isinc, float *p, int grid_x_width) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = k + grid_x_width * j;
  if (i < rec_nx_gpu * rec_ny_gpu) {
    //if (i == 0)
    //  printf("old receiver data: %f\n", p[datageom_gpu0[i]]);
    p[datageom_gpu0[i]] += dir_gpu *
		(sinc_d_table[isinc * nsinc_gpu] * data_gpu0[ntrace_gpu * i + it] +
         sinc_d_table[isinc * nsinc_gpu + 1] * data_gpu0[ntrace_gpu * i + it + 1] +
         sinc_d_table[isinc * nsinc_gpu + 2] * data_gpu0[ntrace_gpu * i + it + 2] +
         sinc_d_table[isinc * nsinc_gpu + 3] * data_gpu0[ntrace_gpu * i + it + 3] +
         sinc_d_table[isinc * nsinc_gpu + 4] * data_gpu0[ntrace_gpu * i + it + 4] +
         sinc_d_table[isinc * nsinc_gpu + 5] * data_gpu0[ntrace_gpu * i + it + 5] +
         sinc_d_table[isinc * nsinc_gpu + 6] * data_gpu0[ntrace_gpu * i + it + 6] +
         sinc_d_table[isinc * nsinc_gpu + 7] * data_gpu0[ntrace_gpu * i + it + 7]);
    //if (i == 0)
    //  printf("new receiver data: %f\n", p[datageom_gpu0[i]]);
  }
}

extern "C" __global__ void zero_data(float *p) {
  long long j = blockIdx.y * blockDim.y + threadIdx.y;
  long long k = blockIdx.x * blockDim.x + threadIdx.x;
  long long i = k + (n1gpu)*j;
  long long it;
  long long nt = ntblock_gpu;

  if (i < (rec_nx_gpu * rec_ny_gpu)) {
    for (it = 0; it < ntblock_gpu; it++) {
      data_gpu0[nt * i + it] = 0;
    }
  }
}
extern "C" __global__ void move_zero_data(float *p) {
  long long j = blockIdx.y * blockDim.y + threadIdx.y;
  long long k = blockIdx.x * blockDim.x + threadIdx.x;
  long long i = k + (n1gpu)*j;
  long long it;
  long long nt = ntblock_gpu - 7;

  if (i < (rec_nx_gpu * rec_ny_gpu)) {
    for (it = 0; it < 7; it++)
      data_gpu0[ntblock_gpu * i + it] = data_gpu0[ntblock_gpu * i + it + nt];
    for (it = 7; it < ntblock_gpu; it++) data_gpu0[ntblock_gpu * i + it] = 0;
  }
}
extern "C" __global__ void new_data_extract_kernel(int it, int isinc,
                                                   float *p) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = k + (n1gpu)*j;

  if (i < (rec_nx_gpu * rec_ny_gpu)) {
    data_gpu0[ntblock_gpu * (i) + it + 0] +=
        p[datageom_gpu0[i]] * sinc_d_table[isinc * nsinc_gpu + 0];
    data_gpu0[ntblock_gpu * (i) + it + 1] +=
        p[datageom_gpu0[i]] * sinc_d_table[isinc * nsinc_gpu + 1];
    data_gpu0[ntblock_gpu * (i) + it + 2] +=
        p[datageom_gpu0[i]] * sinc_d_table[isinc * nsinc_gpu + 2];
    data_gpu0[ntblock_gpu * (i) + it + 3] +=
        p[datageom_gpu0[i]] * sinc_d_table[isinc * nsinc_gpu + 3];
    data_gpu0[ntblock_gpu * (i) + it + 4] +=
        p[datageom_gpu0[i]] * sinc_d_table[isinc * nsinc_gpu + 4];
    data_gpu0[ntblock_gpu * (i) + it + 5] +=
        p[datageom_gpu0[i]] * sinc_d_table[isinc * nsinc_gpu + 5];
    data_gpu0[ntblock_gpu * (i) + it + 6] +=
        p[datageom_gpu0[i]] * sinc_d_table[isinc * nsinc_gpu + 6];
    data_gpu0[ntblock_gpu * (i) + it + 7] +=
        p[datageom_gpu0[i]] * sinc_d_table[isinc * nsinc_gpu + 7];
  }
}

extern "C" __global__ void img_kernel(float *img, float *dat, float *src) {
  int ig = blockIdx.x * blockDim.x + threadIdx.x;
  int jg = blockIdx.y * blockDim.y + threadIdx.y;
  if (ig >= n1gpu || jg >= n2gpu) return;
  int addr = ig + n1gpu * jg;
  int stride = n1gpu * n2gpu;
  for (int iy = 0; iy < n3gpu; iy++) {
    img[addr] += dat[addr] * src[addr];
    // printf("%d %d %d\n", img[addr], dat[addr], src[addr]);
    //img[addr] += dat[addr] * src[addr];
    addr += stride;
  }
}

extern "C" __global__ void img_add_kernel(
    float *img, float *rec_field, float *src_field) {  // as above, added
  long long ig = blockIdx.x * blockDim.x + threadIdx.x;
  long long jg = blockIdx.y * blockDim.y + threadIdx.y;
  long long addr = ig + (long long)n1gpu * jg;
  long long stride = (long long)n1gpu * (long long)n2gpu;
  for (long long iy = 0; iy < n3gpu; iy++) {
    rec_field[addr] += .000001 * src_field[addr] * (img[addr]);
    addr += stride;
  }
}
void source_prop(int n1, int n2, int n3, bool damp, bool get_last, float *p0,
                 float *p1, int jt, int npts, int nt) {
  // Propagate the source wavefield and return the final two 3D wavefield slices
  float *ptemp;

  cudaError_t error = cudaSuccess;

  // int n3_total=n3;
  n3 = (n3 - 2 * radius) / n_gpus + 2 * radius; // this is local n3 size
  // int dim3=n3;
  // if(n_gpus > 1) dim3-=2*radius;

  int dir = 1;
  _jt = jt;
  // TODO: check this
  float sc = (float)dir / (float)_jt;

  int n_bytes_gpu = (n1 * n2 * n3 + lead_pad) * sizeof(float);

  for (int i = 0; i < n_gpus; i++) {
    cudaSetDevice(device[i]);
    // cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    cudaMalloc((void **)&src_p0[i], n_bytes_gpu);
    cudaMalloc((void **)&src_p1[i], n_bytes_gpu);

    cudaMemset(src_p0[i], 0, n_bytes_gpu);
    cudaMemset(src_p1[i], 0, n_bytes_gpu);

    cudaMemcpyToSymbol(dir_gpu, &sc, sizeof(float));
  }

  // fprintf(stderr,"Allocate %d %d %d, %f mbs;
  // %d\n",n1,n2,n3,(float)(n1*n2*n3*4/1000000),lead_pad);

  // Blocks for internal data
  // int nblocks1=(n1-2*FAT)/(2*BLOCKX_SIZE);
  int nblocks1 = (n1 - 2 * FAT + BLOCKX_SIZE - 1) / BLOCKX_SIZE;
  int nblocks2 = (n2 - 2 * FAT + BLOCKY_SIZE - 1) / BLOCKY_SIZE;
  int last_x_block = nblocks1 - 1;
  int last_y_block = nblocks2 - 1;

  dim3 dimBlock(BLOCKX_SIZE, BLOCKY_SIZE);
  dim3 dimGrid(nblocks1, nblocks2);
  //dim3 dimGrid((int)ceilf(1. * (n1 - 2 * FAT) / BLOCKX_SIZE),
  //              (int)ceilf(1. * (n2 - 2 * FAT) / BLOCKY_SIZE));
  fprintf(stderr, "Kernel launch size is %d * %d\n", dimGrid.x, dimGrid.y);

  // Define separate streams for overlapping communication
  cudaStream_t stream_halo[n_gpus], stream_internal[n_gpus];
  cudaEvent_t start, stop;

  cudaSetDevice(device[0]);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Setup coordinate systems for internal domains
  int offset_internal[n_gpus];
  int start3[n_gpus], end3[n_gpus];

  for (int i = 0; i < n_gpus; i++) {
    cudaSetDevice(device[i]);
    cudaStreamCreate(&stream_halo[i]);
    cudaStreamCreate(&stream_internal[i]);

    // Offset_internal is the initial index of our internal domain (out of
    // stencil padding)
    offset_internal[i] = offset;
    if (i > 0) offset_internal[i] += n1 * n2 * radius;

    start3[i] = i * (n3 - 2 * radius) + 2 * radius; // this is correct
    end3[i] = (i + 1) * (n3 - 2 * radius); // this is correct
  }

  start3[0] = radius;
  // TODO: check this
  end3[n_gpus - 1] = n_gpus * (n3 - 2 * radius) + radius;
      //n_gpus * (n3 - 2 * radius);  // I THINK THIS SHOULD BE -RADIUS. LET'S TRY
      //n_gpus * (n3 - radius);  // I THINK THIS SHOULD BE -RADIUS. LET'S TRY
  // start3[0]=0;
  // end3[n_gpus-1]=n_gpus*(n3-2*radius);

  // Set up coordinate systems for the halo exchange
  int offset_snd_h1 = lead_pad + n1 * n2 * radius;
  int offset_snd_h2 = lead_pad + n1 * n2 * (n3 - 2 * radius);
  int offset_rcv_h1 = lead_pad;
  int offset_rcv_h2 = lead_pad + n1 * n2 * (n3 - radius);
  int offset_cmp_h1 = offset_snd_h1 + n1 * radius + radius;
  int offset_cmp_h2 = offset_snd_h2 + n1 * radius + radius;
  //int offset_rcv_h1 = lead_pad;
  //int offset_rcv_h2 = lead_pad + n1 * n2 * (n3 - 2 * radius + radius);
  //int offset_cmp_h1 = offset;
  //long int offset_cmp_h2 =
  //    lead_pad + radius + radius * n1 + n1 * n2 * (n3 - 2 * radius);  //-radius?

  /*int offset_snd_h1=lead_pad+n1*n2*radius;
  int offset_snd_h2=lead_pad+n1*n2*(n3-2*radius);
  int offset_rcv_h1=lead_pad;
  int offset_rcv_h2=lead_pad+n1*n2*(n3-2*radius+radius);
  int offset_cmp_h1=offset;
  long int
  offset_cmp_h2=lead_pad+radius+radius*n1+radius*n1*n2+n1*n2*(n3-2*radius-radius);*/

  cudaSetDevice(device[0]);
  cudaEventRecord(start, 0);

  int src_counter = 0;
  for (int it = 0; it <= nt; it++) {
    int id = it / _jtsS;
    int ii = it - id * _jtsS;

    // Calculate the halo regions first
    for (int i = 0; i < n_gpus; i++) {
      cudaSetDevice(i);

      if (i > 0) {
        wave_kernel<<<dimGrid, dimBlock, 0, stream_halo[i]>>>(
            src_p0[i] + offset_cmp_h1, src_p1[i] + offset_cmp_h1,
            src_p0[i] + offset_cmp_h1, velocity[i] + offset_cmp_h1,
			radius, radius * 2, last_x_block, last_y_block);
      }

      if (i < n_gpus - 1) {
        wave_kernel<<<dimGrid, dimBlock, 0, stream_halo[i]>>>(
            src_p0[i] + offset_cmp_h2, src_p1[i] + offset_cmp_h2,
            src_p0[i] + offset_cmp_h2, velocity[i] + offset_cmp_h2,
            radius, radius * 2, last_x_block, last_y_block);
      }

      cudaStreamQuery(stream_halo[i]);
    }

	// calculate the internal region
    for (int i = 0; i < n_gpus; i++) {
      cudaSetDevice(device[i]);

	  //fprintf(stderr, "wave kernel from %d, range is %d\n", offset_internal[i], (end3[i] - start3[i]));
      wave_kernel<<<dimGrid, dimBlock, 0, stream_internal[i]>>>(
          src_p0[i] + offset_internal[i], src_p1[i] + offset_internal[i],
          src_p0[i] + offset_internal[i], velocity[i] + offset_internal[i],
          start3[i], end3[i], last_x_block, last_y_block);
  error = cudaGetLastError();
  process_error(error, "wave_kernel\n");
    }

    // Overlap internal computation with halo communication
    // Send halos to the 'right'
    for (int i = 0; i < n_gpus - 1; i++) {
      cudaMemcpyPeerAsync(src_p0[i + 1] + offset_rcv_h1, i + 1,
                          src_p0[i] + offset_snd_h2, i,
                          n1 * n2 * radius * sizeof(float), stream_halo[i]);
    }

    // Synchronize to avoid stalling
    for (int i = 0; i < n_gpus - 1; i++) {
      cudaSetDevice(i);
      cudaStreamSynchronize(stream_halo[i]);
    }

    // Send halos to the 'left'
    for (int i = 1; i < n_gpus; i++) {
      cudaMemcpyPeerAsync(src_p0[i - 1] + offset_rcv_h2, i - 1,
                          src_p0[i] + offset_snd_h1, i,
                          n1 * n2 * radius * sizeof(float), stream_halo[i]);
    }

    for (int i = 0; i < n_gpus; i++) {
       cudaSetDevice(i);
       cudaStreamSynchronize(stream_halo[i]);
      if (i == shot_gpu) // TODO: should this shot_gpu change?
      {
        if (id + 7 < ntsource_internal)
		{
          new_src_inject_kernel<<<1, npts_internal, 0, stream_internal[i]>>>(
              id, ii, src_p0[i] + lead_pad);
		  src_counter++;
  error = cudaGetLastError();
  process_error(error, "src_inject\n");
		}
        }
    }

    // Synchronise GPUs and do pointer exchange
    for (int i = 0; i < n_gpus; i++) {
      cudaSetDevice(device[i]);
      cudaDeviceSynchronize();
      ptemp = src_p0[i];
      src_p0[i] = src_p1[i];
      src_p1[i] = ptemp;
    }
  }

  error = cudaGetLastError();
  process_error(error, "kernel");
  fprintf(stderr, "Number of src injections: %d\n", src_counter);

  // Use device 0 to give a performance report
  cudaSetDevice(device[0]);
  cudaEventRecord(stop, 0);

  cudaDeviceSynchronize();

  float time_total;
  cudaEventElapsedTime(&time_total, start, stop);
  fprintf(stderr, "Time for source propagation = %f seconds \n",
          time_total / 1000);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //if (get_last) {
  //  for (int i = 0; i < n_gpus; i++) {
  //    cudaSetDevice(device[i]);
  //    cudaMemcpy(p0 + i * n1 * n2 * (n3 - 2 * radius), // this is correct
  //               src_p0[i] + lead_pad /*+radius*n1*n2*/,
  //               n1 * n2 * n3 * sizeof(float),
  //               cudaMemcpyDeviceToHost);
  //    cudaMemcpy(p1 + i * n1 * n2 * (n3 - 2 * radius),
  //               src_p1[i] + lead_pad /*+radius*n1*n2*/,
  //               n1 * n2 * n3 * sizeof(float),
  //               cudaMemcpyDeviceToHost);
  //    // cudaMemcpy(p0+i*n1*n2*(n3-radius), src_p0[i]+radius*n1*n2,
  //    // n1*n2*(n3-radius)*sizeof(float), cudaMemcpyDeviceToHost);
  //    // cudaMemcpy(p1+i*n1*n2*(n3-radius), src_p1[i]+radius*n1*n2,
  //    // n1*n2*(n3-radius)*sizeof(float), cudaMemcpyDeviceToHost);
  //    // cudaMemcpy(src_p0[i]+lead_pad/*+radius*n1*n2*/,
  //    // p0_s_cpu+i*n1*n2*(n3-2*radius), n1*n2*(n3/*-2*radius*/)*sizeof(float),
  //  }
  //}

  //for (int i = 0; i < n_gpus; i++) {
  //  cudaSetDevice(device[i]);
  //  cudaFree(src_p0[i]);
  //  cudaFree(src_p1[i]);
  //}
  // cudaFree(source_gpu);
  // cudaFree(srcgeom_gpu);
  if (nt % 2 == 1)
  {
	  for (int i = 0; i < n_gpus; ++i)
	  {
	    float *temp = src_p0[i];
		src_p0[i] = src_p1[i];
		src_p1[i] = temp;
		  //cudaMemcpy(src_p1[i], src_p0[i], (n1 * n2 * n3 + lead_pad) * sizeof(float), cudaMemcpyDeviceToDevice);
	  }
  }
}

void rtm_forward(int n1, int n2, int n3, int jt, float *img, float *dat,
                 int npts_src, int nt, int nt_big, int rec_nx, int rec_ny) {
  //
  //
  //
  // if(1==1){
  //
  //  //Born modelling over input image
  //  float *src_p0[n_gpus], *src_p1[n_gpus], *data_p0[n_gpus],
  //  *data_p1[n_gpus], *img_gpu[n_gpus]; float *ptemp, *ptemp2;
  //
  //  cudaError_t error = cudaSuccess;
  //
  //  int n3_total=n3;
  //  n3=(n3-2*radius)/n_gpus + 2*radius;
  //
  //  int dir=1;
  //
  //  int n3s=n3-2*radius;
  //
  //  int nblocks1=(n1-2*FAT)/BLOCKZ_SIZE;
  //  int nblocks2=(n2-2*FAT)/BLOCKX_SIZE;
  //  //int nblocks3=(n3-2*FAT)/BLOCKY_SIZE;
  //
  //  dim3 dimGrid(nblocks1,nblocks2);
  //  dim3 dimBlock(16, 16);
  //
  //  //dim3
  //  dimGridx((int)ceilf(1.*n1/BLOCKX_SIZE),(int)ceilf(1.*n2/BLOCKY_SIZE));
  //  dim3
  //  dimGridx((int)ceilf(1.*n1/BLOCKX_SIZE),(int)ceilf(1.*n2/BLOCKY_SIZE));
  // fprintf(stderr,"CEHCK GRID %d %d %d %d
  // \n",dimBlock.x,dimBlock.y,dimGridx.x,dimGridx.y);
  //
  //
  //
  //  cudaStream_t stream_halo[n_gpus], stream_internal[n_gpus];
  //  cudaEvent_t start,stop;
  //
  //  cudaSetDevice(device[0]);
  //  cudaEventCreate(&start);
  //  cudaEventCreate(&stop);
  //
  //  int offset_internal[n_gpus];
  //  int start3[n_gpus],end3[n_gpus];
  //
  //  for(int i=0; i<n_gpus; i++){
  //    cudaSetDevice(device[i]);
  //    cudaStreamCreate(&stream_halo[i]);
  //    cudaStreamCreate(&stream_internal[i]);
  //
  //    offset_internal[i]=offset;
  //    if(i > 0) offset_internal[i] += n1*n2*radius;
  //
  //    start3[i] = i*(n3-2*radius) + 2*radius;
  //    end3[i] = (i+1)*(n3-2*radius);
  //  }
  //
  //  start3[0]=radius;
  //  end3[n_gpus-1]=n_gpus*(n3-2*radius);
  //
  //  for(int i=0; i<n_gpus; i++){
  //
  //    cudaSetDevice(device[i]);
  //    cudaMalloc((void**) &src_p0[i], (n1*n2*n3+lead_pad)*sizeof(float));
  //    cudaMalloc((void**) &src_p1[i], (n1*n2*n3+lead_pad)*sizeof(float));
  //    cudaMalloc((void**) &data_p0[i],(n1*n2*n3+lead_pad)*sizeof(float));
  //    cudaMalloc((void**) &data_p1[i],(n1*n2*n3+lead_pad)*sizeof(float));
  //    cudaMalloc((void**) &img_gpu[i], n1*n2*n3*sizeof(float));
  //
  //    cudaMemset(data_p0[i], 0,(n1*n2*n3+lead_pad)*sizeof(float));
  //    cudaMemset(data_p1[i], 0,(n1*n2*n3+lead_pad)*sizeof(float));
  //    cudaMemset(src_p0[i],  0,(n1*n2*n3+lead_pad)*sizeof(float));
  //    cudaMemset(src_p1[i],  0,(n1*n2*n3+lead_pad)*sizeof(float));
  //    cudaMemset(img_gpu[i], 0,n1*n2*n3*sizeof(float));
  //
  //    cudaMemcpy( img_gpu[i]/*+radius*n1*n2*/, img+i*n1*n2*(n3-2*radius),
  //    n1*n2*(n3/*-2*radius*/)*sizeof(float), cudaMemcpyHostToDevice);
  //    //DONT DELETE need the -2*r gone to be multi-gpu invariant
  //    //cudaMemcpy( img_gpu[i]+radius*n1*n2, img+i*n1*n2*(n3-2*radius),
  //    n1*n2*(n3-2*radius)*sizeof(float), cudaMemcpyHostToDevice);
  //    cudaMemcpyToSymbol(dir_gpu, &dir, sizeof(float));
  //  }
  //
  //  int offset_snd_h1=lead_pad+n1*n2*radius;
  //  int offset_snd_h2=lead_pad+n1*n2*(n3-2*radius);
  //  int offset_rcv_h1=lead_pad+0;
  //  int offset_rcv_h2=lead_pad+n1*n2*(n3-2*radius+radius);
  //
  //  int offset_cmp_h1=offset;
  //  int offset_cmp_h2=lead_pad+radius+radius*n1+n1*n2*(n3-radius-radius);
  //
  //  cudaSetDevice(device[0]);
  //  cudaEventRecord(start,0);
  //  long long iblock=0;
  //  int icycle=-1;
  //  //zero_data<<<dimGridx,dimBlock,0,stream_internal[0]>>>(data_p0[0]);
  ///*
  // float
  // *temp=(float*)malloc(sizeof(float)*rec_nx*rec_ny*(ntblock_internal+7));
  //      cudaMemcpy(temp, data_gpu,
  //      (7+ntblock_internal)*rec_nx*rec_ny*sizeof(float),
  //      cudaMemcpyDeviceToHost);
  //     srite("book.H",temp,(7+ntblock_internal)*rec_nx*rec_ny*sizeof(float));
  // free(temp);
  //*/
  //
  //  for(int it=0; it < nt; it++){
  //    int id=it/jt;
  //    int ii=it-id*jt;
  //    int id_block=id-((int)(id/ntblock_internal))*ntblock_internal;
  //    if(it%100==10){
  //  //   fprintf(stderr,"WRITING WAVEFIELD %d %d %d \n",n1,n2,n3_total);
  //  //   writeWavefield("src.H",src_p0,n3s,n_gpus,n1,n2,n3_total,radius);
  //   // writeWavefield("dat.H",data_p0,n3s,n_gpus,n1,n2,n3_total,radius);
  //
  //
  //
  //    }
  //
  //    for(int i=0; i<n_gpus; i++){
  //      cudaSetDevice(device[i]);
  //      if(i>0){
  //        wave_kernel_adj<<<dimGrid,dimBlock,0,stream_halo[i]>>>(data_p0[i]+offset_cmp_h1,
  //        data_p1[i]+offset_cmp_h1, data_p0[i]+offset_cmp_h1,
  //        velocity[i]+offset_cmp_h1, radius, 2*radius);
  //        wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(src_p0[i]+offset_cmp_h1,
  //        src_p1[i]+offset_cmp_h1, src_p0[i]+offset_cmp_h1,
  //        velocity[i]+offset_cmp_h1, radius, 2*radius);
  //      }
  //      if(i<n_gpus-1){
  //        wave_kernel_adj<<<dimGrid,dimBlock,0,stream_halo[i]>>>(data_p0[i]+offset_cmp_h2,
  //        data_p1[i]+offset_cmp_h2, data_p0[i]+offset_cmp_h2,
  //        velocity[i]+offset_cmp_h2, (n3-radius)-radius, n3-radius);
  //        wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(src_p0[i]+offset_cmp_h2,
  //        src_p1[i]+offset_cmp_h2, src_p0[i]+offset_cmp_h2,
  //        velocity[i]+offset_cmp_h2, (n3-radius)-radius, n3-radius);
  //      }
  //      cudaStreamQuery(stream_halo[i]);
  //    }
  //
  //    if(ii==0) {
  //      icycle++;
  //      if(icycle%ntblock_internal==0){
  //        load_source(id);
  //      }
  //    }
  //
  //    for(int i=0; i<n_gpus; i++){
  //      cudaSetDevice(device[i]);
  //
  //      damp_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(data_p0[i],
  //      data_p1[i], start3[i], end3[i], i, n_gpus);
  //      damp_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(data_p0[i],
  //      data_p1[i], start3[i], end3[i], i, n_gpus);
  //      damp_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(src_p0[i],
  //      src_p1[i], start3[i], end3[i], i, n_gpus);
  //
  //      wave_kernel_adj<<<dimGrid,dimBlock,0,stream_internal[i]>>>(data_p0[i]+offset_internal[i],
  //      data_p1[i]+offset_internal[i], data_p0[i]+offset_internal[i],
  //      velocity[i]+offset_internal[i], start3[i], end3[i]);
  //
  //      wave_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(src_p0[i]+offset_internal[i],
  //      src_p1[i]+offset_internal[i], src_p0[i]+offset_internal[i],
  //      velocity[i]+offset_internal[i], start3[i], end3[i]);
  //
  //    fprintf(stderr,"in m0odeling %d of %d %d \n",it,nt,id_block);
  //
  //      if(i==shot_gpu) {
  //
  //         if(npts_src<100)
  //         new_src_inject_kernel<<<1,npts_src,0,stream_internal[i]>>>(icycle
  //         ,ii,src_p0[i]+lead_pad);
  //
  //         else{
  //           new_src_inject2_kernel
  //           <<<dimGridx,dimBlock,0,stream_internal[i]>>>(icycle
  //           ,ii,src_p0[i]+lead_pad,source_gpu);
  //         }
  //      }
  //    }
  //
  // if(1==3){
  // float
  // *temp=(float*)malloc(sizeof(float)*rec_nx*rec_ny*(ntblock_internal+7));
  //      cudaMemcpy(temp, data_gpu,
  //      (7+ntblock_internal)*rec_nx*rec_ny*sizeof(float),
  //      cudaMemcpyDeviceToHost);
  //      srite("book.H",temp,(7+ntblock_internal)*rec_nx*rec_ny*sizeof(float));
  // free(temp);
  //}
  //    for(int i=0; i<n_gpus-1; i++){
  //      cudaMemcpyPeerAsync(data_p0[i+1]+offset_rcv_h1,i+1,data_p0[i]+offset_snd_h2,i,n1*n2*radius*sizeof(float),stream_halo[i]);
  //      cudaMemcpyPeerAsync(src_p0[i+1]+offset_rcv_h1,i+1,src_p0[i]+offset_snd_h2,i,n1*n2*radius*sizeof(float),stream_halo[i]);
  //    }
  //    for(int i=0; i<n_gpus-1; i++){
  //      cudaSetDevice(device[i]);
  //      cudaStreamSynchronize(stream_halo[i]);
  //    }
  //    for(int i=1; i<n_gpus; i++){
  //      cudaMemcpyPeerAsync(data_p0[i-1]+offset_rcv_h2,i-1,data_p0[i]+offset_snd_h1,i,n1*n2*radius*sizeof(float),stream_halo[i]);
  //      cudaMemcpyPeerAsync(src_p0[i-1]+offset_rcv_h2,i-1,src_p0[i]+offset_snd_h1,i,n1*n2*radius*sizeof(float),stream_halo[i]);
  //    }
  //    for(int i=0; i<n_gpus; i++){
  //      cudaSetDevice(device[i]);
  //      cudaSetDevice(device[i]);
  //      cudaStreamSynchronize(stream_internal[i]);
  //    }
  //    for(int i=0; i<n_gpus; i++){
  //      cudaSetDevice(device[i]);
  //      //if(it%jt==0)
  //      img_add_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(img_gpu[i]+offset_snd_h1,data_p0[i]+offset_snd_h1,src_p0[i]+offset_snd_h1);
  //      if(it%jt==0)
  //      img_add_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(img_gpu[i],data_p0[i],src_p0[i]);
  //      //if(it%jt==0)
  //      img_add_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(img_gpu[i]+offset_internal[i]-offset,data_p0[i]+offset_internal[i]-offset,src_p0[i]+offset_internal[i]-offset);
  //      //works for multis
  //    }
  //    fprintf(stderr,"in 2modeling %d of %d %d \n",it,nt,id_block);
  //
  //    for(int i=0; i<n_gpus; i++){
  //      cudaSetDevice(device[i]);
  //      cudaStreamSynchronize(stream_internal[i]);
  //      cudaDeviceSynchronize();
  //    }
  //
  //    cudaSetDevice(device[0]);
  //
  //    if(icycle==ntblock_internal){
  //      icycle=0;
  //
  //      float
  //      *temp=(float*)malloc(sizeof(float)*rec_nx*rec_ny*(ntblock_internal+7));
  //      cudaMemcpy(temp, data_gpu,
  //      (7+ntblock_internal)*rec_nx*rec_ny*sizeof(float),
  //      cudaMemcpyDeviceToHost);
  //      srite("book.H",temp,(7+ntblock_internal)*rec_nx*rec_ny*sizeof(float));
  //    float sm=0;
  //      for(int k=0; k <rec_nx*rec_ny*ntblock_internal; k++){
  //        sm+=fabs(temp[k]);
  //      }
  //      int itr=0;
  //      for(int iy=0; iy < rec_ny; iy++){
  //        for(int ix=0; ix < rec_nx; ix++,itr++){
  //           memcpy(&dat[nt_big*itr+iblock*ntblock_internal],&temp[(ntblock_internal+7)*itr],ntblock_internal*sizeof(float));
  //        }
  //      }
  //      iblock++;
  //      move_zero_data<<<dimGridx,dimBlock,0,stream_internal[0]>>>(data_p0[0]);
  //    //  cudaMemcpy(temp, data_gpu,
  //    (7+ntblock_internal)*rec_nx*rec_ny*sizeof(float),
  //    cudaMemcpyDeviceToHost);
  //    //
  //    srite("book.H",temp,(7+ntblock_internal)*rec_nx*rec_ny*sizeof(float));
  //
  //      free(temp);
  //    }
  //    new_data_extract_kernel<<<dimGridx,dimBlock,0,stream_internal[0]>>>(icycle
  //    ,ii,data_p0[0]/*,datageom_gpu,data_gpu+offset_snd_h1*/);
  //    fprintf(stderr,"in modelaxing %d of %d %d \n",it,nt,id_block);
  //
  //    for(int i=0; i<n_gpus; i++){
  //      cudaSetDevice(i);
  //      cudaDeviceSynchronize();
  //      ptemp2=src_p1[i]; src_p1[i]=src_p0[i]; src_p0[i]=ptemp2;
  //      ptemp=data_p1[i]; data_p1[i]=data_p0[i]; data_p0[i]=ptemp;
  //    }
  //   /*
  //    fprintf(stderr,"in modelin2g %d of %d %d \n",it,nt,id_block);
  // float
  // *temp=(float*)malloc(sizeof(float)*rec_nx*rec_ny*(ntblock_internal+7));
  //      cudaMemcpy(temp, data_gpu,
  //      (7+ntblock_internal)*rec_nx*rec_ny*sizeof(float),
  //      cudaMemcpyDeviceToHost);
  //      srite("book.H",temp,(7+ntblock_internal)*rec_nx*rec_ny*sizeof(float));
  // free(temp);
  //
  //     writeWavefield("src.H",src_p0,n3s,n_gpus,n1,n2,n3_total,radius);
  //    seperr("");
  //    */
  //  }
  //
  //  //error = cudaGetLastError();
  //  //process_error( error, "kernel" );
  //  int ic=0;
  //  if(nt_big!=ntblock_internal*iblock){
  //     float
  //     *temp=(float*)malloc(sizeof(float)*rec_nx*rec_ny*(ntblock_internal+7));
  //     cudaMemcpy(temp, data_gpu,
  //       (7+ntblock_internal)*rec_nx*rec_ny*sizeof(float),
  //         cudaMemcpyDeviceToHost);
  //
  //     int itr=0;
  //     for(int iy=0; iy < rec_ny; iy++){
  //        for(int ix=0; ix < rec_nx; ix++,itr++){
  //          memcpy(&dat[nt_big*itr+iblock*ntblock_internal],
  //           &temp[(ntblock_internal+7)*itr],(nt_big-ntblock_internal*iblock)*sizeof(float));
  //        }
  //     }
  //
  //           float sm=0;
  //      for(int k=0; k <rec_nx*rec_ny*ntblock_internal; k++){
  //        sm+=fabs(temp[k]);
  //        if(fabs(temp[k])>.000001) ic++;
  //      }
  //     free(temp);
  //   }
  //
  //
  //  cudaSetDevice(device[0]);
  //  cudaEventRecord(stop,0);
  //
  //  cudaDeviceSynchronize();
  //
  //  float time_total;
  //  cudaEventElapsedTime(&time_total,start,stop);
  //  fprintf(stderr,"Time for Born modelling = %f seconds \n",time_total/1000);
  //
  //  cudaEventDestroy(start);
  //  cudaEventDestroy(stop);
  //  cudaSetDevice(device[0]);
  // // cudaMemcpy(dat, data_gpu, nt_big*rec_nx*rec_ny*sizeof(float),
  // cudaMemcpyDeviceToHost);
  //  for(int i=0; i<n_gpus; i++){
  //
  //    cudaSetDevice(i);
  //
  //    cudaFree(data_p0[i]);
  //
  //
  //    cudaFree(img_gpu[i]);
  //
  //    cudaFree(data_p1[i]);
  //
  //    cudaFree(src_p0[i]);
  //
  //    cudaFree(src_p1[i]);
  //
  //
  //  }
  //  cudaSetDevice(0);
  //
  //  cudaFree(data_gpu);
  //
  //  cudaFree(datageom_gpu);
  //
  //  cudaFree(source_gpu);
  //
  //  cudaFree(srcgeom_gpu);
  //
  //
  //}
  //
  //
  //
  //
}

// void writeWavefield(char *tag, float **dat, int n3s,int ngpu, int n1, int n2,
// int n3,int edge){
//  long long big_block;
//  int igpu,block;
//  long long doing,done,toDo;
//  long long big=500*1000*1000;
//  big_block=(long long)n1*(long long)n2 *(long long)n3;
//  big_block=big_block/ngpu+n1*n2;
//
// fprintf(stderr,"CXXX %d %d %d \n",n1,n2,n3s*2);
//  float *buf=(float*)malloc(sizeof(float*)*big_block);
//  for(igpu=0; igpu < 2; igpu++){
//      toDo=(long long)n1*(long long)n2*n3s;
//     cudaMemcpy(buf,dat[igpu]+n1*n2*edge,toDo*sizeof(float),cudaMemcpyDeviceToHost);
//     done=0;
//     while(done < toDo){
//       doing=toDo; if(doing > big) doing=big;
//       block=doing;
//       srite(tag,&buf[done],block*4);
//       done+=doing;
//     }
//   }
//   free(buf);
//   // seperr("");
//}

void rtm_adjoint(int n1, int n2, int n3, int jt, float *p0_s_cpu,
                 float *p1_s_cpu, float *img, int npts_src, int nt) {
  float *data_p0[n_gpus], *data_p1[n_gpus], *img_gpu[n_gpus];
  float *ptemp, *ptemp2;

  cudaError_t error = cudaSuccess;

  int n3_total = n3;
  n3 = (n3 - 2 * radius) / n_gpus + 2 * radius; // local n3 size

  int dir = -1;
  // TODO: check this
  float sc = (float)dir / (float)_jt;

  for (int i = 0; i < n_gpus; i++) {
    cudaSetDevice(device[i]);
	float *temp = src_p0[i];
	src_p0[i] = src_p1[i];
	src_p1[i] = temp;
    //cudaMalloc((void **)&src_p0[i],
    //           (n1 * n2 * n3 + lead_pad) * sizeof(float));
    ////cudaMalloc((void **)&src_p0[i],
    ////           (n1 * n2 * n3 + radius * n1 * n2 + lead_pad) * sizeof(float));
    //cudaMalloc((void **)&src_p1[i],
    //           (n1 * n2 * n3 + lead_pad) * sizeof(float));
    //cudaMalloc((void **)&src_p1[i],
    //           (n1 * n2 * n3 + radius * n1 * n2 + lead_pad) * sizeof(float));
    cudaMalloc((void **)&data_p0[i], (n1 * n2 * n3 + lead_pad) * sizeof(float));
    cudaMalloc((void **)&data_p1[i], (n1 * n2 * n3 + lead_pad) * sizeof(float));

    cudaMalloc((void **)&img_gpu[i], (n1 * n2 * n3) * sizeof(float));

    cudaMemset(data_p0[i], 0, (n1 * n2 * n3 + lead_pad) * sizeof(float));
    cudaMemset(data_p1[i], 0, (n1 * n2 * n3 + lead_pad) * sizeof(float));
    cudaMemset(img_gpu[i], 0, (n1 * n2 * n3) * sizeof(float));
    //cudaMemset(src_p0[i], 0,
    //           (n1 * n2 * n3 + lead_pad) * sizeof(float));
    //cudaMemset(src_p1[i], 0,
    //           (n1 * n2 * n3 + lead_pad) * sizeof(float));
    //cudaMemset(src_p0[i], 0,
    //           (n1 * n2 * n3 + radius * n1 * n2 + lead_pad) * sizeof(float));
    //cudaMemset(src_p1[i], 0,
    //           (n1 * n2 * n3 + radius * n1 * n2 + lead_pad) * sizeof(float));

    //cudaMemcpy(src_p0[i] + lead_pad /*+radius*n1*n2*/,
    //           p0_s_cpu + i * n1 * n2 * (n3 - 2 * radius),
    //           n1 * n2 * (n3 /*-2*radius*/) * sizeof(float),
    //           cudaMemcpyHostToDevice);
    //cudaMemcpy(src_p1[i] + lead_pad /*+radius*n1*n2*/,
    //           p1_s_cpu + i * n1 * n2 * (n3 - 2 * radius),
    //           n1 * n2 * (n3 /*-2*radius*/) * sizeof(float),
    //           cudaMemcpyHostToDevice);
    // cudaMemcpy(src_p0[i]+lead_pad+radius*n1*n2,
    // p0_s_cpu/*+n1*n2*radius*/+i*n1*n2*(n3-2*radius),
    // n1*n2*(n3-2*radius)*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(src_p1[i]+lead_pad+radius*n1*n2,
    // p1_s_cpu/*+n1*n2*radius*/+i*n1*n2*(n3-2*radius),
    // n1*n2*(n3-2*radius)*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(src_p0[i]+lead_pad+radius*n1*n2,
    // p0_s_cpu/*+n1*n2*radius*/+i*n1*n2*(n3-radius),
    // n1*n2*(n3-radius)*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(src_p1[i]+lead_pad+radius*n1*n2,
    // p1_s_cpu/*+n1*n2*radius*/+i*n1*n2*(n3-radius),
    // n1*n2*(n3-radius)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dir_gpu, &sc, sizeof(float));
  }


  int nblocks1 = (n1 - 2 * FAT + BLOCKX_SIZE - 1) / BLOCKX_SIZE;
  int nblocks2 = (n2 - 2 * FAT + BLOCKY_SIZE - 1) / BLOCKY_SIZE;
  int last_x_block = nblocks1 - 1;
  int last_y_block = nblocks2 - 1;
  dim3 dimBlock(BLOCKX_SIZE, BLOCKY_SIZE);
  dim3 dimGrid(nblocks1, nblocks2);

  cudaStream_t stream_halo[n_gpus], stream_halo_data[n_gpus], stream_internal[n_gpus], stream_internal_data[n_gpus];
  cudaEvent_t start, stop;

  cudaSetDevice(device[0]);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int offset_internal[n_gpus];
  int start3[n_gpus], end3[n_gpus];

  for (int i = 0; i < n_gpus; i++) {
    cudaSetDevice(device[i]);
    cudaStreamCreate(&stream_halo[i]);
    cudaStreamCreate(&stream_halo_data[i]);
    cudaStreamCreate(&stream_internal[i]);
	cudaStreamCreate(&stream_internal_data[i]);

    offset_internal[i] = offset;
    if (i > 0) offset_internal[i] += n1 * n2 * radius;

    start3[i] = i * (n3 - 2 * radius) + 2 * radius;
    end3[i] = (i + 1) * (n3 - 2 * radius);
  }

  start3[0] = radius;
  end3[n_gpus - 1] = n_gpus * (n3 - 2 * radius) + radius;

  int offset_snd_h1 = lead_pad + n1 * n2 * radius;
  int offset_snd_h2 = lead_pad + n1 * n2 * (n3 - 2 * radius);
  int offset_rcv_h1 = lead_pad;
  int offset_rcv_h2 = lead_pad + n1 * n2 * (n3 - radius);
  int offset_cmp_h1 = offset_snd_h1 + n1 * radius + radius;
  int offset_cmp_h2 = offset_snd_h2 + n1 * radius + radius;

  cudaSetDevice(device[0]);
  cudaEventRecord(start, 0);

  int id_s = (nt + 1) / jt;
  int i_s = nt + 1 - id_s * jt;

  // main loop
  int data_counter = 0;
  int src_counter = 0;
  for (int it = nt - 1; it >= 0; it--) {
fprintf(stderr,"running adjoint %d\n",it);
    id_s = (it + 1) / _jtsS;
    i_s = it + 1 - id_s * _jtsS;
    int id = it / _jtdD;
    int ii = it - id * _jtdD;

    // halo region
    for (int i = 0; i < n_gpus; i++) {
      // TODO: stop recv when it == 1?
      cudaSetDevice(i);
      if (i > 0) {
	    if (it > 0)
          wave_kernel<<<dimGrid, dimBlock, 0, stream_halo_data[i]>>>(
              data_p0[i] + offset_cmp_h1, data_p1[i] + offset_cmp_h1,
              data_p0[i] + offset_cmp_h1, velocity2[i] + offset_cmp_h1, radius,
              radius * 2, last_x_block, last_y_block);

        if (it < nt - 1)
          wave_kernel<<<dimGrid, dimBlock, 0, stream_halo[i]>>>(
              src_p0[i] + offset_cmp_h1, src_p1[i] + offset_cmp_h1,
              src_p0[i] + offset_cmp_h1, velocity[i] + offset_cmp_h1, radius,
              radius * 2, last_x_block, last_y_block);
      }
      if (i < n_gpus - 1) {
        // TODO: validate
		if (it > 0)
          wave_kernel<<<dimGrid, dimBlock, 0, stream_halo_data[i]>>>(
              data_p0[i] + offset_cmp_h2, data_p1[i] + offset_cmp_h2,
              data_p0[i] + offset_cmp_h2, velocity2[i] + offset_cmp_h2,
              radius, radius * 2, last_x_block, last_y_block);

        if (it < nt - 1)
          wave_kernel<<<dimGrid, dimBlock, 0, stream_halo[i]>>>(
              src_p0[i] + offset_cmp_h2, src_p1[i] + offset_cmp_h2,
              src_p0[i] + offset_cmp_h2, velocity[i] + offset_cmp_h2,
              radius, radius * 2, last_x_block, last_y_block);  // THIS IS THE NAUGHTY ONE
        // wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(data_p0[i]+offset_cmp_h2,
        // data_p1[i]+offset_cmp_h2, data_p0[i]+offset_cmp_h2,
        // velocity[i]+offset_cmp_h2, radius, 2*radius);

        // if(it<nt-1)
        // wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(src_p0[i]+offset_cmp_h2,
        // src_p1[i]+offset_cmp_h2, src_p0[i]+offset_cmp_h2,
        // velocity[i]+offset_cmp_h2, radius, 2*radius); //THIS IS THE NAUGHTY
        // ONE
      }
      cudaStreamQuery(stream_halo[i]);
    }
    // damp_kernel<<<dimGrid, dimBlock>>>(nblocksz,data_p0,data_p1);

    // internal region
    for (int i = 0; i < n_gpus; i++) {
      cudaSetDevice(i);
      if (it > 0) {
        wave_kernel<<<dimGrid, dimBlock, 0, stream_internal_data[i]>>>(
            data_p0[i] + offset_internal[i], data_p1[i] + offset_internal[i],
            data_p0[i] + offset_internal[i], velocity2[i] + offset_internal[i],
            start3[i], end3[i], last_x_block, last_y_block);
      }

      if (it < (nt - 1)) {
        wave_kernel<<<dimGrid, dimBlock, 0, stream_internal[i]>>>(
            src_p0[i] + offset_internal[i], src_p1[i] + offset_internal[i],
            src_p0[i] + offset_internal[i], velocity[i] + offset_internal[i],
            start3[i], end3[i], last_x_block, last_y_block);
      }
    }

    for (int i = 1; i < n_gpus; i++) {
	  if (it > 0)
      cudaMemcpyPeerAsync(data_p0[i - 1] + offset_rcv_h2, i - 1,
                          data_p0[i] + offset_snd_h1, i,
                          n1 * n2 * radius * sizeof(float), stream_halo_data[i]);
      if (it < nt - 1)
        cudaMemcpyPeerAsync(src_p0[i - 1] + offset_rcv_h2, i - 1,
                            src_p0[i] + offset_snd_h1, i,
                            n1 * n2 * radius * sizeof(float), stream_halo[i]);
    }
    for (int i = 1; i < n_gpus; i++) {
      cudaSetDevice(i);
      cudaStreamSynchronize(stream_halo[i]);
	  cudaStreamSynchronize(stream_halo_data[i]);
    }
    for (int i = 0; i < n_gpus - 1; i++) {
	  if (it > 0)
      cudaMemcpyPeerAsync(data_p0[i + 1] + offset_rcv_h1, i + 1,
                          data_p0[i] + offset_snd_h2, i,
                          n1 * n2 * radius * sizeof(float), stream_halo_data[i]);
      if (it < nt - 1)
        cudaMemcpyPeerAsync(src_p0[i + 1] + offset_rcv_h1, i + 1,
                            src_p0[i] + offset_snd_h2, i,
                            n1 * n2 * radius * sizeof(float), stream_halo[i]);
    }

    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(stream_halo[i]);
        if (it < (nt - 1)) {
        if (i == shot_gpu) {
			if (id_s + 7 < ntsource_internal) {
              new_src_inject_kernel<<<1, npts_internal, 0, stream_internal[i]>>>(
                  id_s, i_s, src_p1[i] + lead_pad);
			  //fprintf(stderr, "id_s: %d, ntsource_internal: %d\n", id_s, ntsource_internal);
			  src_counter++;
			}
        }
	}
    }
    for (int i = 0; i < n_gpus; i++) {
      cudaSetDevice(i);
	  cudaStreamSynchronize(stream_halo_data[i]);
    }

	// damp after peer access
	if (n_gpus == 1)
	{
		int i = 0;
        	damp_kernel<<<dimGrid, dimBlock, 0, stream_internal_data[i]>>>(
          	data_p0[i] + offset_cmp_h1, data_p1[i] + offset_cmp_h1, start3[i], end3[i], i, n_gpus);
	} else {
		for (int i = 0; i < n_gpus - 1; i++)
		{
		  cudaSetDevice(i);
		  //cudaDeviceSynchronize(); // sync the halo & internal region computation
		  if (i == 0)
        		damp_kernel<<<dimGrid, dimBlock, 0, stream_internal_data[i]>>>(
        	  	data_p0[i] + offset_cmp_h1, data_p1[i] + offset_cmp_h1, start3[i], end3[i] + 2 * radius, i, n_gpus);
		  else if (i == n_gpus - 1)
        		damp_kernel<<<dimGrid, dimBlock, 0, stream_internal_data[i]>>>(
        	  	data_p0[i] + offset_rcv_h1 + radius + n1 * radius, data_p1[i] + offset_rcv_h1 + radius + n1 * radius, start3[i] - 2 * radius, end3[i], i, n_gpus);
		  else
        		damp_kernel<<<dimGrid, dimBlock, 0, stream_internal_data[i]>>>(
        	  	data_p0[i] + offset_rcv_h1 + radius + n1 * radius, data_p1[i] + offset_rcv_h1 + radius + n1 * radius, start3[i] - 2 * radius, end3[i] + 2 * radius, i, n_gpus);
		}
	}
    cudaSetDevice(device[0]);
	int rec_x_width = ((rec_nx + BLOCKX_SIZE - 1) / BLOCKX_SIZE) * BLOCKX_SIZE;
	dim3 dimGridDataInject((rec_nx + BLOCKX_SIZE - 1) / BLOCKX_SIZE,
			(rec_ny + BLOCKY_SIZE - 1) / BLOCKY_SIZE);
	if (id + 7 < ntreceiver_internal) {
      new_data_inject_kernel<<<dimGridDataInject, dimBlock, 0, stream_internal_data[0]>>>(
          id, ii, data_p0[0] + lead_pad/*+offset_snd_h1*/, rec_x_width);
	  data_counter++;
	}

      for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(device[i]);
		cudaStreamSynchronize(stream_internal_data[i]);
		cudaStreamSynchronize(stream_internal[i]);
		dim3 dimGridImg((n1 + 15)/ 16, (n2 + 15) / 16);
        img_kernel<<<dimGridImg, dimBlock, 0, stream_internal[i]>>>(
            img_gpu[i], data_p0[i] + lead_pad, src_p0[i] + lead_pad);
      }

    for (int i = 0; i < n_gpus; i++) {
      cudaSetDevice(i);
	  cudaStreamSynchronize(stream_internal_data[i]);
      cudaStreamSynchronize(stream_internal[i]);
      cudaStreamSynchronize(stream_halo_data[i]);
      cudaStreamSynchronize(stream_halo[i]);
      ptemp = src_p0[i];
      src_p0[i] = src_p1[i];
      src_p1[i] = ptemp;
      ptemp2 = data_p1[i];
      data_p1[i] = data_p0[i];
      data_p0[i] = ptemp2;
    }
  }

  fprintf(stderr, "Number of source injects: %d, number of data injects: %d\n", src_counter, data_counter);

  error = cudaGetLastError();
  process_error(error, "kernel");

  cudaSetDevice(device[0]);
  cudaEventRecord(stop, 0);

  cudaDeviceSynchronize();

  float time_total;
  cudaEventElapsedTime(&time_total, start, stop);
  fprintf(stderr, "Time for imaging = %f seconds \n", time_total / 1000);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  for (int i = 0; i < n_gpus; i++) {
    cudaSetDevice(i);
    fprintf(stderr, "CHECK2 %d %d %d \n", n1, n2, n3);
    cudaMemcpy(img + i * n1 * n2 * (n3 - 2 * radius), img_gpu[i] /*+radius*n1*n2*/,
        n1 * n2 * (n3 /*-2*radius*/) * sizeof(float), cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < n_gpus; i++) {
    cudaSetDevice(i);
    cudaFree(img_gpu[i]);
    cudaFree(data_p0[i]);
    cudaFree(data_p1[i]);
    cudaFree(src_p0[i]);
    cudaFree(src_p1[i]);
  }
  cudaFree(data_gpu);
  cudaFree(datageom_gpu);
}

void transfer_sinc_table_s(int nsinc, int ns, float *tables) {
  _jtsS = ns;
  cudaSetDevice(0);
  cudaMalloc((void **)&sincstable, ns * nsinc * sizeof(float));
  cudaMemcpy(sincstable, tables, ns * nsinc * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(sinc_s_table, &sincstable, sizeof(float *));
  cudaMemcpyToSymbol(nsinc_gpu, &nsinc, sizeof(int));
}

void transfer_sinc_table_d(int nsinc, int nd, float *tabled) {
  _jtdD = nd;
  cudaSetDevice(0);
  cudaMalloc((void **)&sincdtable, nd * nsinc * sizeof(float));
  cudaMemcpy(sincdtable, tabled, nd * nsinc * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(sinc_d_table, &sincdtable, sizeof(float *));
}

//void load_source(int it) {
//  int ibeg = it;
//  int iend = it + ntblock_internal + 7;
//  int ntb = ntblock_internal + 7;
//  if (iend > ntsource_internal) iend = ntsource_internal;
//  float *vals = (float *)malloc(ntb * sizeof(float) * npts_internal);
//  float sm = 0;
//  for (int ipt = 0; ipt < npts_internal; ipt++) {
//    memcpy(&vals[ipt * ntb], &source_buf[it + ntsource_internal * ipt],
//           (iend - ibeg) * sizeof(float));
//    for (int i = 0; i < iend - ibeg; i++) {
//      sm += fabs(vals[i + (iend - ibeg) * ipt]);
//    }
//    if (iend - ibeg < ntblock_internal + 7) {
//      for (int it = 0; it < ntblock_internal - 7 - (iend - ibeg); it++) {
//        vals[ipt * ntb + (iend - ibeg) + it] = 0.;
//      }
//    }
//  }
//  fprintf(stderr, "SM LOAD SOUrCE %f \n", sm);
//  cudaMemcpy(source_gpu, vals, ntb * npts_internal * sizeof(float),
//             cudaMemcpyHostToDevice);
//  free(vals);
//}

void set_ntblock(int nblock) {
  cudaSetDevice(device[0]);
  ntblock_internal = nblock;
  int nt = ntblock_internal + 7;
  cudaMemcpyToSymbol(ntblock_gpu, &nt, sizeof(int));
}

void transfer_source_func(int npts, int nt, int *locs, float *vals) {
  shot_gpu = 0;
  cudaSetDevice(device[0]);
  npts_internal = npts;
  ntsource_internal = nt;
  cudaMalloc((void **)&source_gpu, nt * npts * sizeof(float));
  cudaMalloc((void **)&srcgeom_gpu, npts * sizeof(int));
  fprintf(stderr, "TRANSFER 1 %d %d %d \n", nt, npts, sizeof(float));
  //cudaMemset(source_gpu, 0, (nt * npts) * sizeof(float));
  cudaMemcpy(srcgeom_gpu, locs, npts * sizeof(int), cudaMemcpyHostToDevice);
  //int imin, imax = imin = locs[0];
  //for (int i = 0; i < npts; i++) {
  //  if (locs[i] < imin) imin = locs[i];
  //  if (locs[i] > imax) imax = locs[i];
  //}
  //source_buf = (float *)malloc(sizeof(float) * nt * npts);
  //memcpy(source_buf, vals, nt * npts * sizeof(float));
  //float sm = 0;
  //fprintf(stderr, "CHECK MIN MAX %d %d \n", imin, imax);
  //for (int i = 0; i < nt * npts; i++) {
  //  sm += fabsf(source_buf[i]);
  //}
  //fprintf(stderr, "SM %f \n", sm);

  // srite("srccheck.H",vals,nt*npts*4);
  // fprintf(stderr,"%d %d \n",nt,npts);
  cudaMemcpy(source_gpu, vals, nt*npts*sizeof(float),
     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ntrace_gpu, &nt, sizeof(int));
  cudaMemcpyToSymbol(srcgeom_gpu0, &srcgeom_gpu, sizeof(int *));
  //cudaMemcpyToSymbol(npts_gpu, &npts, sizeof(int));

  cudaMemcpyToSymbol(source_gpu0, &source_gpu, sizeof(float *));
  //int ntt=ntblock_internal+7;
  //cudaMemcpyToSymbol(ntblock_gpu, &ntt, sizeof(int));
}
 
void transfer_receiver_func(int nx, int ny, int nt, int *locs, float *vals) {
  ntreceiver_internal = nt;
  cudaSetDevice(device[0]);
  cudaMalloc((void**) &data_gpu,nt*nx*ny*sizeof(float));
  //cudaMalloc((void **)&data_gpu,
  //           (7 + ntblock_internal) * nx * ny * sizeof(float));
  //cudaMemset(data_gpu, 0, (7 + ntblock_internal) * nx * ny * sizeof(float));
  cudaMalloc((void **)&datageom_gpu, nx * ny * sizeof(int));

  fprintf(stderr, "TRASN RECEIVER %d %d %d \n", nx, ny, nx * ny);
  cudaMemcpy(datageom_gpu, locs, nx * ny * sizeof(int),
             cudaMemcpyHostToDevice);
  //cudaMemcpy(data_gpu, vals,(7+ ntblock_internal)*nx*ny*sizeof(float),
  //   cudaMemcpyHostToDevice);
  cudaMemcpy(data_gpu, vals,(long long)nt*(long long)nx*(long long)ny*sizeof(float),
     cudaMemcpyHostToDevice);
  rec_nx = nx; rec_ny = ny;
  cudaMemcpyToSymbol(rec_nx_gpu, &nx, sizeof(int));
  cudaMemcpyToSymbol(rec_ny_gpu, &ny, sizeof(int));
  cudaMemcpyToSymbol(ntrace_gpu, &nt, sizeof(int));
  cudaMemcpyToSymbol(datageom_gpu0, &datageom_gpu, sizeof(int *));
  cudaMemcpyToSymbol(data_gpu0, &data_gpu, sizeof(float *));
  //int ntt=ntblock_internal+7;
  //cudaMemcpyToSymbol(ntblock_gpu, &ntt, sizeof(int));
}
void transfer_vel_func1(int n1, int n2, int n3, float *vel) {
  n3 = (n3 - 2 * radius) / n_gpus + 2 * radius;
  for (int i = 0; i < n_gpus; i++) {
    cudaSetDevice(device[i]);
    cudaMemcpy(velocity[i] +lead_pad,
	       vel + i * n1 * n2 * (n3 - 2 * radius),
               n1 * n2 * n3 * sizeof(float), cudaMemcpyHostToDevice);
  }
}
void transfer_vel_func2(int n1, int n2, int n3, float *vel) {
  n3 = (n3 - 2 * radius) / n_gpus + 2 * radius;
  for (int i = 0; i < n_gpus; i++) {
    cudaSetDevice(device[i]);
    cudaMemcpy(velocity2[i] +lead_pad,
               vel + i * n1 * n2 * (n3 - 2 * radius),
               n1 * n2 * n3 * sizeof(float), cudaMemcpyHostToDevice);
  }
}

void create_gpu_space(float d1, float d2, float d3, float bc_a, float bc_b,
                      float bc_b_y, int n1, int n2, int n3) {
  //lead_pad = 0;  // 32-radius;
  lead_pad = 32-radius;
  n3 = (n3 - 2 * radius) / n_gpus + 2 * radius;
  offset = radius * n1 * n2 + radius * n1 + radius + lead_pad;

  float coeffs_cpu[COEFFS_SIZE] =
      get_coeffs((float)d1, (float)d2, (float)d3);
  fprintf(stderr, "coeffs %f %f %f %f\n", coeffs_cpu[0], coeffs_cpu[4], coeffs_cpu[5], coeffs_cpu[6]);

  std::vector<float> bound_cpu = compute_bound();

  for (int i = 0; i < n_gpus; i++) {
    cudaSetDevice(device[i]);
    cudaMalloc((void **)&velocity[i],
               (n1 * n2 * n3 +lead_pad) * sizeof(float));
    cudaMalloc((void **)&velocity2[i],
               (n1 * n2 * n3 +lead_pad) * sizeof(float));
    fprintf(stderr, "CHECK N1GPU %d \n", n1);
    cudaMemcpyToSymbol(n1gpu, &n1, sizeof(int));
    cudaMemcpyToSymbol(n2gpu, &n2, sizeof(int));
    cudaMemcpyToSymbol(n3gpu, &n3, sizeof(int));

    cudaMemcpyToSymbol(bc_agpu, &bc_a, sizeof(float));
    cudaMemcpyToSymbol(bc_bgpu, &bc_b, sizeof(float));
    cudaMemcpyToSymbol(coeffs, coeffs_cpu, COEFFS_SIZE * sizeof(float), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(bound, bound_cpu.data(), 40 * sizeof(float), 0,
                       cudaMemcpyHostToDevice);
  }
}
