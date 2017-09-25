//#include "cutil_inline.h"
//#include "cutil_math.h"
#include "gpu_finite_3d.h"
#include <stdlib.h>
#include "sep3d.h"
#include "seplib.h"  
#include "cudaErrors.cu"
#include "wave_fkernel.3d8o.cu"

void setup_cuda(int ngpus, int argc, char **argv){
  n_gpus=ngpus;
  fprintf(stderr,"Today, we are using %d GPUs; specifically: \n",n_gpus);
  int dr;

  for(int i=0; i<n_gpus; i++) device[i]=i;

  for(int i=0; i<n_gpus; i++){
    cudaDeviceSynchronize();

    cudaSetDevice(device[i]);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,device[i]);
    cudaDriverGetVersion(&dr);

    fprintf(stderr,"  GPU %s (%d),",properties.name, device[i]);
    if(properties.unifiedAddressing) fprintf(stderr," UVA initiated,");
    else fprintf(stderr," UVA not working ");
    fprintf(stderr," driver %d\n",dr);

    //Enable P2P memcopies between GPUs
    if(n_gpus > 1){
      for(int j=0; j<n_gpus; j++){
        if(i==j) continue;
        int peer_access_available=0;
        cudaDeviceCanAccessPeer( &peer_access_available,device[i],device[j]);
        if(peer_access_available){
	  //fprintf(stderr,"Make the GPUs talk %d %d\n",device[i],device[j]);
          cudaDeviceEnablePeerAccess(device[j],0);
        }
      }
    }

  }

}

void process_error( const cudaError_t &error, char *string=0, bool verbose=false )
{
    if( error != cudaSuccess || verbose )
    {
        int current_gpu = -1;
        cudaGetDevice( &current_gpu );

        fprintf(stderr, "GPU %d: ", current_gpu );
        if( string )
            printf( string );
        fprintf(stderr, ": %s\n", cudaGetErrorString( error ) );
    }

    if( error != cudaSuccess )
        exit(-1);
}

extern "C" __global__ void new_src_inject_kernel(int it, int isinc,float *p){
  int ix=blockIdx.x*blockDim.x+threadIdx.x;
  p[srcgeom_gpu0[ix]]+=dir_gpu*(
  sinc_s_table[isinc*nsinc_gpu]*  source_gpu0[ntrace_gpu*ix+it]+
  sinc_s_table[isinc*nsinc_gpu+1]*source_gpu0[ntrace_gpu*ix+it+1]+
  sinc_s_table[isinc*nsinc_gpu+2]*source_gpu0[ntrace_gpu*ix+it+2]+
  sinc_s_table[isinc*nsinc_gpu+3]*source_gpu0[ntrace_gpu*ix+it+3]+
  sinc_s_table[isinc*nsinc_gpu+4]*source_gpu0[ntrace_gpu*ix+it+4]+
  sinc_s_table[isinc*nsinc_gpu+5]*source_gpu0[ntrace_gpu*ix+it+5]+
  sinc_s_table[isinc*nsinc_gpu+6]*source_gpu0[ntrace_gpu*ix+it+6]+
  sinc_s_table[isinc*nsinc_gpu+7]*source_gpu0[ntrace_gpu*ix+it+7]);
}

extern "C" __global__ void new_data_inject_kernel(int it, int isinc,float *p){
  int j=blockIdx.y*blockDim.y+threadIdx.y;
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=k+n1gpu*j;
  if(i< rec_nx_gpu*rec_ny_gpu){
    p[datageom_gpu0[i]]+=
    sinc_d_table[isinc*nsinc_gpu]*data_gpu0[ntrace_gpu*i+it] +
    sinc_d_table[isinc*nsinc_gpu+1]*data_gpu0[ntrace_gpu*i+it+1]+
    sinc_d_table[isinc*nsinc_gpu+2]*data_gpu0[ntrace_gpu*i+it+2]+
    sinc_d_table[isinc*nsinc_gpu+3]*data_gpu0[ntrace_gpu*i+it+3]+
    sinc_d_table[isinc*nsinc_gpu+4]*data_gpu0[ntrace_gpu*i+it+4]+
    sinc_d_table[isinc*nsinc_gpu+5]*data_gpu0[ntrace_gpu*i+it+5]+
    sinc_d_table[isinc*nsinc_gpu+6]*data_gpu0[ntrace_gpu*i+it+6]+
    sinc_d_table[isinc*nsinc_gpu+7]*data_gpu0[ntrace_gpu*i+it+7];
  }

}

extern "C" __global__ void new_data_extract_kernel(int it, int isinc,float *p){
  long long j=blockIdx.y*blockDim.y+threadIdx.y;
  long long k=blockIdx.x*blockDim.x+threadIdx.x;
  long long i=k+(n1gpu)*j;
  long nt=ntrace_gpu;
  if(i< (rec_nx_gpu*rec_ny_gpu)){
    data_gpu0[nt*i+it+0]+=p[datageom_gpu0[i]]*sinc_d_table[isinc*nsinc_gpu+0];
    data_gpu0[nt*i+it+1]+=p[datageom_gpu0[i]]*sinc_d_table[isinc*nsinc_gpu+1];
    data_gpu0[nt*i+it+2]+=p[datageom_gpu0[i]]*sinc_d_table[isinc*nsinc_gpu+2];
    data_gpu0[nt*i+it+3]+=p[datageom_gpu0[i]]*sinc_d_table[isinc*nsinc_gpu+3];
    data_gpu0[nt*i+it+4]+=p[datageom_gpu0[i]]*sinc_d_table[isinc*nsinc_gpu+4];
    data_gpu0[nt*i+it+5]+=p[datageom_gpu0[i]]*sinc_d_table[isinc*nsinc_gpu+5];
    data_gpu0[nt*i+it+6]+=p[datageom_gpu0[i]]*sinc_d_table[isinc*nsinc_gpu+6];
    data_gpu0[nt*i+it+7]+=p[datageom_gpu0[i]]*sinc_d_table[isinc*nsinc_gpu+7];
  }
}

extern "C" __global__ void img_kernel( float* img, float*dat, float*src){
  int ig = blockIdx.x * blockDim.x + threadIdx.x;
  int jg = blockIdx.y * blockDim.y + threadIdx.y;
  int addr= ig + n1gpu * jg;
  int stride = n1gpu*n2gpu;
  for(int iy=0; iy<n3gpu; iy++){
    img[addr]+=.000001*dat[addr]*src[addr];
    addr+=stride;
  }
}

extern "C" __global__ void img_add_kernel( float* img, float*rec_field, float *src_field){//as above, added 
 long long ig = blockIdx.x * blockDim.x + threadIdx.x;
  long long jg = blockIdx.y * blockDim.y + threadIdx.y;
  long long addr =  ig + (long long) n1gpu * jg;
  long long  stride = (long long) n1gpu*(long long)n2gpu;
  for(long long iy=0; iy<n3gpu; iy++){
    rec_field[addr]+=.000001*src_field[addr]*(img[addr]);
    addr+=stride;
  }
}

void source_prop(int n1, int n2, int n3, bool damp, bool get_last, float *p0, float *p1, int jt, int npts,int nt){

  //Propagate the source wavefield and return the final two 3D wavefield slices

  float *ptemp;
  float *src_p0[n_gpus],*src_p1[n_gpus];

  cudaError_t error = cudaSuccess;

  //int n3_total=n3;
  n3=(n3-2*radius)/n_gpus + 2*radius;
  //int dim3=n3;
  //if(n_gpus > 1) dim3-=2*radius;

  int dir=1;

  int n_bytes_gpu=(n1*n2*n3+lead_pad)*sizeof(float);

  for(int i=0; i<n_gpus; i++){

    cudaSetDevice(device[i]);
    //cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    cudaMalloc((void**) &src_p0[i],n_bytes_gpu);
    cudaMalloc((void**) &src_p1[i],n_bytes_gpu);

    cudaMemset(src_p0[i], 0,n_bytes_gpu);
    cudaMemset(src_p1[i], 0,n_bytes_gpu);

    cudaMemcpyToSymbol(dir_gpu, &dir, sizeof(float));

  }

//fprintf(stderr,"Allocate %d %d %d, %f mbs; %d\n",n1,n2,n3,(float)(n1*n2*n3*4/1000000),lead_pad);

  //Blocks for internal data
  //int nblocks1=(n1-2*FAT)/(2*BLOCKX_SIZE);
  int nblocks1=(n1-2*FAT)/BLOCKX_SIZE;
  int nblocks2=(n2-2*FAT)/BLOCKY_SIZE;

  dim3 dimBlock(BLOCKX_SIZE, BLOCKY_SIZE);
  dim3 dimGrid(nblocks1,nblocks2);

  //Define separate streams for overlapping communication
  cudaStream_t stream_halo[n_gpus], stream_internal[n_gpus];
  cudaEvent_t start,stop;

  cudaSetDevice(device[0]);
  cudaEventCreate(&start);
  cudaEventCreate(&stop); 

  //Setup coordinate systems for internal domains
  int offset_internal[n_gpus];
  int start3[n_gpus],end3[n_gpus];

  for(int i=0; i<n_gpus; i++){
    cudaSetDevice(device[i]);
    cudaStreamCreate(&stream_halo[i]);
    cudaStreamCreate(&stream_internal[i]);

    //Offset_internal is the initial index of our internal domain (out of stencil padding)
    offset_internal[i]=offset;
    if(i > 0) offset_internal[i] += n1*n2*radius;

    start3[i] = i*(n3-2*radius) + 2*radius;
    end3[i] = (i+1)*(n3-2*radius) /*- radius*/;
    //start3[i] = i*(n3-2*radius) + radius;
    //end3[i] = (i+1)*(n3-2*radius) - radius;
  }

  start3[0]=radius;
  end3[n_gpus-1]=n_gpus*(n3-2*radius);//I THINK THIS SHOULD BE -RADIUS. LET'S TRY
  //start3[0]=0;
  //end3[n_gpus-1]=n_gpus*(n3-2*radius);

  //Set up coordinate systems for the halo exchange
  int offset_snd_h1=lead_pad+n1*n2*radius;
  int offset_snd_h2=lead_pad+n1*n2*(n3-2*radius);
  int offset_rcv_h1=lead_pad;
  int offset_rcv_h2=lead_pad+n1*n2*(n3-2*radius+radius);
  int offset_cmp_h1=offset;
  long int offset_cmp_h2=lead_pad+radius+radius*n1+n1*n2*(n3-2*radius);//-radius?

  /*int offset_snd_h1=lead_pad+n1*n2*radius;
  int offset_snd_h2=lead_pad+n1*n2*(n3-2*radius);
  int offset_rcv_h1=lead_pad;
  int offset_rcv_h2=lead_pad+n1*n2*(n3-2*radius+radius);
  int offset_cmp_h1=offset;
  long int offset_cmp_h2=lead_pad+radius+radius*n1+radius*n1*n2+n1*n2*(n3-2*radius-radius);*/


  cudaSetDevice(device[0]);
  cudaEventRecord(start,0);

  for(int it=0; it<=nt; it++){
    int id=it/jt;
    int ii=it-id*jt;

    // Calculate the halo regions first
    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(i);

      if(i>0){
        wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(src_p0[i]+offset_cmp_h1, src_p1[i]+offset_cmp_h1, src_p0[i]+offset_cmp_h1, velocity[i]+offset_cmp_h1, radius, 2*radius);
      }

      if(i<n_gpus-1){
        wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(src_p0[i]+offset_cmp_h2, src_p1[i]+offset_cmp_h2, src_p0[i]+offset_cmp_h2, velocity[i]+offset_cmp_h2, (n3-radius)-radius,n3-radius);
      }

      cudaStreamQuery(stream_halo[i]);
    }

    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(device[i]);

      wave_kernel<<<dimGrid, dimBlock,0,stream_internal[i]>>>(src_p0[i]+offset_internal[i], src_p1[i]+offset_internal[i], src_p0[i]+offset_internal[i], velocity[i]+offset_internal[i], start3[i], end3[i]);
      if(i==shot_gpu) new_src_inject_kernel<<<1,npts,0,stream_internal[i]>>>(id ,ii, src_p0[i]+lead_pad);
    }

    //Overlap internal computation with halo communication

    //Send halos to the 'right'
    for(int i=0; i<n_gpus-1; i++){
      cudaMemcpyPeerAsync(src_p0[i+1]+offset_rcv_h1,i+1,src_p0[i]+offset_snd_h2,i,n1*n2*radius*sizeof(float),stream_halo[i]);
    }

    //Synchronize to avoid stalling
    for(int i=0; i<n_gpus-1; i++){
      cudaSetDevice(i);
      cudaStreamSynchronize(stream_halo[i]);
    }

    //Send halos to the 'left'
    for(int i=1; i<n_gpus; i++){
      cudaMemcpyPeerAsync(src_p0[i-1]+offset_rcv_h2,i-1,src_p0[i]+offset_snd_h1,i,n1*n2*radius*sizeof(float),stream_halo[i]);
    }

    //Synchronise GPUs and do pointer exchange
    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(device[i]);
      cudaDeviceSynchronize();
      ptemp=src_p0[i]; src_p0[i]=src_p1[i];src_p1[i]=ptemp;
    }
  }

  error = cudaGetLastError();
  process_error( error, "kernel" );

  //Use device 0 to give a performance report
  cudaSetDevice(device[0]);
  cudaEventRecord(stop,0);

  cudaDeviceSynchronize();

  float time_total;
  cudaEventElapsedTime(&time_total,start,stop);
  fprintf(stderr,"Time for source propagation = %f seconds \n",time_total/1000);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if(get_last){
    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(device[i]);
      cudaMemcpy(p0+i*n1*n2*(n3-2*radius), src_p0[i]+lead_pad/*+radius*n1*n2*/, n1*n2*(n3/*-2*radius*/)*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(p1+i*n1*n2*(n3-2*radius), src_p1[i]+lead_pad/*+radius*n1*n2*/, n1*n2*(n3/*-2*radius*/)*sizeof(float), cudaMemcpyDeviceToHost);
      //cudaMemcpy(p0+i*n1*n2*(n3-radius), src_p0[i]+radius*n1*n2, n1*n2*(n3-radius)*sizeof(float), cudaMemcpyDeviceToHost);
      //cudaMemcpy(p1+i*n1*n2*(n3-radius), src_p1[i]+radius*n1*n2, n1*n2*(n3-radius)*sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(src_p0[i]+lead_pad/*+radius*n1*n2*/, p0_s_cpu+i*n1*n2*(n3-2*radius), n1*n2*(n3/*-2*radius*/)*sizeof(float),
    }
  }


  for(int i=0; i<n_gpus; i++){
    cudaSetDevice(device[i]);
    cudaFree(src_p0[i]);
    cudaFree(src_p1[i]);
  }
  cudaFree(source_gpu);
  cudaFree(srcgeom_gpu);

}

void rtm_forward(int n1, int n2, int n3, int jt, float *img, float *dat, int npts_src, int nt,int nt_big, int rec_nx, int rec_ny){

fprintf(stderr,"in born modeling 1\n");
  //Born modelling over input image
  float *src_p0[n_gpus], *src_p1[n_gpus], *data_p0[n_gpus], *data_p1[n_gpus], *img_gpu[n_gpus];
  float *ptemp, *ptemp2;

  cudaError_t error = cudaSuccess;

  int n3_total=n3;
  n3=(n3-2*radius)/n_gpus + 2*radius;

  int dir=1;


  int nblocks1=(n1-2*FAT)/BLOCKZ_SIZE; 
  int nblocks2=(n2-2*FAT)/BLOCKX_SIZE; 
  //int nblocks3=(n3-2*FAT)/BLOCKY_SIZE; 

  dim3 dimGrid(nblocks1,nblocks2);
  dim3 dimBlock(16, 16);

  //dim3 dimGridx((int)ceilf(1.*n1/BLOCKX_SIZE),(int)ceilf(1.*n2/BLOCKY_SIZE));
  dim3 dimGridx((int)ceilf(1.*n1/BLOCKX_SIZE),(int)ceilf(1.*n2/BLOCKY_SIZE));

  cudaStream_t stream_halo[n_gpus], stream_internal[n_gpus];
  cudaEvent_t start,stop;

  cudaSetDevice(device[0]);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int offset_internal[n_gpus];
  int start3[n_gpus],end3[n_gpus];

fprintf(stderr,"in born modeling 2\n");
  for(int i=0; i<n_gpus; i++){
    cudaSetDevice(device[i]);
    cudaStreamCreate(&stream_halo[i]);
    cudaStreamCreate(&stream_internal[i]);

    offset_internal[i]=offset;
    if(i > 0) offset_internal[i] += n1*n2*radius;

    start3[i] = i*(n3-2*radius) + 2*radius;
    end3[i] = (i+1)*(n3-2*radius);
  }
fprintf(stderr,"in born modeling 3\n");

  start3[0]=radius;
  end3[n_gpus-1]=n_gpus*(n3-2*radius);

   fprintf(stderr,"CHECK THIS %d %d %d %d \n",n1,n2,n3,lead_pad);
  for(int i=0; i<n_gpus; i++){

    cudaSetDevice(device[i]);
    cudaMalloc((void**) &src_p0[i], (n1*n2*n3+lead_pad)*sizeof(float));
    cudaMalloc((void**) &src_p1[i], (n1*n2*n3+lead_pad)*sizeof(float));
    cudaMalloc((void**) &data_p0[i],(n1*n2*n3+lead_pad)*sizeof(float));
    cudaMalloc((void**) &data_p1[i],(n1*n2*n3+lead_pad)*sizeof(float));
    cudaMalloc((void**) &img_gpu[i], n1*n2*n3*sizeof(float));

    cudaMemset(data_p0[i], 0,(n1*n2*n3+lead_pad)*sizeof(float));
    cudaMemset(data_p1[i], 0,(n1*n2*n3+lead_pad)*sizeof(float));
    cudaMemset(src_p0[i],  0,(n1*n2*n3+lead_pad)*sizeof(float));
    cudaMemset(src_p1[i],  0,(n1*n2*n3+lead_pad)*sizeof(float));
    cudaMemset(img_gpu[i], 0,n1*n2*n3*sizeof(float));

    cudaMemcpy( img_gpu[i]/*+radius*n1*n2*/, img+i*n1*n2*(n3-2*radius), n1*n2*(n3/*-2*radius*/)*sizeof(float), cudaMemcpyHostToDevice);
    //DONT DELETE need the -2*r gone to be multi-gpu invariant
    //cudaMemcpy( img_gpu[i]+radius*n1*n2, img+i*n1*n2*(n3-2*radius), n1*n2*(n3-2*radius)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dir_gpu, &dir, sizeof(float));
  }

  int offset_snd_h1=lead_pad+n1*n2*radius;
  int offset_snd_h2=lead_pad+n1*n2*(n3-2*radius);
  int offset_rcv_h1=lead_pad+0;
  int offset_rcv_h2=lead_pad+n1*n2*(n3-2*radius+radius);

  int offset_cmp_h1=offset;
  int offset_cmp_h2=lead_pad+radius+radius*n1+n1*n2*(n3-radius-radius);

  cudaSetDevice(device[0]);
  cudaEventRecord(start,0);

  for(int it=0; it < nt; it++){
    int id=it/jt;
    int ii=it-id*jt;
    if(it%50==0) fprintf(stderr,"in modeling %d of %d \n",it,nt);

//    kernel_exec(damp_kernel<<<dimGrid, dimBlock>>>(nblocksz,data_p0, data_p1));
    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(device[i]);
      if(i>0){
        wave_kernel_adj<<<dimGrid,dimBlock,0,stream_halo[i]>>>(data_p0[i]+offset_cmp_h1, data_p1[i]+offset_cmp_h1, data_p0[i]+offset_cmp_h1, velocity[i]+offset_cmp_h1, radius, 2*radius);
        wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(src_p0[i]+offset_cmp_h1, src_p1[i]+offset_cmp_h1, src_p0[i]+offset_cmp_h1, velocity[i]+offset_cmp_h1, radius, 2*radius);
      } 
      if(i<n_gpus-1){
        wave_kernel_adj<<<dimGrid,dimBlock,0,stream_halo[i]>>>(data_p0[i]+offset_cmp_h2, data_p1[i]+offset_cmp_h2, data_p0[i]+offset_cmp_h2, velocity[i]+offset_cmp_h2, (n3-radius)-radius, n3-radius);
        wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(src_p0[i]+offset_cmp_h2, src_p1[i]+offset_cmp_h2, src_p0[i]+offset_cmp_h2, velocity[i]+offset_cmp_h2, (n3-radius)-radius, n3-radius);
      }
      cudaStreamQuery(stream_halo[i]);
    }

    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(device[i]);

      damp_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(data_p0[i], data_p1[i], start3[i], end3[i], i, n_gpus);
      damp_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(data_p0[i], data_p1[i], start3[i], end3[i], i, n_gpus);
      damp_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(src_p0[i], src_p1[i], start3[i], end3[i], i, n_gpus);

      wave_kernel_adj<<<dimGrid,dimBlock,0,stream_internal[i]>>>(data_p0[i]+offset_internal[i], data_p1[i]+offset_internal[i], data_p0[i]+offset_internal[i], velocity[i]+offset_internal[i], start3[i], end3[i]);

      wave_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(src_p0[i]+offset_internal[i], src_p1[i]+offset_internal[i], src_p0[i]+offset_internal[i], velocity[i]+offset_internal[i], start3[i], end3[i]);


      if(i==shot_gpu) new_src_inject_kernel<<<1,npts_src,0,stream_internal[i]>>>(id ,ii,src_p0[i]+lead_pad);
    }

    for(int i=0; i<n_gpus-1; i++){
      cudaMemcpyPeerAsync(data_p0[i+1]+offset_rcv_h1,i+1,data_p0[i]+offset_snd_h2,i,n1*n2*radius*sizeof(float),stream_halo[i]);
      cudaMemcpyPeerAsync(src_p0[i+1]+offset_rcv_h1,i+1,src_p0[i]+offset_snd_h2,i,n1*n2*radius*sizeof(float),stream_halo[i]);
    }
    for(int i=0; i<n_gpus-1; i++){
      cudaSetDevice(device[i]);
      cudaStreamSynchronize(stream_halo[i]);
    }
    for(int i=1; i<n_gpus; i++){
      cudaMemcpyPeerAsync(data_p0[i-1]+offset_rcv_h2,i-1,data_p0[i]+offset_snd_h1,i,n1*n2*radius*sizeof(float),stream_halo[i]);
      cudaMemcpyPeerAsync(src_p0[i-1]+offset_rcv_h2,i-1,src_p0[i]+offset_snd_h1,i,n1*n2*radius*sizeof(float),stream_halo[i]);
    }


    for(int i=0; i<n_gpus; i++){ 
      cudaSetDevice(device[i]);
      cudaSetDevice(device[i]);
      cudaStreamSynchronize(stream_internal[i]);
    }
    
 
    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(device[i]);
      //if(it%jt==0) img_add_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(img_gpu[i]+offset_snd_h1,data_p0[i]+offset_snd_h1,src_p0[i]+offset_snd_h1);
      if(it%jt==0) img_add_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(img_gpu[i],data_p0[i],src_p0[i]);
      //if(it%jt==0) img_add_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(img_gpu[i]+offset_internal[i]-offset,data_p0[i]+offset_internal[i]-offset,src_p0[i]+offset_internal[i]-offset); //works for multis
    }


    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(device[i]);
      cudaStreamSynchronize(stream_internal[i]);
      cudaDeviceSynchronize();
    }
    cudaSetDevice(device[0]);
    new_data_extract_kernel<<<dimGridx,dimBlock,0,stream_internal[0]>>>(id ,ii, data_p0[0]/*+offset_snd_h1*/);


    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(i);
      cudaDeviceSynchronize();
      ptemp2=src_p1[i]; src_p1[i]=src_p0[i]; src_p0[i]=ptemp2;
      ptemp=data_p1[i]; data_p1[i]=data_p0[i]; data_p0[i]=ptemp;
    }
   

  }
  //error = cudaGetLastError();
  //process_error( error, "kernel" );

  cudaSetDevice(device[0]);
  cudaEventRecord(stop,0);

  cudaDeviceSynchronize();

  float time_total;
  cudaEventElapsedTime(&time_total,start,stop);
  fprintf(stderr,"Time for Born modelling = %f seconds \n",time_total/1000.);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaSetDevice(device[0]);
  cudaMemcpy(dat, data_gpu, nt_big*rec_nx*rec_ny*sizeof(float), cudaMemcpyDeviceToHost);

  for(int i=0; i<n_gpus; i++){
    cudaSetDevice(i);
    cudaFree(data_p0[i]);
    cudaFree(img_gpu[i]);
    cudaFree(data_p1[i]);
    cudaFree(src_p0[i]);
    cudaFree(src_p1[i]);
  }
  cudaSetDevice(0);
  cudaFree(data_gpu);
  cudaFree(datageom_gpu);
  cudaFree(source_gpu);
  cudaFree(srcgeom_gpu);

}



void rtm_adjoint(int n1, int n2, int n3, int jt, float *p0_s_cpu, float *p1_s_cpu, float *img, int npts_src,int nt){
 
  float *src_p0[n_gpus],*src_p1[n_gpus],*data_p0[n_gpus],*data_p1[n_gpus],*img_gpu[n_gpus];
  float *ptemp,*ptemp2;

  cudaError_t error = cudaSuccess;

  int n3_total=n3;
  n3=(n3-2*radius)/n_gpus + 2*radius;

  int dir=-1;

  for(int i=0; i<n_gpus; i++){

    cudaSetDevice(device[i]);
    cudaMalloc((void**) &src_p0[i],  (n1*n2*n3+radius*n1*n2+lead_pad)*sizeof(float));
    cudaMalloc((void**) &src_p1[i],  (n1*n2*n3+radius*n1*n2+lead_pad)*sizeof(float));
    cudaMalloc((void**) &data_p0[i], (n1*n2*n3+lead_pad)*sizeof(float));
    cudaMalloc((void**) &data_p1[i], (n1*n2*n3+lead_pad)*sizeof(float));

    cudaMalloc((void**) &img_gpu[i], n1*n2*n3*sizeof(float));

    cudaMemset(data_p0[i], 0,(n1*n2*n3+lead_pad)*sizeof(float));
    cudaMemset(data_p1[i], 0,(n1*n2*n3+lead_pad)*sizeof(float));
    cudaMemset(img_gpu[i], 0,(n1*n2*n3+lead_pad)*sizeof(float));
    cudaMemset(src_p0[i], 0,(n1*n2*n3+radius*n1*n2+lead_pad)*sizeof(float));
    cudaMemset(src_p1[i], 0,(n1*n2*n3+radius*n1*n2+lead_pad)*sizeof(float));

    cudaMemcpy(src_p0[i]+lead_pad/*+radius*n1*n2*/, p0_s_cpu+i*n1*n2*(n3-2*radius), n1*n2*(n3/*-2*radius*/)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(src_p1[i]+lead_pad/*+radius*n1*n2*/, p1_s_cpu+i*n1*n2*(n3-2*radius), n1*n2*(n3/*-2*radius*/)*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(src_p0[i]+lead_pad+radius*n1*n2, p0_s_cpu/*+n1*n2*radius*/+i*n1*n2*(n3-2*radius), n1*n2*(n3-2*radius)*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(src_p1[i]+lead_pad+radius*n1*n2, p1_s_cpu/*+n1*n2*radius*/+i*n1*n2*(n3-2*radius), n1*n2*(n3-2*radius)*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(src_p0[i]+lead_pad+radius*n1*n2, p0_s_cpu/*+n1*n2*radius*/+i*n1*n2*(n3-radius), n1*n2*(n3-radius)*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(src_p1[i]+lead_pad+radius*n1*n2, p1_s_cpu/*+n1*n2*radius*/+i*n1*n2*(n3-radius), n1*n2*(n3-radius)*sizeof(float), cudaMemcpyHostToDevice);
   }


  cudaMemcpyToSymbol(dir_gpu, &dir, sizeof(float));

  int nblocks1=(n1-2*FAT)/BLOCKZ_SIZE;
  int nblocks2=(n2-2*FAT)/BLOCKX_SIZE;
  //int nblocks3=(n3-2*FAT)/BLOCKY_SIZE;

  dim3 dimGrid(nblocks1,nblocks2);
  dim3 dimBlock(16, 16);

  dim3 dimGridx((int)ceilf(1.*n1/BLOCKX_SIZE),(int)ceilf(1.*n2/BLOCKY_SIZE));

  cudaStream_t stream_halo[n_gpus], stream_internal[n_gpus];
  cudaEvent_t start,stop;

  cudaSetDevice(device[0]);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int offset_internal[n_gpus];
  int start3[n_gpus],end3[n_gpus];

  for(int i=0; i<n_gpus; i++){
    cudaSetDevice(device[i]);
    cudaStreamCreate(&stream_halo[i]);
    cudaStreamCreate(&stream_internal[i]);

    offset_internal[i]=offset;  //  offset=radius*n1*n2+radius*n1+radius+lead_pad;
    if(i > 0) offset_internal[i] += n1*n2*radius;

    start3[i] = i*(n3-2*radius) + 2*radius;
    end3[i] = (i+1)*(n3-2*radius);
  }

  start3[0]=radius;
  end3[n_gpus-1]=n_gpus*(n3-2*radius);

  int offset_snd_h1=lead_pad+n1*n2*radius;
  int offset_snd_h2=lead_pad+n1*n2*(n3-2*radius);
  int offset_rcv_h1=lead_pad;
  int offset_rcv_h2=lead_pad+n1*n2*(n3-2*radius+radius)/*+radius+radius*n1*n2*/;
  int offset_cmp_h1=offset;
  int offset_cmp_h2=lead_pad+radius+radius*n1+n1*n2*(n3-radius-radius);

  cudaSetDevice(device[0]);
  cudaEventRecord(start,0);

  int id_s=(nt+1)/jt;
  int i_s=nt+1-id_s*jt;

  //new_src_inject_kernel<<<1,npts_src>>>(id_s,i_s,src_p0[0]+lead_pad);

  //float *snap;
  //snap=(float*)malloc(4*n1*n2*n3_total);

  for(int it=nt-1; it >=0 ;it--){
    id_s=(it+1)/jt;
    i_s=it+1-id_s*jt;
    int id=it/jt;
    int ii=it-id*jt;

    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(i);
      if(i>0){
       wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(data_p0[i]+offset_cmp_h1, data_p1[i]+offset_cmp_h1, data_p0[i]+offset_cmp_h1, velocity[i]+offset_cmp_h1, radius, 2*radius);

      if(it<nt-1) wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(src_p0[i]+offset_cmp_h1, src_p1[i]+offset_cmp_h1, src_p0[i]+offset_cmp_h1, velocity[i]+offset_cmp_h1, radius, 2*radius);
      }
      if(i<n_gpus-1){
        wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(data_p0[i]+offset_cmp_h2, data_p1[i]+offset_cmp_h2, data_p0[i]+offset_cmp_h2, velocity[i]+offset_cmp_h2, radius, 2*radius);

        if(it<nt-1) wave_kernel<<<dimGrid,dimBlock,0,stream_halo[i]>>>(src_p0[i]+offset_cmp_h2, src_p1[i]+offset_cmp_h2, src_p0[i]+offset_cmp_h2, velocity[i]+offset_cmp_h2, radius, 2*radius); //THIS IS THE NAUGHTY ONE
      }
      cudaStreamQuery(stream_halo[i]);
    }
    //damp_kernel<<<dimGrid, dimBlock>>>(nblocksz,data_p0,data_p1);

    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(i);
      damp_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(data_p0[i], data_p1[i], start3[i], end3[i], i, n_gpus);

      wave_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(data_p0[i]+offset_internal[i], data_p1[i]+offset_internal[i], data_p0[i]+offset_internal[i], velocity[i]+offset_internal[i], start3[i], end3[i]);

      if(it<(nt-1)){

        wave_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(src_p0[i]+offset_internal[i], src_p1[i]+offset_internal[i], src_p0[i]+offset_internal[i], velocity[i]+offset_internal[i], start3[i], end3[i]);

        if(i==shot_gpu) new_src_inject_kernel<<<1,npts_src,0,stream_internal[i]>>>(id_s,i_s, src_p1[i]+lead_pad); //p1??
      }
    }


    for(int i=1; i<n_gpus; i++){
      cudaMemcpyPeerAsync(data_p0[i-1]+offset_rcv_h2,i-1,data_p0[i]+offset_snd_h1,i,n1*n2*radius*sizeof(float),stream_halo[i]);
      if(it<nt-1) cudaMemcpyPeerAsync(src_p0[i-1] +offset_rcv_h2,i-1,src_p0[i] +offset_snd_h1,i,n1*n2*radius*sizeof(float),stream_halo[i]);
    }
    for(int i=0; i<n_gpus-1; i++){
      cudaSetDevice(i);
      cudaStreamSynchronize(stream_halo[i]);
    }
    for(int i=0; i<n_gpus-1; i++){
      cudaMemcpyPeerAsync(data_p0[i+1]+offset_rcv_h1,i+1,data_p0[i]+offset_snd_h2,i,n1*n2*radius*sizeof(float),stream_halo[i]);
      if(it<nt-1) cudaMemcpyPeerAsync(src_p0[i+1] +offset_rcv_h1,i+1,src_p0[i] +offset_snd_h2,i,n1*n2*radius*sizeof(float),stream_halo[i]);
    }

  /*if(it==(nt-200)){
    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(device[i]);
      cudaMemcpy(snap+i*n1*n2*(n3-2*radius), data_p0[i]+radius*n1*n2, n1*n2*(n3-2*radius)*sizeof(float), cudaMemcpyDeviceToHost);
    }
   srite("s",snap,4*n1*n2*n3_total);
  }*/

    cudaSetDevice(device[0]);
    new_data_inject_kernel<<<dimGridx,dimBlock,0,stream_internal[0]>>>(id,ii, data_p0[0]/*+offset_snd_h1*/);


    if(ii==0){
      for(int i=0; i<n_gpus; i++){
        cudaSetDevice(device[i]);
        img_kernel<<<dimGrid,dimBlock,0,stream_internal[i]>>>(img_gpu[i]+lead_pad,data_p0[i]+lead_pad,src_p0[i]+lead_pad);
      }
    }

    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(i);
      cudaDeviceSynchronize();
      if(it<nt-1){ ptemp=src_p0[i]; src_p0[i]=src_p1[i];src_p1[i]=ptemp;}
      ptemp2=data_p1[i]; data_p1[i]=data_p0[i]; data_p0[i]=ptemp2;
    }
  }

  //error = cudaGetLastError();
  //process_error( error, "kernel" );

  cudaSetDevice(device[0]);
  cudaEventRecord(stop,0);

  cudaDeviceSynchronize();

  float time_total;
  cudaEventElapsedTime(&time_total,start,stop);
  fprintf(stderr,"Time for imaging = %f seconds \n",time_total/1000);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  for(int i=0; i<n_gpus; i++){
    cudaSetDevice(i);
   fprintf(stderr,"CHECK2 %d %d %d \n",n1,n2,n3);
    cudaMemcpy(img+i*n1*n2*(n3-2*radius), img_gpu[i]/*+radius*n1*n2*/, n1*n2*(n3/*-2*radius*/)*sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(img+i*n1*n2*(n3-2*radius), img_gpu[i]+radius*n1*n2, n1*n2*(n3-2*radius)*sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy( img_gpu[i]/*+radius*n1*n2*/, img+i*n1*n2*(n3-2*radius), n1*n2*(n3/*-2*radius*/)*sizeof(float), cudaMemcpyHostToDevice);
  }

  for(int i=0; i<n_gpus; i++){
    cudaSetDevice(i);
    cudaFree(data_p0[i]);
    cudaFree(img_gpu[i]);
    cudaFree(data_p1[i]);
    cudaFree(src_p0[i]);
    cudaFree(src_p1[i]);
  }
  cudaFree(data_gpu);
  cudaFree(datageom_gpu);
  cudaFree(source_gpu0);
  cudaFree(srcgeom_gpu0);
}

void transfer_sinc_table_s(int nsinc, int ns,  float **tables){
	cudaSetDevice(0);
   float *tmp_table1=(float*)malloc(sizeof(float)*nsinc*ns);
   for(int i=0; i < ns; i++) memcpy((tmp_table1+nsinc*i),tables[i],nsinc*sizeof(float));
   cudaMalloc((void**) &sincstable,ns*nsinc*sizeof(float));
   cudaMemcpy(sincstable, tmp_table1, ns*nsinc*sizeof(float),cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(sinc_s_table, &sincstable, sizeof(float*));
   free(tmp_table1);
   cudaMemcpyToSymbol(nsinc_gpu, &nsinc, sizeof(int));
 }

void transfer_sinc_table_d(int nsinc,  int nd, float **tabled){
	cudaSetDevice(0);
   float *tmp_table2=(float*)malloc(sizeof(float)*nsinc*nd);
   for(int i=0; i < nd; i++) memcpy((tmp_table2+nsinc*i),tabled[i],nsinc*sizeof(float));
   cudaMalloc((void**) &sincdtable,nd*nsinc*sizeof(float));
   cudaMemcpy(sincdtable, tmp_table2, nd*nsinc*sizeof(float),cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(sinc_d_table, &sincdtable, sizeof(float*));
   free(tmp_table2);
}

void transfer_source_func(int npts, int nt, int *locs, float *vals){
   shot_gpu=0;
   cudaSetDevice(device[0]);
   cudaMalloc((void**) &source_gpu,nt*npts*sizeof(float));
   cudaMalloc((void**) &srcgeom_gpu,npts*sizeof(int));
   fprintf(stderr,"transfer 1 %d %d %d \n",nt,npts,sizeof(float));
   cudaMemcpy(srcgeom_gpu, locs, npts*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(source_gpu, vals, nt*npts*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(ntrace_gpu, &nt, sizeof(int));
   cudaMemcpyToSymbol(srcgeom_gpu0, &srcgeom_gpu, sizeof(int*));
   cudaMemcpyToSymbol(source_gpu0, &source_gpu, sizeof(float*));
}

void transfer_receiver_func(int nx, int ny, int nt, int *locs, float *vals){
   cudaSetDevice(device[0]);
   cudaMalloc((void**) &data_gpu,nt*nx*ny*sizeof(float));
   cudaMalloc((void**) &datageom_gpu,nx*ny*sizeof(float));
   cudaMemcpy(datageom_gpu, locs, nx*ny*sizeof(float),	 cudaMemcpyHostToDevice);
   cudaMemcpy(data_gpu, vals, (long long)nt*(long long)nx*(long long)ny*sizeof(float), cudaMemcpyHostToDevice);
     fprintf(stderr,"transfer 1 %d %d %d %d \n",nt,nx,ny,sizeof(float));

  cudaMemcpyToSymbol(rec_nx_gpu, &nx, sizeof(int));
  cudaMemcpyToSymbol(rec_ny_gpu, &ny, sizeof(int));
  cudaMemcpyToSymbol(ntrace_gpu, &nt, sizeof(int));
   cudaMemcpyToSymbol(datageom_gpu0, &datageom_gpu, sizeof(int*));
   cudaMemcpyToSymbol(data_gpu0, &data_gpu, sizeof(float*));
}


void transfer_vel_func1(int n1, int n2, int n3, float *vel){
    n3=(n3-2*radius)/n_gpus+2*radius;
    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(device[i]);
      cudaMemcpy( velocity[i]/*+lead_pad*/, vel+i*n1*n2*(n3-2*radius), n1*n2*n3*sizeof(float), cudaMemcpyHostToDevice);
    }
}

void transfer_vel_func2(int n1, int n2, int n3, float *vel){
    n3=(n3-2*radius)/n_gpus+2*radius;
    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(device[i]);
      cudaMemcpy( velocity2[i]/*+lead_pad*/, vel+i*n1*n2*(n3-2*radius), n1*n2*n3*sizeof(float), cudaMemcpyHostToDevice);
    }
}

void create_gpu_space(float d1, float d2, float d3, float bc_a, float bc_b, float bc_b_y, int n1, int n2, int n3){

    lead_pad=0;//32-radius;
    n3=(n3-2*radius)/n_gpus+2*radius;
    offset=radius*n1*n2+radius*n1+radius+lead_pad;

    float coeffs_cpu[COEFFS_SIZE]=get_coeffs((double)d1,(double)d2,(double)d3);

    dd1=1./(double)d1/(double)d1;
    dd2=1./(double)d2/(double)d2;
    dd3=1./(double)d3/(double)d3;

    for(int i=0; i<n_gpus; i++){
      cudaSetDevice(device[i]);
      cudaMalloc( (void**)&velocity[i], (n1*n2*n3/*+lead_pad*/)*sizeof(float));
      cudaMalloc( (void**)&velocity2[i], (n1*n2*n3/*+lead_pad*/)*sizeof(float));

      cudaMemcpyToSymbol(n1gpu, &n1, sizeof(int));
      cudaMemcpyToSymbol(n2gpu, &n2, sizeof(int));
      cudaMemcpyToSymbol(n3gpu, &n3, sizeof(int));

      cudaMemcpyToSymbol(bc_agpu, &bc_a, sizeof(float));
      cudaMemcpyToSymbol(bc_bgpu, &bc_b, sizeof(float));
      cudaMemcpyToSymbol(coeffs, coeffs_cpu, COEFFS_SIZE*sizeof(float), 0,cudaMemcpyHostToDevice);

    }
}

