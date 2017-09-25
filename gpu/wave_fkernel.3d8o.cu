/* 3D Time domain finite difference kernel
	By: Chris Leader, Abdullah AlTheyab
	Ref: similar to Paulius code but for 2D, then re-extended to 3D
 */
#define BLOCKZ_SIZE 16
#define BLOCKX_SIZE 16 
#define BLOCKY_SIZE 16
#define FAT 4
#define COEFFS_SIZE 13
#define radius 4

#define C0  0
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

#define C_C00(d) (8.0/(5.0*(d)*(d)))
#define get_coeffs(d1,d2,d3) {-1025.0/576.0*(C_C00(d1)+C_C00(d2)+C_C00(d3)), C_C00(d1), C_C00(d2), C_C00(d3),-C_C00(d1)/8.0,-C_C00(d2)/8.0,-C_C00(d3)/8.0,C_C00(d1)/63.0,C_C00(d2)/63.0,C_C00(d3)/63.0,-C_C00(d1)/896.0,-C_C00(d2)/896.0,-C_C00(d3)/896.0}

__constant__ float coeffs[COEFFS_SIZE];

extern "C" __global__ void wave_kernel(float *p0, float *p1, float *p2, float *vel, const int start3, const int end3){
	__shared__ float p1s[BLOCKY_SIZE+2*FAT][BLOCKX_SIZE+2*FAT];

	int ig = blockIdx.x * blockDim.x + threadIdx.x; // Global coordinates for the fastest two axes
	int jg = blockIdx.y * blockDim.y + threadIdx.y;

	int il = threadIdx.x + FAT; 			//Local coordinates for the fastest two axes
	int jl = threadIdx.y + FAT; 

	float p1y[2*radius+1]; 				//Array of elements to hold slow axis values

	int stride = n1gpu * n2gpu;  			// Number of elements between wavefield slices
	int addr = ig + n1gpu * jg;			// Index of the central slow-axis element
	int addr_fwd= addr - radius * stride; 		// Index of the first slow-axis element

	// Assign slow axis values
	p1y[1]=p1[addr_fwd];
	p1y[2]=p1[addr_fwd+=stride];
	p1y[3]=p1[addr_fwd+=stride];
	p1s[jl][il]=p1[addr_fwd+=stride]; 		//Copy to shared memory
	p1y[5]=p1[addr_fwd+=stride];
	p1y[6]=p1[addr_fwd+=stride];
	p1y[7]=p1[addr_fwd+=stride];
	p1y[8]=p1[addr_fwd+=stride];

	//#pragma unroll 9
	for(int yl=start3; yl<end3; yl++){

		// Update slow axis values
		p1y[0]=p1y[1];
		p1y[1]=p1y[2];
		p1y[2]=p1y[3];
		p1y[3]=p1s[jl][il];	
		p1s[jl][il]=p1y[5];
		p1y[5]=p1y[6];
		p1y[6]=p1y[7];
		p1y[7]=p1y[8];
		p1y[8]=p1[addr_fwd+=stride];


		if(threadIdx.y<FAT){
			p1s[threadIdx.y][il]=p1[addr-FAT*n1gpu];
			p1s[jl+BLOCKX_SIZE][il]=p1[addr+BLOCKX_SIZE*n1gpu];
		}
        	if( threadIdx.y >= FAT && threadIdx.y < FAT ){
			p1s[threadIdx.y+BLOCKY_SIZE][il]=p1[addr+(BLOCKY_SIZE-FAT)*n1gpu];
			p1s[threadIdx.y+BLOCKY_SIZE][il+BLOCKX_SIZE]=p1[addr+(BLOCKY_SIZE-FAT)*n1gpu+BLOCKX_SIZE];
		}
		if(threadIdx.x<FAT){
			p1s[jl][threadIdx.x]=p1[addr-FAT];
			p1s[jl][il+BLOCKX_SIZE]=p1[addr+BLOCKX_SIZE];
		}

		__syncthreads();

		p2[addr]=
		vel[addr]*
			(
			 coeffs[C0]*p1s[jl][il]
			+coeffs[CZ1]*(p1s[jl][il+1]+p1s[jl][il-1])
			+coeffs[CX1]*(p1s[jl+1][il]+p1s[jl-1][il])
			+coeffs[CY1]*(p1y[radius+1]+p1y[radius-1])
			+coeffs[CZ2]*(p1s[jl][il+2]+p1s[jl][il-2])
			+coeffs[CX2]*(p1s[jl+2][il]+p1s[jl-2][il])
			+coeffs[CY2]*(p1y[radius+2]+p1y[radius-2])
			+coeffs[CZ3]*(p1s[jl][il+3]+p1s[jl][il-3])
			+coeffs[CX3]*(p1s[jl+3][il]+p1s[jl-3][il])
			+coeffs[CY3]*(p1y[radius+3]+p1y[radius-3])
			+coeffs[CZ4]*(p1s[jl][il+4]+p1s[jl][il-4])
			+coeffs[CX4]*(p1s[jl+4][il]+p1s[jl-4][il])
			+coeffs[CY4]*(p1y[radius+4]+p1y[radius-4])
			)
			+p1s[jl][il]+p1s[jl][il]-p0[addr];
			addr+=stride;
	}
}


extern "C" __global__ void wave_kernel_adj(float *p0, float *p1, float *p2, float *vel, const int start3, const int end3){
	__shared__ float p1s[BLOCKX_SIZE+2*FAT][BLOCKY_SIZE+2*FAT];
	__shared__ float vls[BLOCKX_SIZE+2*FAT][BLOCKY_SIZE+2*FAT];

	int ig = blockIdx.x * blockDim.x + threadIdx.x; // Global coordinates for the fastest two axes
	int jg = blockIdx.y * blockDim.y + threadIdx.y;

	int il = threadIdx.x + FAT; 			//Local coordinates for the fastest two axes
	int jl = threadIdx.y + FAT; 

	float p1y[2*radius+1]; 				//Array of elements to hold slow axis values
	float vly[2*radius+1];				//Array to hold out of plane velocity values

	int stride = n1gpu * n2gpu;  			// Number of elements between wavefield slices
	int addr = ig + n1gpu * jg;			// Index of the central slow-axis element
	int addr_fwd= addr - radius * stride; 		// Index of the first slow-axis element

	// Assign slow axis values
	p1y[1]=p1[addr_fwd];			vly[1]=vel[addr_fwd];
	p1y[2]=p1[addr_fwd+=stride];		vly[2]=vel[addr_fwd];
	p1y[3]=p1[addr_fwd+=stride];		vly[3]=vel[addr_fwd];
	p1s[jl][il]=p1[addr_fwd+=stride];	vls[jl][il]=vel[addr_fwd];
	p1y[5]=p1[addr_fwd+=stride];		vly[5]=vel[addr_fwd];
	p1y[6]=p1[addr_fwd+=stride];		vly[6]=vel[addr_fwd];
	p1y[7]=p1[addr_fwd+=stride];		vly[7]=vel[addr_fwd];
	p1y[8]=p1[addr_fwd+=stride];		vly[8]=vel[addr_fwd];

	//#pragma unroll 9
	for(int yl=start3; yl<end3; yl++){

		// Update slow axis values
		p1y[0]=p1y[1];			vly[0]=vly[1];
		p1y[1]=p1y[2];			vly[1]=vly[2];
		p1y[2]=p1y[3];			vly[2]=vly[3];
		p1y[3]=p1s[jl][il];		vly[3]=vls[jl][il];
		p1s[jl][il]=p1y[5];		vls[jl][il]=vly[5];
		p1y[5]=p1y[6];			vly[5]=vly[6];
		p1y[6]=p1y[7];			vly[6]=vly[7];
		p1y[7]=p1y[8];			vly[7]=vly[8];
		p1y[8]=p1[addr_fwd+=stride];	vly[8]=vel[addr_fwd];

		if(threadIdx.x<FAT){
			p1s[jl][threadIdx.x]=p1[addr-FAT];
			p1s[jl][il+BLOCKZ_SIZE]=p1[addr+BLOCKZ_SIZE];
			vls[jl][threadIdx.x]=vel[addr-FAT];
			vls[jl][il+BLOCKZ_SIZE]=vel[addr+BLOCKZ_SIZE];
		}
		if(threadIdx.y<FAT){
			p1s[threadIdx.y][il]=p1[addr-FAT*n1gpu];
			p1s[jl+BLOCKX_SIZE][il]=p1[addr+BLOCKX_SIZE*n1gpu];
			vls[threadIdx.y][il]=vel[addr-FAT*n1gpu];
			vls[jl+BLOCKX_SIZE][il]=vel[addr+BLOCKX_SIZE*n1gpu];
		}
		__syncthreads();

		p2[addr]=
			 coeffs[C0]*p1s[jl][il]*vls[jl][il]
			+coeffs[CZ1]*(p1s[jl][il+1]*vls[jl][il+1]+p1s[jl][il-1]*vls[jl][il-1])
			+coeffs[CX1]*(p1s[jl+1][il]*vls[jl+1][il]+p1s[jl-1][il]*vls[jl-1][il])
			+coeffs[CY1]*(p1y[radius+1]*vly[radius+1]+p1y[radius-1]*vly[radius-1])
			+coeffs[CZ2]*(p1s[jl][il+2]*vls[jl][il+2]+p1s[jl][il-2]*vls[jl][il-2])
			+coeffs[CX2]*(p1s[jl+2][il]*vls[jl+2][il]+p1s[jl-2][il]*vls[jl-2][il])
			+coeffs[CY2]*(p1y[radius+2]*vly[radius+2]+p1y[radius-2]*vly[radius-2])
			+coeffs[CZ3]*(p1s[jl][il+3]*vls[jl][il+3]+p1s[jl][il-3]*vls[jl][il-3])
			+coeffs[CX3]*(p1s[jl+3][il]*vls[jl+3][il]+p1s[jl-3][il]*vls[jl-3][il])
			+coeffs[CY3]*(p1y[radius+3]*vly[radius+3]+p1y[radius-3]*vly[radius-3])
			+coeffs[CZ4]*(p1s[jl][il+4]*vls[jl][il+4]+p1s[jl][il-4]*vls[jl][il-4])
			+coeffs[CX4]*(p1s[jl+4][il]*vls[jl+4][il]+p1s[jl-4][il]*vls[jl-4][il])
			+coeffs[CY4]*(p1y[radius+4]*vly[radius+4]+p1y[radius-4]*vly[radius-4])
			+p1s[jl][il]+p1s[jl][il]-p0[addr];
			addr+=stride;
	}
}

extern "C" __global__ void damp_kernel(float *p0, float *p1, const int start3, const int end3, const int gpu_id, const int n_gpus){

	int ig = blockIdx.x * blockDim.x + threadIdx.x; // Global coordinates for the fastest two axes
	int jg = blockIdx.y * blockDim.y + threadIdx.y;

	int stride = n1gpu * n2gpu;  			// Number of elements between wavefield slices
	int edge=0;
	//int bc_agpu=40;
	//float bc_bgpu=0.0005;

	for(int zg=0; zg<n3gpu; zg++){
		if(n_gpus==1) 			edge=min2(ig-FAT, min2(zg-FAT,min2(min2(n2gpu-FAT-jg,n1gpu-FAT-ig),min2(n3gpu-FAT-zg,jg-FAT)))); // Damp all
		else if(gpu_id==0) 		edge=min2(ig-FAT, min2(zg-FAT,min2(min2(n2gpu-FAT-jg,n1gpu-FAT-ig),jg-FAT))); //Don't damp bottom
		else if(gpu_id==(n_gpus-1))	edge=min2(ig-FAT, min2(min2(n2gpu-FAT-jg,n1gpu-FAT-ig),min2(n3gpu-FAT-zg,jg-FAT))); //Don't damp top
		else 				edge=min2(ig-FAT, min2(min2(n2gpu-FAT-jg,n1gpu-FAT-ig),jg-FAT)); //Don't damp top or bottom
		//edge=min2(ig-FAT, min2(zg-FAT,min2(min2(n2gpu-FAT-jg,n1gpu-FAT-ig),min2(n3gpu-FAT-zg,jg-FAT))));
		if(edge>=0){
		  int addr=ig+n1gpu*jg+stride*zg;
		  float temp=expf(-bc_bgpu*(bc_agpu-edge));
		  if(temp<1.){
		    p1[addr]*=temp;
		    p0[addr]*=temp;
		  }
		}
	}


}

extern "C" __global__ void damp_top_kernel(int nblocksz, float *p0, float *p1){
	int yblock=(blockIdx.x/nblocksz);
	int y0=FAT+yblock*BLOCKY_SIZE;
	int zl=FAT+threadIdx.x; 				// local  zdir
	int zg=(blockIdx.x-yblock*nblocksz)*blockDim.x+zl; 	// global zdir
	int xl=FAT+threadIdx.y; 				// local  xdir
	int xg=blockIdx.y*blockDim.y+xl; 			// global xdir	

	int stride=n1gpu*n2gpu;
	int bc_agpu=50;
	float bc_bgpu=0.0005;

	for(int yl=0; yl<BLOCKY_SIZE; ++yl){
		int y=y0+yl;
		int edge=n1gpu-FAT-zg;//min2(zg-FAT ,n1gpu-FAT-zg);
		if(edge>=0){
		  int addr=zg+n1gpu*xg+stride*y;
		  float temp=expf(-bc_bgpu*(bc_agpu-edge));
		  if(temp<1.){
		    p1[addr]*=temp;
		    p0[addr]*=temp;
		  }
		}
	}
}
