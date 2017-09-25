#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include "cpu_prop.h"
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

cpuProp::cpuProp(std::shared_ptr<SEP::genericIO> io){
	storeIO(io);

}

void cpuProp::rtmForward(int n1, int n2, int n3, int jt, float *img,
	float *rec, int npts, int nt, int nt_big, int rec_nx, int rec_ny){
    _jt=jt;
	std::vector<float> rec_p0(_n123,0.),rec_p1(_n123,0.);
	std::vector<float> src_p0(_n123,0.),src_p1(_n123,0.);
    float *r_p0=rec_p0.data(),*r_p1=rec_p1.data();
    float *s_p0=src_p0.data(),*s_p1=src_p1.data();


_dir=1;
for(int it=0; it < nt; it++){
    int id=it/jt;
    int ii=it-id*jt;
fprintf(stderr,"forward   %d of %d \n",it,nt);


	prop(s_p0,s_p1,_vel2);
    prop(r_p0,r_p1,_vel2);
    //stats(s_p0,"after prop");
    damp(s_p0,s_p1);
    damp(r_p0,r_p1);
    injectSource(id,ii,s_p0);

    imageAdd(img,r_p0,s_p0);
     
    dataExtract(id,ii,r_p0);


    float *pt=s_p0;
		s_p0=s_p1;
		s_p1=pt;
		pt=r_p0;
		r_p0=r_p1;
		r_p1=pt;
    }


}

void cpuProp::rtmAdjoint(int n1, int n2, int n3, int jtd, float *src_p0, float *src_p1,
	float *img, int npts_s, int nt){
//    rtm_adjoint(ad1.n,ad2.n,ad3.n,jtd,src_p0->vals,src_p1->vals,img->vals,npts_s,nt/*,src,recx*/);
	std::vector<float> rec_p0(_n123,0.),rec_p1(_n123,0.);
	float *r_p0=rec_p0.data(), *r_p1=rec_p1.data();
	_dir=-1;

		   float sm1=0,sm2=0;
		   for(int i=0; i < n1*n2*n3; i++){
		     sm1=sm1+fabsf(src_p0[i]);
		     		     sm2=sm2+fabsf(src_p1[i]);

		    }

int ic=0;

	for(int it=nt-1; it >=0; it--) {
	
	
	
	fprintf(stderr,"running adjoint %d  _nz=%d \n",it,_nz);
		int id_s=(it+1)/_jtsS;
		int i_s=it+1-id_s*_jtsS;
		int id=it/_jtdD;
		int ii=it-id*_jtdD;


		if(it>0) prop(r_p0,r_p1,_vel2);
		if(it< nt-1) {
			prop(src_p0,src_p1,_vel1);
			injectSource(id_s,i_s,src_p1);                                                                                                                                                                                                                                                                                                                                                                                                                               //I think this should be p0;
		}
		damp(r_p0,r_p1);
		injectReceivers(id,ii,r_p0);
		imageCondition(r_p0,src_p0,img);


		float *pt=src_p0;
		src_p0=src_p1;
		src_p1=pt;
		pt=r_p0;
		r_p0=r_p1;
		r_p1=pt;
		
		ic++;

	}

}
void cpuProp::imageCondition(float *rec, float *src, float *img){
 tbb::parallel_for(tbb::blocked_range<int>(0,_n123),[&](
  const tbb::blocked_range<int>&r){
  for(int  i=r.begin(); i!=r.end(); ++i){
		img[i]+=src[i]*rec[i];
	}

});
}
void cpuProp::sourceProp(int nx, int ny, int nz, bool damp, bool getLast,
	float *p0, float *p1, int jts, int npts, int nt){

    _jt=jts;


	int n12=_nx*_ny;
	_dir=1;
	for(int it=0; it<=nt; it++) {
		int id=it/_jtsS;
		int ii=it-id*_jtsS;
		prop(p0,p1,_vel1);

		injectSource(id,ii,p0);

		float *pt=p1; p1=p0; p0=pt;	
	}
	if(nt%2==1){
	   float *x=new float[_n123];
	   memcpy(x,p0,sizeof(float)*_n123);
	   memcpy(p1,p0,sizeof(float)*_n123);
	   memcpy(p0,x,sizeof(float)*_n123); 
	}

}
void cpuProp::stats(float *buf, std::string title){
float en= tbb::parallel_reduce(
        tbb::blocked_range<float*>( &buf[0], &buf[0]+_n123 ),
        0.f,
        [](const tbb::blocked_range<float*>& r, double init)->double {
            for( float* a=r.begin(); a!=r.end(); ++a )
                init += (*a)*(*a);
            return init;
        },
        []( double x, double y )->double {
            return x+y;
        }
    );

std::cerr<<title<<":STATS:"<<en<<std::endl;
}
void cpuProp::damp(float *p0,float *p1){

     tbb::parallel_for(tbb::blocked_range<int>(4,_nz-4),[&](
  const tbb::blocked_range<int>&r){
  for(int  i3=r.begin(); i3!=r.end(); ++i3){
		int edge1=std::min(i3-4,_nz-4-i3);
		for(int i2=4; i2 < _ny-4; i2++) {
			int edge2=std::min(edge1,std::min(i2-4,_ny-4-i2));
			int ii=i2*_nx+4+_n12*i3;
			for(int i1=4; i1 < _nx-4; i1++,ii++) {
				int edge=std::min(edge2,std::min(i1-4,_nx-4-i1));
	
				if(edge>=0 && edge < _bound.size()) {
					//		if(i2==600 && i1==600 && i3 < 70)
				 // fprintf(stderr,"check i3=%d edge=%d value=%f \n",
				 //x i3,edge,_bound[edge]); 
					p0[ii]*=_bound[edge];
					p1[ii]*=_bound[edge];
				}
			}
		}
	}
});
}
void cpuProp::injectSource(int id, int ii, float *p){
   
   if(id+7 >= _ntSrc)  return;
	for(int i=0; i < _nptsS; i++) {

		p[_locsS[i]]+=_dir/(float)_jt*(
			_tableS[ii][0]*_sourceV[_ntSrc*i+id]+
			_tableS[ii][1]*_sourceV[_ntSrc*i+id+1]+
			_tableS[ii][2]*_sourceV[_ntSrc*i+id+2]+
			_tableS[ii][3]*_sourceV[_ntSrc*i+id+3]+
			_tableS[ii][4]*_sourceV[_ntSrc*i+id+4]+
			_tableS[ii][5]*_sourceV[_ntSrc*i+id+5]+
			_tableS[ii][6]*_sourceV[_ntSrc*i+id+6]+
			_tableS[ii][7]*_sourceV[_ntSrc*i+id+7]);
			

			
	}
	
}
void cpuProp::injectReceivers(int id, int ii, float *p){


   if(id+7 >= _ntRec)  return;
float sc=(float)_dir/(float) _jt;
     tbb::parallel_for(tbb::blocked_range<int>(0,_nRecs),[&](
  const tbb::blocked_range<int>&r){
  for(int  i=r.begin(); i!=r.end(); ++i){
		p[_locsR[i]]+=sc*(
			_tableD[ii][0]*_rec[_ntRec*i+id]+
			_tableD[ii][1]*_rec[_ntRec*i+id+1]+
			_tableD[ii][2]*_rec[_ntRec*i+id+2]+
			_tableD[ii][3]*_rec[_ntRec*i+id+3]+
			_tableD[ii][4]*_rec[_ntRec*i+id+4]+
			_tableD[ii][5]*_rec[_ntRec*i+id+5]+
			_tableD[ii][6]*_rec[_ntRec*i+id+6]+
			_tableD[ii][7]*_rec[_ntRec*i+id+7]);


	}
	});
}
void cpuProp::dataExtract(int id, int ii, float *p){


   if(id+7 >= _ntRec)  return;

     tbb::parallel_for(tbb::blocked_range<int>(0,_nRecs),[&](
  const tbb::blocked_range<int>&r){
  for(int  i=r.begin(); i!=r.end(); ++i){
    _rec[_ntRec*i+id]+=p[_locsR[i]]*_tableD[ii][0];
    _rec[_ntRec*i+id+1]+=p[_locsR[i]]*_tableD[ii][1];
    _rec[_ntRec*i+id+2]+=p[_locsR[i]]*_tableD[ii][2];
    _rec[_ntRec*i+id+3]+=p[_locsR[i]]*_tableD[ii][3];
    _rec[_ntRec*i+id+4]+=p[_locsR[i]]*_tableD[ii][4];
    _rec[_ntRec*i+id+5]+=p[_locsR[i]]*_tableD[ii][5];
    _rec[_ntRec*i+id+6]+=p[_locsR[i]]*_tableD[ii][6];
    _rec[_ntRec*i+id+7]+=p[_locsR[i]]*_tableD[ii][7];
	}
	});
}
void cpuProp::imageAdd(float *img,  float *recField, float *srcField){
     tbb::parallel_for(tbb::blocked_range<int>(0,_n123),[&](
  const tbb::blocked_range<int>&r){
  for(int  i=r.begin(); i!=r.end(); ++i){
    recField[i]+=.000001*srcField[i]*img[i];
  }});
}
void cpuProp::prop(float *p0, float *p1, float *vel){



		     tbb::parallel_for(tbb::blocked_range<int>(4,_nz-4),[&](
  const tbb::blocked_range<int>&r){
  for(int  i3=r.begin(); i3!=r.end(); ++i3){
//	for(int i3=4; i3 < _nz-4; i3++){
		for(int i2=4; i2 < _ny-4; i2++) {
			int ii=i2*_nx+4+_n12*i3;
			for(int i1=4; i1 < _nx-4; i1++,ii++) {
		       float x=
				p0[ii]=vel[ii]*
					      (
					coeffs[C0]*p1[ii]
					+coeffs[CX1]*(p1[ii-1]+p1[ii+1])+
					+coeffs[CX2]*(p1[ii-2]+p1[ii+2])+
					+coeffs[CX3]*(p1[ii-3]+p1[ii+3])+
					+coeffs[CX4]*(p1[ii-4]+p1[ii+4])+
					+coeffs[CY1]*(p1[ii-_nx]+p1[ii+_nx])+
					+coeffs[CY2]*(p1[ii-2*_nx]+p1[ii+2*_nx])+
					+coeffs[CY3]*(p1[ii-3*_nx]+p1[ii+3*_nx])+
					+coeffs[CY4]*(p1[ii-4*_nx]+p1[ii+4*_nx])+
					+coeffs[CZ1]*(p1[ii-1*_n12]+p1[ii+1*_n12])+
					+coeffs[CZ2]*(p1[ii-2*_n12]+p1[ii+2*_n12])+
					+coeffs[CZ3]*(p1[ii-3*_n12]+p1[ii+3*_n12])+
					+coeffs[CZ4]*(p1[ii-4*_n12]+p1[ii+4*_n12])
				        )
				        +p1[ii]+p1[ii]-p0[ii];
				        
				      
			}
		}
	}
	});
	
}
void cpuProp::transferSincTableD(int nsinc, int jtd, std::vector<std::vector<float>> &table){
// transfer_sinc_table_d(nsinc,jtd,myr.table);
	_nsincD=nsinc;
	_jtdD=jtd;
	_tableD=table;

}
void cpuProp::transferSourceFunc(int npts,int nt_big,std::vector<int> &locs,float *vals){
	_nptsS=npts; _ntSrc=nt_big; _locsS=locs; _sourceV=vals;
	
}
void cpuProp::transferVelFunc1(int nx, int ny, int nz, float *vloc){
	_nx=nx; _ny=ny; _nz=nz; _vel1=vloc;
}
void cpuProp::transferVelFunc2(int nx, int ny, int nz, float *vloc){
	_nx=nx; _ny=ny; _nz=nz; _vel2=vloc;

}

void cpuProp::transferReceiverFunc(int nx, int ny, int nt, std::vector<int> &locs,
	float * rec){
	_nRecs=nx*ny; _ntRec=nt; _locsR=locs; _rec=rec;
	int ii=0;
	
}
void cpuProp::transferSincTableS(int nsinc, int jts, std::vector<std::vector<float>> &table){
  _tableS=table;
	_nsincS=nsinc;
	_jtsS=jts;
//	_tableS=table;
}
#define C_C00(d) (8.0/(5.0*(d)*(d)))

void cpuProp::createSpace(float d1, float d2, float d3,float bc_a, float bc_b, float bc_y,
	int nx, int ny, int nz){
	_bcA=bc_a; _bcB=bc_b; bc_y=_bcY;
	_nx=nx; _ny=ny; _nz=nz;
	_n12=_nx*_ny;
	coeffs.resize(13);
	coeffs[0]=-8/12.5/12.5;
	coeffs[1]=coeffs[2]=coeffs[3]=1./12.5/12.5;
	coeffs[4]=coeffs[5]=coeffs[6]=0.;
	coeffs[7]=coeffs[8]=coeffs[9]=0.;
	coeffs[10]=coeffs[11]=coeffs[12]=0.;

	
	coeffs[0]=-1025.0/576.0*(C_C00(d1)+C_C00(d2)+C_C00(d3));
	coeffs[1]=C_C00(d1);
	coeffs[2]=C_C00(d2);
	coeffs[3]=C_C00(d3);
	coeffs[4]=-C_C00(d1)/8.0;
	coeffs[5]=-C_C00(d2)/8.0;
	coeffs[6]=-C_C00(d3)/8.0;
	coeffs[7]=C_C00(d1)/63.0;
	coeffs[8]=C_C00(d2)/63.0;
	coeffs[9]=C_C00(d3)/63.0;
	coeffs[10]=-C_C00(d1)/896.0;
	coeffs[11]=-C_C00(d2)/896.0;
	coeffs[12]=-C_C00(d3)/896.0;
	
	
	

	_n123=_n12*nz;
	_bcB=.0005;
	_bcA=40;
	_bound.resize(40);
	for(int i=0;i < _bound.size(); i++) _bound[i]=expf(-_bcB*(_bcA-i));
/*
	coeffs[1]=.8/d1/d1;
	coeffs[2]=.8/d2/d2;
	coeffs[3]=.8/d3/d3;
	coeffs[4]=-.1/d1/d1;
	coeffs[5]=-.1/d2/d2;
	coeffs[6]=-.1/d3/d3;
    coeffs[7]=.0126984126/d1/d1;
    coeffs[8]=.0126984126/d2/d2;
    coeffs[9]=.0126984126/d3/d3;
    coeffs[10]=-.0008928571428/d1/d1;
    coeffs[11]=-.0008928571428/d2/d2;
    coeffs[12]=-.0008928571428/d3/d3;
    coeffs[0]=0;
    for(int i=1; i < 13; i++) coeffs[0]-=coeffs[i];
    */



	//create_gpu_space(d1,d2,d3,bc_a,bc_b,bc_b_y,nx,ny,nz);
}
