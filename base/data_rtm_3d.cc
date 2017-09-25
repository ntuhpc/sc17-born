#include "data_rtm_3d.h"
#include <math.h>

data_rtm_3d::data_rtm_3d(std::string tg, std::shared_ptr<SEP::genericIO> io){

	// Here we set up the data vectors, relevant axes and set to zero

	_io=io;


	tagInit(tg);
	SEP::axis a=getAxis(1);
	max_time=a.o+a.d*(a.n-1);
	max_time=max_time;
	dt=a.d;
	nt=a.n;
	rec_locs=0; data=0;
	dsamp=dt;

	sz=io->getParamObj()->getFloat("rec_depth",0.);

}

int data_rtm_3d::get_points(){
	// Simply get the surface grid size
	return getAxis(2).n*getAxis(3).n;
}

int data_rtm_3d::get_points1(){
	// Return x axis length
	return getAxis(2).n;
}

int data_rtm_3d::get_points2(){
	// Return y axis length
	return getAxis(3).n;
}

void data_rtm_3d::add_data(int ishot, std::shared_ptr<hypercube_float> dat){

	int n1=getAxis(1).n;
	int n2=getAxis(2).n;
	int n3=getAxis(3).n;
	_file->seekTo((long long)(getAxis(2).n*ishot*getAxis(3).n)*(long long)(getAxis(1).n*4),0);

	std::shared_ptr<hypercube_float> tmp=dat->clone(); tmp->scale(0.);
	for(long long i3=0; i3 < getAxis(3).n; i3++)
		_file->readFloatStream(tmp->vals+i3*n1*n2,n1*n2);

	for(long long i=0; i <  tmp->getN123(); i++) tmp->vals[i]+=dat->vals[i];

	_file->seekTo((long long)(getAxis(2).n*ishot*getAxis(3).n)*(long long)(getAxis(1).n*4),0);
	for(long long i3=0; i3 < getAxis(3).n; i3++)
		_file->writeFloatStream(tmp->vals+i3*n1*n2,n1*n2);

}

void data_rtm_3d::get_source_func(std::shared_ptr<hypercube_float> domain, int ishot,
	std::vector<float>&s_x, std::vector<float>&s_y, std::vector<float>&s_z, int nsinc,
	int nts,std::shared_ptr<hypercube_float> time){

	// Read the data and set up the source geometry

	SEP::axis a1=domain->getAxis(1);
	SEP::axis a2=domain->getAxis(2);
	SEP::axis a3=domain->getAxis(3);
	SEP::axis at=getAxis(1);

	// Seek to shot position and read
	
	
	_file->seekTo(ishot*(long long)(getAxis(3).n*getAxis(2).n)*(long long)(getAxis(1).n*4),0);
	_file->readFloatStream(time->vals,getAxis(1).n*getAxis(2).n*getAxis(3).n);

	int i=0;


	// Set up the regular source geometry
	for(int i3=0; i3< (getAxis(3).n); i3++) {
		for(int i2=0; i2< getAxis(2).n; i2++, i++) {
			s_z[i]=(sz-a3.o)/a3.d;
			s_x[i]=(getAxis(2).o+getAxis(2).d*i2-a1.o)/a1.d;
			s_y[i]=(getAxis(3).o+getAxis(3).d*i3-a2.o)/a2.d;
		}
	}
	_file->seekTo(0,0);
}

/*void data_rtm_3d::get_source_func_encode(hypercube_float *domain, int ishot, bool encode, int *rvec, float *s_z, float *s_x, float *s_y, int nsinc,  int nts,hypercube_float *time){

   // Read the data and set up the source geometry

   axis a1=domain->getAxis(1);
   axis a2=domain->getAxis(2);
   axis a3=domain->getAxis(3);
   axis at=getAxis(1);

   float sx=getAxis(4).o+getAxis(4).d*ishot;
   float sy=getAxis(5).o+getAxis(5).d*ishot;
   int ns=getAxis(4).n*getAxis(5).n;

   fprintf(stderr,"  RTM Data Details, encode=%d, nshots=%d \n",encode,ns);

   int i=0;

   // Encode and sum all shots into one supershot
   hypercube_float *tmp=(hypercube_float*)time->clone_zero();
   for(int is=0; is<ns; is++){
    sseek_block(myf->tagit.c_str(),getAxis(3).n*getAxis(2).n,getAxis(1).n*4*is,0);
    sreed(myf->tagit.c_str(),tmp->vals,getAxis(1).n*getAxis(2).n*getAxis(3).n*4);

    for(int k=0; k < (int) tmp->get_n123(); k++) time->vals[k] += rvec[is]*tmp->vals[k]/ns;
      fprintf(stderr,"    Data summing, %d %d \n",is,rvec[is]);


   }
   fprintf(stderr,"Shots summed \n");
    // Set up the regular source geometry RIGHT NOW THIS IS THE SAME FOR ALL SHOTS
    for(int i3=0; i3< (getAxis(3).n); i3++){
      for(int i2=0; i2< getAxis(2).n; i2++, i++){
        s_z[i]=(sz-a1.o)/a1.d;
        s_x[i]=(getAxis(2).o+getAxis(2).d*i2-a2.o)/a2.d;
        s_y[i]=(getAxis(3).o+getAxis(3).d*i3-a3.o)/a3.d;
      }
    }

   sseek(myf->tagit.c_str(),0,0);

   }*/
