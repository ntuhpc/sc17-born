#include "source_func_3d.h"
#include <math.h>






void source_func::set_compute_size(std::shared_ptr<SEP::hypercube> dom,
	float ap,int nbt, int nb, int nby, int fat, int blocksize){
	
	
	ax=dom->getAxis(1);
	ay=dom->getAxis(2);
	az=dom->getAxis(3);

	nbound=nb+fat; nboundt=nbt; nbound_y=nby+fat;
	aper=ap;

	ay.n+=2*nbound_y;
	ax.n+=2*nbound;
	az.n+=2*nbound;                                                                                                                                                                                          //nbt+nb+2*fat;
	//az.n+=nbt+nb+2;//*fat;




	int rem_x=ax.n-((int) (ax.n/16))*16;
	int rem_y=ay.n-((int) (ay.n/16))*16;
	int rem_z=az.n-((int) (az.n/16))*16;

	if(rem_y!=0) rem_y=16-rem_y;
	if(rem_z!=0) rem_z=16-rem_z;
	if(rem_x!=0) rem_x=16-rem_x;

	//fprintf(stderr,"  Remainders are %d %d %d from %d %d %d. Padding z with %d + %d ; %d\n",rem_x,rem_y,rem_z,ax.n,ay.n,az.n,nbt,nb,nboundt);

	ay.n+=rem_y;
	ax.n+=rem_x;
	az.n+=rem_z;
//fprintf(stderr," Set comp %d %d %d , %f %f %f\n",ay.n,ax.n,az.n,ay.o,ax.o,az.o);
	/* ay.n=(int)(ap/ay.d)*2+1+2*nbound_y;
	   ax.n=(int)(ap/ax.d)*2+1+2*nbound;
	   az.n=az.n+nbt+nb+fat*2;
	   fprintf(stderr," Set comp %d %d %d , %f %f %f\n",ay.n,ax.n,az.n,ay.o,ax.o,az.o);

	   int t1=(ax.n-fat*2)/blocksize;
	   int rem=ax.n-t1*blocksize-2*fat;
	   if(rem!=0) rem=blocksize-rem;
	   if(rem!=0) ax.n+=+rem;

	   t1=(ay.n-fat*2)/blocksize;
	   rem=ay.n-t1*blocksize-2*fat;
	   if(rem!=0) rem=blocksize-rem;
	   if(rem!=0) ay.n+=+rem;

	   t1=(az.n-fat*2)/blocksize;
	   rem=az.n-t1*blocksize-2*fat;
	   if(rem!=0) rem=blocksize-rem;
	   if(rem!=0) az.n+=+rem;
	   fprintf(stderr," Set comp %d %d %d , %f %f %f\n",ay.n,ax.n,az.n,ay.o,ax.o,az.o);
	 */
//  az.o=-20.;//az.o;//+(nbt+fat)*az.d;
}

//hypercube_float *source_func::create_domain(int ishotx, int ishoty){
std::shared_ptr<hypercube_float> source_func::create_domain(int ishot){

	az.o+=-az.d*nbound;
	ax.o+=-ax.d*nbound;
	ay.o+=-ay.d*nbound;
	std::vector<SEP::axis>  axes; axes.push_back(ax); axes.push_back(ay); axes.push_back(az);
	std::shared_ptr<hypercube_float> tmp(new hypercube_float(axes));
	return tmp;
}

void wavelet_source_func::get_source_func(std::shared_ptr<hypercube_float> domain,
	int ishot, int nts, std::vector<int> &locs, std::vector<float> &time){
	SEP::axis a1=domain->getAxis(1);
	SEP::axis a2=domain->getAxis(2);
	SEP::axis a3=domain->getAxis(3);
	SEP::axis at=getAxis(1);

	float fz=(sz[ishot]-a3.o)/a3.d+.5;
	float fy=(sy[ishot]-a2.o)/a2.d+.5;
	float fx=(sx[ishot]-a1.o)/a1.d+.5;
	int iy=(int)fy;
	int ix=(int)fx;
	int iz=(int)fz;
	int i=0;

	for(int it=0; it < nts*9; it++) {
		time[it]=0;
	}
	for(int i3=iz-1; i3 <= iz-1; i3++) {
		float zn=fz-i3;
		for(int i2=iy-1; i2 <= iy+1; i2++) {
			float yn=fy-i2;
			for(int i1=ix-1; i1<= ix+1; i1++,i++) {
				float xn=fx-i1;
				locs[i]=i1+i2*a1.n+i3*a1.n*a2.n;
				float scale=expf(-zn*zn-xn*xn-yn*yn);
				for(int it=0; it < at.n; it++) {
					time[nts*i+it+4]=scale*wavelet->vals[it];
				}
			}
		}
	}

}

int wavelet_source_func::get_points(bool e ){
	return 9;
}
wavelet_source_func::wavelet_source_func(std::shared_ptr<SEP::genericIO> io, std::string tag){

	_io=io;
	tagInit(tag);
	std::shared_ptr<hypercube_float> tmp(new hypercube_float(getHyper()));

 wavelet.reset( new hypercube_float(getHyper()));

	readAll(tmp);
	wavelet->vals[getAxis(1).n-1]=0;
	for(int it=0; it < this->getAxis(1).n-1; it++) {
		wavelet->vals[it]=(tmp->vals[it+1]-tmp->vals[it])/tmp->getAxis(1).d;
	}

	dt=getAxis(1).d;

}

void wavelet_source_func::set_sources_axes(float s_z, SEP::axis src_axis1, SEP::axis src_axis2){

	for(int i2=0; i2 < src_axis2.n; i2++) {
		for(int i1=0; i1 < src_axis1.n; i1++) {
			sz.push_back(s_z);
			sx.push_back(src_axis1.o+src_axis1.d*i1);
			sy.push_back(src_axis2.o+src_axis2.d*i2);
		}
	}

}

wavefield_source_func::wavefield_source_func(std::shared_ptr<SEP::genericIO> io,std::string tag){
	_io=io;
	tagInit(tag);
	std::shared_ptr<hypercube_float> tmp(new hypercube_float(getHyper()));

	std::shared_ptr<hypercube_float>  wavefield(new hypercube_float(getHyper()));

	readAll(tmp);
	// wavelet->vals[getAxis(1).n-1]=0;
	long long nt=this->getAxis(1).n;
	long long ntraces=this->getHyper()->getN123()/this->getAxis(1).n;
	for(long long i=0; i < ntraces; i++) {
		for(long long it=0; it < nt-1; it++) {
			wavefield->vals[it+i*nt]=(tmp->vals[it+1+i*nt]-tmp->vals[it+i*nt])/tmp->getAxis(1).d;
		}
	}

	dt=getAxis(1).d;
}

void wavefield_source_func::get_source_func(std::shared_ptr<hypercube_float> domain, int ishot, int nts, std::vector<int> &locs, std::vector<float> &time){
	SEP::axis a1=domain->getAxis(1);
	SEP::axis a2=domain->getAxis(2);
	SEP::axis a3=domain->getAxis(3);
	SEP::axis at=getAxis(1);


	int iz=(sz[0]-a3.o)/a3.d+.5;
	int ipos=0;
	for(long long i=0; i < (long long) getAxis(3).n*(long long) getAxis(2).n*(nts); i++)
		time[i]=0;




	int nmin=std::min(at.n,nts-4);
	for(int i3=0; i3 < getAxis(3).n; i3++) {
		int iy=(int)(((getAxis(3).o+getAxis(3).d*i3-a2.o)/a2.d)+.5);
		for(int i2=0; i2 < getAxis(2).n; i2++,ipos++) {
			int ix=(int)(((getAxis(2).o+getAxis(2).d*i2-a1.o)/a1.d)+.5);
			locs[ipos]=ix+iy*a1.n+iz*a1.n*a2.n;
			// fprintf(stderr,"CHECK iz,ix,iy=%d,%d,%d (%f %f)(%f %f) \n",
			//   iz,ix,iy,getAxis(2).o,getAxis(2).d,getAxis(3).o,getAxis(3).d);
			for(int it=0; it <nmin; it++) {
				time[nts*ipos+it+4  ] =wavefield->vals[it+at.n*ipos];
			}
		}
	}
//assert(1==2);



}

int wavefield_source_func::get_points(bool e ){
	return this->getHyper()->getN123()/this->getAxis(1).n;
}
