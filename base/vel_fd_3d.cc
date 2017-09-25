
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include "vel_fd_3d.h"
#include <math.h>
vel_fd_3d::vel_fd_3d(std::shared_ptr<SEP::genericIO> io, std::string tag){
	_io=io;
	_old=_mid=_zero=false;
	_dens=nullptr;
	_vFile=_io->getRegFile(tag,SEP::usageIn);
	_vel.reset(new hypercube_float(_vFile->getHyper()));
	_vFile->readFloatStream(_vel->vals,_vFile->getHyper()->getN123());


}

vel_fd_3d::vel_fd_3d(std::shared_ptr<SEP::genericIO> io, std::string tagV, std::string tagD){
	_io=io;
	_old=_mid=_zero=false;
	_vFile=_io->getRegFile(tagV,SEP::usageIn);
	_vel.reset(new hypercube_float(_vFile->getHyper()));
	_vFile->readFloatStream(_vel->vals,_vel->getN123());

	_dFile=_io->getRegFile(tagD,SEP::usageIn);
	_dens.reset(new hypercube_float(_dFile->getHyper()));
	_dFile->readFloatStream(_dens->vals,_dFile->getHyper()->getN123());

}

std::shared_ptr<hypercube_float> vel_fd_3d::rand_subspace(const int irand,const int nrand1,
	const int nrand2, const int nrand3, const float dt, std::shared_ptr<hypercube_float> space){
	float vmax=max_vel();
	srand(irand);

	std::vector<SEP::axis> axes; axes.push_back(space->getAxis(1));
	axes.push_back(space->getAxis(2)); axes.push_back(space->getAxis(3));

	std::shared_ptr<hypercube_float> tmp(new hypercube_float(axes));
	SEP::axis av_1=_vel->getAxis(1);
	SEP::axis av_2=_vel->getAxis(2);
	SEP::axis av_3=_vel->getAxis(3);
	
	float *temp=new float[axes[0].n*axes[1].n*axes[2].n];
	fprintf(stderr,"the size %d %d %d dt=%f \n",axes[0].n,axes[1].n,axes[2].n,dt);
		fprintf(stderr,"the size %d %d %d \n",av_1.n,av_2.n,av_3.n);


     tbb::parallel_for(tbb::blocked_range<int>(0,axes[2].n),[&](
  const tbb::blocked_range<int>&r){
  for(int  iz=r.begin(); iz!=r.end(); ++iz){
 	float dy2,dx2,dz2,rnd;
   int i=iz*axes[1].n*axes[0].n;
		int iz_v=std::max(0,std::min(av_3.n-1,(int)((axes[2].o+axes[2].d*iz-av_3.o)/av_3.d)));
		if(iz < nrand3) dz2=(float)((nrand3-iz)*(nrand3-iz))/(float)(nrand3*nrand3);
		else if(iz >= axes[2].n-nrand3)
			dz2=(float)((iz-(axes[2].n-nrand3-1))*(iz-(axes[2].n-nrand3-1)))/(float)(nrand3*nrand3);
		else dz2=0;
		for(int iy=0; iy < axes[1].n; iy++) {
			int iy_v=std::max(0,std::min(av_2.n-1,(int)((axes[1].o+axes[1].d*iy-av_2.o)/av_2.d)));
			if(iy < nrand2) dy2=(float)((nrand2-iy)*(nrand2-iy))/(float)(nrand2*nrand2);
			else if(iy >= axes[1].n-nrand2)
				dy2=(float)((iy-(axes[1].n-nrand2-1))*(iy-(axes[1].n-nrand2-1)))/(float)(nrand2*nrand2);
			else dy2=0;

			for(int ix=0; ix < axes[0].n; ix++,i++) {                                                                                                                                                                                                                                            // NOTE the i++
				int ix_v=std::max(0,std::min(av_1.n-1,(int)((axes[0].o+axes[0].d*ix-av_1.o)/av_1.d)));
				float val=_vel->vals[ix_v+iy_v*av_1.n+iz_v*av_1.n*av_2.n];
				if(ix < nrand1) dx2=(float)((nrand1-ix)*(nrand1-ix))/(float)(nrand1*nrand1);
				else if(ix >= axes[0].n-nrand1)
					dx2=(float)((ix-(axes[0].n-nrand1-1))*(ix-(axes[0].n-nrand1-1)))/(float)(nrand1*nrand1);
				else dx2=0;
				if(dx2<0. || dx2>1 ) { fprintf(stderr,"Vel problem! %f %f %f, %d %d %d; %d %d\n",dx2,dy2,dz2,ix,iy,iz,nrand1,axes[0].n); dx2=1; }
				float dist=sqrtf(dx2+dy2+dz2);
				bool found=false;
				float dev;
				if(dist<.0001) found=true;
				else{
					//  val=val*1.03*(1.-MIN(1.,dist)*.83);
				}
				dev=0.;

				while(!found && nrand2!=0) {
					rnd=(float)rand()/RAND_MAX-.5-.49*dist;
					dev=rnd*1.3*dist*val; 
					if(fabs(dev+val) < vmax*1.03*(1.-std::min(1.f,dist)*.4) && dev+val >0.0001) found=true;
				}


				temp[i]=val+dev;
				tmp->vals[i]=(val+dev)*(val+dev)*dt*dt;
			}
		}
	}
});
 // int ierr=fwrite(_vel->vals,4,axes[0].n*axes[1].n*axes[2].n,myf);

	delete []temp;
	return tmp;
}
