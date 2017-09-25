#include "fd_prop_3d.h"
#include "sinc_bob.h"
//#include "gpu_funcs_3d.h"
#include <math.h>


bool fd_prop::calc_stability(std::vector<float> &ds,int n){

	vmax=vel->max_vel();
	float vmin=vel->min_vel();

	float dmin=vel->get_min_samp();

	float dmax=vel->get_max_samp();


	float d=.47*dmin/vmax;

	dt=ds[0]/ceilf(ds[0]/d);

//  dt=.001;
	nt=(n-1)*(int)(ds[0]/dt)+1;
	for(int i=0; i < (int)ds.size(); i++) d=ds[i]/dt-(int)(ds[i]/dt+.001);

/*
        if(fabs(d) >.01) {
                param->error(std::string("sampling match problem ")+std::to_string(d)+
                        std::string(" ")+std::to_string(dt)+std::string("\n"));
        }
 */
	fprintf(stderr,"Stability check vmax=%f d=%f dt=%f fmax=%f \n",vmax,dmin,dt,vmin/2.8/dmax);
	return true;

}

void fd_prop::create_transfer_sinc_source(int nsinc){
	jts=(int)((source->get_dt())/dt);
	sinc_bob mys(jts,nsinc);
	_prop->transferSincTableS(nsinc, jts, mys.table);
	//transfer_sinc_table_s(nsinc,jts,mys.table);
}

void fd_prop::set_fd_basics(std::shared_ptr<SEP::paramObj> par,
	std::shared_ptr<source_func> source_func, float ap,bool v){
	set_verb(v);
	aper=ap;
	param=par;
	float bc_a=par->getFloat("bc_a",50.);
	float bc_b=par->getFloat("bc_b",.0005);
	float bc_b_y=bc_b;
	source=source_func;
	nbound=par->getInt("nbound",(int)bc_a);
	nboundt=par->getInt("nboundt",nbound);
	int fat=4;
	int blocksize=16;

	create_transfer_sinc_source(8);

	source->set_compute_size(vel->_vel,aper,nbound,nbound,nbound,fat,blocksize);
	nx=source->x_points();
	nz=source->z_points();
	ny=source->y_points();
	_prop->createSpace(vel->getAxis(1).d,vel->getAxis(2).d,
		vel->getAxis(3).d,bc_a,bc_b,bc_b_y,nx,ny,nz);

}
