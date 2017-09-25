#include "map_data_3d.h"
#include "math.h"

map_data::map_data(int npt,std::vector<float> &s_x,std::vector<float>&s_y,
	std::vector<float> &s_z, std::vector<int> & locs,
	std::shared_ptr<hypercube_float>model,
	std::shared_ptr<hypercube_float> dom, std::shared_ptr<hypercube_float>
	ran,int ntbig){

	scale.resize(npt,1.);
	map=locs;
	int h=0;
	ntb=ntbig;
	int ia=0;
	int n1=model->getAxis(1).n;
	int n2=model->getAxis(2).n;


	for(int i4=0; i4 < npt; i4++) {

		int ix=(int)s_x[i4];
		int iz=(int)s_z[i4];
		int iy=(int)s_y[i4];

		float sm=0;
		float yn=s_y[i4]-iy;
		float xn=s_x[i4]-ix;

		int ia=i4;
		float zn=s_z[i4]-iz;
		zn=0.;
		scale[ia]=expf(-zn*zn-xn*xn-yn*yn);
		map[ia]=ix+iy*n1+iz*n1*n2;
	}
	locs=map;
	set_domain(dom);
	set_range(ran);
}


bool map_data::adjoint(bool add, std::shared_ptr<my_vector> model, std::shared_ptr<my_vector>data,int iter){

	std::shared_ptr<float_3d >d=std::dynamic_pointer_cast<float_3d> ( data);
	std::shared_ptr<float_3d> m=std::dynamic_pointer_cast<float_3d> (model);
	if(!add) m->zero();

	int nt=d->getAxis(1).n;
	int n2=d->getAxis(2).n;
	int n3=d->getAxis(3).n;

	for(int i3=0; i3 < n3; i3++) {
		for(int i2=0; i2 < n2; i2++) {
			for(int it=0; it < nt; it++) {
				m->vals[ntb*i2+it+4+i3*ntb*n2]+=d->vals[it+i2*nt+i3*nt*n2];
			}
		}
	}

}

bool map_data::forward(bool add, std::shared_ptr<my_vector> model, std::shared_ptr<my_vector>data,int iter){

	std::shared_ptr<float_3d >d=std::dynamic_pointer_cast<float_3d> ( data);
	std::shared_ptr<float_3d> m=std::dynamic_pointer_cast<float_3d> (model);
	if(!add) d->zero();

	int nt=d->getAxis(1).n;
	int n2=d->getAxis(2).n;
	int n3=d->getAxis(3).n;

	for(int i3=0; i3 < n3; i3++) {
		for(int i2=0; i2 < n2; i2++) {
			for(int it=0; it < nt; it++) {
				d->vals[it+i2*nt+i3*nt*n2]+=m->vals[ntb*i2+it+4+i3*ntb*n2];
			}
		}
	}

}
