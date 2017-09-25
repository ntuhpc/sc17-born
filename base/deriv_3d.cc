#include "deriv_3d.h"
#include "float_3d.h"

bool deriv::adjoint(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data,int iter){

	std::shared_ptr<float_3d> m= std::dynamic_pointer_cast<float_3d>( model);
	std::shared_ptr<float_3d> d= std::dynamic_pointer_cast<float_3d>( data);
	if(!add) m->zero();

	SEP::axis a1=d->getAxis(1);
	SEP::axis a2=d->getAxis(2);
	SEP::axis a3=d->getAxis(3);
	for(int i3=0; i3 < a3.n; i3++) {
		m->vals [i3*a1.n*a2.n]+=(d->vals[i3*a1.n*a2.n+1] - d->vals[i3*a1.n*a2.n])/a1.d;
		for(int i2=0; i2 < a2.n; i2++) {
			m->vals [i2*a1.n+i3*a1.n*a2.n]+=(d->vals[i2*a1.n+i3*a1.n*a2.n+1] - d->vals[i2*a1.n+i3*a1.n*a2.n ])/a1.d;
			for(int i1=0; i1 < a1.n-1; i1++) {
				m->vals [i3*a1.n*a2.n+i2*a1.n+i1]+=(d->vals[i1+i2*a1.n+i3*a1.n*a2.n+1] - d->vals[i1+i2*a1.n+i3*a1.n*a2.n])/a1.d;
			}
		}
	}

}

bool deriv::forward(bool add, std::shared_ptr<my_vector> model, std::shared_ptr<my_vector>data,int iter){
	std::shared_ptr<float_3d> m= std::dynamic_pointer_cast<float_3d>( model);
	std::shared_ptr<float_3d> d= std::dynamic_pointer_cast<float_3d>( data);

	SEP::axis a1=d->getAxis(1);
	SEP::axis a2=d->getAxis(2);
	SEP::axis a3=d->getAxis(3);
	if(!add) d->zero();
	for(int i3=0; i3 < a3.n; i3++) {
		d->vals [1+i3*a1.n*a2.n]+= m->vals [i3*a1.n*a2.n]/a1.d;
		d->vals [i3*a1.n*a2.n]-= m->vals [i3*a1.n*a2.n]/a1.d;
		for(int i2=0; i2 < a2.n; i2++) {
			d->vals [1+i2*a1.n+i3*a1.n*a2.n]+= m->vals [i2*a1.n+i3*a1.n*a2.n]/a1.d;
			d->vals [i2*a1.n+i3*a1.n*a2.n]-= m->vals [i2*a1.n+i3*a1.n*a2.n]/a1.d;
			for(int i1=0; i1 < a1.n-1; i1++) {
				d->vals [i1+i2*a1.n+i3*a1.n*a2.n]-= m->vals [i3*a1.n*a2.n+i2*a1.n+i1]/a1.d;
				d->vals [i1+i2*a1.n+i3*a1.n*a2.n+1]+= m->vals [i3*a1.n*a2.n+i2*a1.n+i1]/a1.d;
			}
		}
	}

}
