#include "tmute.h"
#include "oc_float.h"

bool tmute::adjoint(bool add,std::shared_ptr<my_vector> model, std::shared_ptr<my_vector>data){
	std::shared_ptr<oc_float> dat = std::dynamic_pointer_cast<oc_float>( data);
	std::shared_ptr<oc_float> mod = std::dynamic_pointer_cast<oc_float>( model);

	if(!add) mod->zero();
	SEP::axis a1=dat->getAxis(1);
	SEP::axis a2=dat->getAxis(2);
	SEP::axis a3=dat->getAxis(3);
	SEP::axis a4=dat->getAxis(4);
	SEP::axis a5=dat->getAxis(5);
	std::vector<SEP::axis> as;
	as.push_back(a1); as.push_back(a2); as.push_back(a3);
	std::shared_ptr<hypercube_float >d(new hypercube_float(as));
	std::shared_ptr<hypercube_float>m( new hypercube_float(as));
	dat->_file->seekTo(0,0);
	mod->_file->seekTo(0,0);
	long long n123=a1.n*a2.n*a3.n;
	for(int i5=0; i5 < a5.n; i5++) {
		for(int i4=0; i4 < a4.n; i4++) {
			dat->_file->readFloatStream(d->vals,n123);
			mod->_file->readFloatStream(d->vals,n123);
			for(int i3=0; i3 < a3.n; i3++) {

				for(int i2=0; i2 < a2.n; i2++) {
					float t0=t_0+(a2.o+a2.d*i2)/vmute;
					for(int i1=(int)((t0-a1.o)/a1.d); i1 < a1.n; i1++) {
						m->vals [i3*a1.n*a2.n+i2*a1.n+i1]+=d->vals[i3*a1.n*a2.n+i2*a1.n+i1];
					}
				}
			}
			mod->_file->seekTo(-n123,1);
			mod->_file->writeFloatStream(m->vals,n123);

		}
	}

}
bool tmute::forward(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector> data){
	std::shared_ptr<oc_float> dat = std::dynamic_pointer_cast<oc_float>( data);
	std::shared_ptr<oc_float> mod = std::dynamic_pointer_cast<oc_float>( model);
	SEP::axis a1=dat->getAxis(1);
	SEP::axis a2=dat->getAxis(2);
	SEP::axis a3=dat->getAxis(3);
	SEP::axis a4=dat->getAxis(4);
	SEP::axis a5=dat->getAxis(5);
	std::vector<SEP::axis> as; as.push_back(a1); as.push_back(a2); as.push_back(a3);
	std::shared_ptr<hypercube_float>d(new hypercube_float(as));
	std::shared_ptr<hypercube_float>m(new hypercube_float(as));

	if(!add) dat->zero();
	dat->_file->seekTo(0,0);
	mod->_file->seekTo(0,0);
	long long n123=a1.n*a2.n*a3.n;
	for(int i5=0; i5 < a5.n; i5++) {
		for(int i4=0; i4 < a4.n; i4++) {
			for(int i3=0; i3 < a3.n; i3++) {
				dat->_file->readFloatStream(d->vals,n123);
				mod->_file->readFloatStream(d->vals,n123);
				for(int i2=0; i2 < a2.n; i2++) {
					float t0=t_0+(a2.o+a2.d*i2)/vmute;
					for(int i1=(int)((t0-a1.o)/a1.d); i1 < a1.n; i1++) {
						d->vals [i3*a1.n*a2.n+i2*a1.n+i1]+=m->vals[i3*a1.n*a2.n+i2*a1.n+i1];
					}
				}
			}
			dat->_file->seekTo(-n123,1);
			dat->_file->writeFloatStream(d->vals,n123);
		}
	}

}
