#ifndef DERIV_3D_H
#define DERIV_3D_H 1
#include "my_operator.h"
#include "source_func_3d.h"
class deriv : public my_operator {
public:
deriv(std::shared_ptr<hypercube_float>mod, std::shared_ptr<hypercube_float>dat){
	set_domain(mod); set_range(dat);
};

~deriv(){
};
virtual bool forward(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data,int iter=0);
virtual bool adjoint(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data,int iter=0);



};
#endif
