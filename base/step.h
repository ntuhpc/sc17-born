#ifndef STEP_H
#define STEP_H 1
#include <my_vector.h>

class step {
public:
step(){
};
virtual void alloc_step(std::shared_ptr<my_vector>mod, std::shared_ptr<my_vector>dat){
}
virtual bool step_it(int iter, std::shared_ptr<my_vector>x,
	std::shared_ptr<my_vector>g, std::shared_ptr<my_vector>rr, std::shared_ptr<my_vector>gg,double *sc){
	return false;
}
};

#endif
