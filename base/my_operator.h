#ifndef MY_OPERATOR_H
#define MY_OPERATOR_H 1
#include <cassert>
#include "stdio.h"
#include "my_vector.h"
class my_operator {
public:
my_operator(){
	basic_init_op();
}

void basic_init_op(){
	domain=0; range=0; solver_own=false; scale=1.;
}
bool check_domain_range(std::shared_ptr<my_vector> dom,std::shared_ptr<my_vector>range);
virtual bool forward(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data, int iter=1){
	return false;
}
virtual bool adjoint(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data, int iter=1){
	return false;
}
void set_domain(std::shared_ptr<my_vector>dom);
void set_range(std::shared_ptr<my_vector>ran);
virtual void hessian(std::shared_ptr<my_vector>in, std::shared_ptr<my_vector>out);
bool   dot_test(bool verb);
virtual void set_scale(float s){
	scale=s;
}
virtual std::shared_ptr<my_vector>range_vec(std::shared_ptr<my_vector>v1,
	std::shared_ptr<my_vector>v2){
	fprintf(stderr,"can not request range vec of non-combo op\n");
	assert(1==2);
	return 0;
}
virtual std::shared_ptr<my_vector>domain_vec(std::shared_ptr<my_vector>v1,
	std::shared_ptr<my_vector>v2){

	fprintf(stderr,"can not request domain vec of non-combo op\n");
	assert(1==2);
	return 0;
}



float scale;
std::shared_ptr<my_vector> domain,range;
bool solver_own;

};
#endif
