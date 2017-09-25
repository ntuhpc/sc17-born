#ifndef MY_VECTOR_H
#define MY_VECTOR_H 1
#include <string>
#include <stdio.h>
#include <memory>
#include <assert.h>
class my_vector {

public:
my_vector(){
};
virtual std::shared_ptr<my_vector>clone_vec(){
	return 0;
}
virtual std::shared_ptr<my_vector>clone_space(){
	return 0;
}
virtual std::shared_ptr<my_vector>clone_zero(){
	std::shared_ptr<my_vector>v=clone_vec();
	v->zero();
	return v;
}
virtual void allocate_space(){
};
virtual void scale(const double num){
	if(num==0) assert(0==1);
}
virtual void scale_add(const double mes, std::shared_ptr<my_vector>vec, const double other) {
	if(mes==0 || vec==0 || other==0) ;
	assert(0==1);
};

virtual void take_min(std::shared_ptr<my_vector>vec,std::shared_ptr<my_vector>other=0){
	if(vec==0 || other==0) ; assert(0==1);
}
virtual void take_max(std::shared_ptr<my_vector>vec,std::shared_ptr<my_vector>other=0){
	if (vec==0 || other==0) ; assert(0==1);
}
virtual void set_val(double val){
	if(val==val) ; assert(0==1);
}
virtual void add(std::shared_ptr<my_vector>vec){
	if(vec==0) ; assert(0==1);
};
virtual void mult(std::shared_ptr<my_vector>vec){
	if(vec==0) ; assert(0==1);
}
virtual double sum(){
	assert(1==0); return 0.;
}
virtual void scale_add(const double sc1, std::shared_ptr<my_vector>v1,double sc2,
	std::shared_ptr<my_vector>v2){
	scale_add(0.,v1,sc1);
	scale_add(1.,v2,sc2);
}
virtual double my_min(){
	assert(0==1);
}
virtual double my_max(){
	assert(0==1);
}
//virtual long long my_size(){asset(0==1);}
void zero(){
	this->set_val(0.);
}
virtual double dot(std::shared_ptr<my_vector>vec){
	assert(0==1); if(vec==0) return 0.;
};
virtual void info(char *str, int level=0){
	if(str==0 && level==0) ; assert(0==1);
};
virtual void random(){
	assert(0==1);
};
virtual void inverse_hessian(std::shared_ptr<my_vector>vec){
	if(vec==0) ; assert(0==1);
};
virtual ~my_vector(){
};
virtual bool check_match(const std::shared_ptr<my_vector>v2){
	if(name!=v2->name) return true;
	fprintf(stderr, "WARNING: TYPES DON'T MATCH %s %s \n",
		name.c_str(),v2->name.c_str());
	return false;
}
std::string name;





};
#endif
