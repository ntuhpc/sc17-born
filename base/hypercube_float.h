#ifndef HYPERCUBE_FLOAT_H
#define HYPERCUBE_FLOAT_H 1
#include <axis.h>
#include <hypercube.h>
#include <my_vector.h>

class hypercube_float : public SEP::hypercube, public my_vector {

public:
hypercube_float()
{
	this->vals=0;
}                     //Default

virtual std::shared_ptr<my_vector>clone_vec();
virtual std::shared_ptr<my_vector>clone_space();
void check_same(std::shared_ptr<my_vector>other){
	if(other==0) ;
}
double dot( std::shared_ptr<my_vector>other);
virtual void scale_add(const double mes,  std::shared_ptr<my_vector>vec, const double other);
void add( std::shared_ptr<my_vector>other);
hypercube_float(std::vector<SEP::axis> axes,bool alloc=true);
hypercube_float(std::vector<SEP::axis> axes, float *vals);
hypercube_float(std::shared_ptr<SEP::hypercube> hyper);
virtual void random();
virtual void scale(double r){
	for(int i=0; i< getN123(); i++) vals[i]=vals[i]*r;
}
void add(std::shared_ptr<hypercube_float>vec);
virtual void take_min(std::shared_ptr<my_vector>vec,std::shared_ptr<my_vector>other=0);
virtual void take_max(std::shared_ptr<my_vector>vec,std::shared_ptr<my_vector>other=0);
virtual void mult(std::shared_ptr<my_vector>vec);
virtual double sum();
virtual void scale_add(const double sc1, std::shared_ptr<my_vector>v1,double sc2, std::shared_ptr<my_vector>v2);


virtual double my_min();
virtual double my_max();
std::shared_ptr<hypercube_float>clone(bool alloc=true);
void set(float *vals);
void set_val(double val);
void normalize(float val);
void allocate(){
	if(this->vals!=0) deallocate();
	this->vals=new float[this->getN123()];
}
void init_ndf(std::vector<SEP::axis> ax){
	initNd(ax); allocate();
}


void deallocate(){
	if(this->vals!=0) delete []this->vals;
}
virtual ~hypercube_float(){
	this->deallocate();
}

virtual bool check_match(const std::shared_ptr<my_vector> v2);

void info(char *str,int level=0);
float *vals;



private:
};

 #endif
