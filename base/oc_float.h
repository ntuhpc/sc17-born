#ifndef oc_float_H
#define oc_float_H 1
//#include<cg.h>
#include <axis.h>
#include <hypercube.h>
#include <my_vector.h>
#include <genericIO.h>
#include "hypercube_float.h"
#include <genericFile.h>
class oc_float :  public my_vector {

public:
oc_float(){
};
virtual std::shared_ptr<my_vector>clone_vec();
virtual std::shared_ptr<my_vector>clone_space();
void check_same(std::shared_ptr<my_vector>other){
	if(other==0) ;
}
void zeroFile();
double dot(std::shared_ptr<my_vector>other);
void initNewFile(std::string tag, std::shared_ptr<SEP::genericIO> io,
	std::shared_ptr<SEP::hypercube> hyper,const SEP::usage_code usage);
virtual void scale_add(const double mes,  std::shared_ptr<my_vector>vec, const double other);
void add(std::shared_ptr<my_vector>other);
oc_float(std::shared_ptr<SEP::genericIO> io,std::string tag,std::vector<SEP::axis> axes,
	bool alloc=true);
oc_float(std::shared_ptr<SEP::genericIO> io,std::string tag,
	std::shared_ptr<SEP::hypercube> hyper);
oc_float(std::shared_ptr<SEP::genericIO> io,std::string tag,
	std::shared_ptr<oc_float> fle);
oc_float(std::shared_ptr<SEP::genericIO> io,std::string tag);
virtual void random();
virtual void scale(double r);
void setIO(std::shared_ptr<SEP::genericIO> io){
	_io=io;
}
virtual void mult(std::shared_ptr<my_vector>vec);
virtual double sum();
virtual void scale_add(const double sc1,std::shared_ptr<my_vector>v1,double sc2, std::shared_ptr<my_vector>v2);
virtual double my_min();
virtual double my_max();
std::shared_ptr<oc_float> clone(bool alloc=true, std::string tag="NONE");
void set_val(double val);
void normalize(float val);
void allocate();
void init_ndf(std::vector<SEP::axis> ax);

void deallocate();
virtual ~oc_float(){
	//if(temp)  this->deallocate();
}
std::string make_temp();
void tagInit(std::string tag);
virtual bool check_match(const std::shared_ptr<my_vector>v2);
void info(char *str,int level=0);
SEP::axis getAxis(int iax){
	return _file->getHyper()->getAxis(iax);
}
std::shared_ptr<SEP::hypercube> getHyper(){
	return _file->getHyper();
}
void readAll(std::shared_ptr<hypercube_float> hyper);
std::shared_ptr<SEP::genericRegFile> _file;
/***
   Return tag
 */
std::string getTag(){
	return _tag;
}
std::shared_ptr<SEP::genericIO> _io;

private:
bool temp;
std::string _tag;
};

 #endif
