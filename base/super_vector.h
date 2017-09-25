#ifndef SUPER_VECTOR_H
#define SUPER_VECTOR_H 1
#include "my_vector.h"
#include <vector>
class super_vector : public my_vector {
public:
super_vector(){
};
super_vector(std::shared_ptr<my_vector>v1, std::shared_ptr<my_vector>v2,bool sp=false);
super_vector(std::shared_ptr<my_vector>v1, std::shared_ptr<my_vector>v2, std::shared_ptr<my_vector>v3,bool sp=false);
super_vector(std::vector<std::shared_ptr<my_vector> > vs,bool sp=false);
void add_vector(std::shared_ptr<my_vector> v1);
std::shared_ptr<my_vector>return_vector(int ivec);
virtual void scale(const double num);
virtual void scale_add(const double mes, std::shared_ptr<my_vector>vec, const double other);
virtual void set_val(double val);
virtual void add(std::shared_ptr<my_vector>vec);
virtual double dot(std::shared_ptr<my_vector>vec);
virtual void info(char *str,int level=0);
virtual void random();
virtual void inverse_hessian(std::shared_ptr<my_vector>vec);
std::shared_ptr<my_vector> clone_vec(){
	return clone_it(false);
}
std::shared_ptr<my_vector>clone_space(){
	return clone_it(true);
}
std::shared_ptr<my_vector>clone_it(bool alloc);

virtual bool check_match(const std::shared_ptr<my_vector>v2);
void super_init(std::string n){
	name=n;
}




bool just_space;
std::vector<std::shared_ptr<my_vector> > vecs;



};

#endif
