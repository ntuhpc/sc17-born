#ifndef CGSTEP_H
#define CGSTEP_H 1
#include <my_vector.h>
#include <step.h>
#include <stdio.h>
class cgstep : public step {
public:

virtual bool step_it(int iter,std::shared_ptr<my_vector> x,std::shared_ptr<my_vector>g, std::shared_ptr<my_vector>rr, std::shared_ptr<my_vector>gg, double *sc);
virtual void alloc_step(std::shared_ptr<my_vector>mod,std::shared_ptr<my_vector>dat);
void forget_it(){
	forget=true;
	s->zero(); ss->zero();
	small=1e-20;
}


private:
std::shared_ptr<my_vector> s,ss;
bool forget;
double small;
};


#endif
