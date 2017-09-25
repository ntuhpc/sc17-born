#ifndef FLOAT_1D_H
#define FLOAT_1D_H
#include <hypercube_float.h>


class float_1d : public hypercube_float {

public:
float_1d(){
}                 //Default constructor
float_1d(SEP::axis a1);
float_1d(SEP::axis a1,  float *array);
float_1d(int n1);
float_1d(int n1, float *array);
virtual ~float_1d(){
};
private:
void base_init_1df(SEP::axis a1);



};
#endif
