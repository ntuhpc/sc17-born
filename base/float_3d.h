#ifndef FLOAT_3D_H
#define FLOAT_3D_H
#include "hypercube_float.h"

class float_3d : public hypercube_float {

public:
float_3d(){
}                 //Default constructor
float_3d(SEP::axis a1, SEP::axis a2, SEP::axis a3);
float_3d(SEP::axis a1, SEP::axis a2, SEP::axis a3, float *array);
float_3d(int n1, int n2, int n3);
float_3d(int n1, int n2, int n3, float *array);
virtual ~float_3d(){
};
private:
void base_init_3df(SEP::axis a1,SEP::axis a2,SEP::axis a3);



};
																#endif
