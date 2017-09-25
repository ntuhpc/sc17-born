#ifndef WAVEFIELD_INSERT_3D_H
#define WAVEFIELD_INSERT_3D_H 1
#include "oc_float.h"
class wavefield_insert_3d : public oc_float {
public:
wavefield_insert_3d();

virtual ~wavefield_insert_3d(){
};
virtual void insert_data_tile(int nx, float ox, float dx, int ny, float oy, float dy,int nz, float oz, float dz,  float *fld, float tm,int b1,int e1,int b2,int e2){
};
float dsamp;
int check_what;
virtual int get_points(){
	return 0;
}

};
#endif
