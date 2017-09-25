#ifndef FDPROP_3D_H
#define FDPROP_3D_H 1
#include "vel_fd_3d.h"
#include "source_func_3d.h"
#include "paramObj.h"
#include "base_prop.h"
class fd_prop {
public:


fd_prop(){
	;
}

bool  calc_stability(std::vector<float> & ds,int n);
void setProp(std::shared_ptr<baseProp> prop){
	_prop=prop;
};
void set_vel(std::shared_ptr<vel_fd_3d> v){
	vel=v;
}

void set_verb(bool v){
	verb=v;
}
void set_bounds(int n1, int n2, int n3){
	nboundt=n1; nbound=n2, nbound_y=n3;
}
void set_fd_basics(std::shared_ptr<SEP::paramObj> par, std::shared_ptr<source_func> source_func, float ap,bool v);
void create_transfer_sinc_source(int nsinc);

std::shared_ptr<vel_fd_3d> vel;
float dt;
float vmax;
int nt;
int nz,nx,ny;
int nboundt,nbound,nbound_y;
std::shared_ptr<source_func> source;
std::shared_ptr<SEP::paramObj> param;
float aper;
int jts;
float dtw;
bool verb;
std::shared_ptr<baseProp> _prop;


};
#endif
