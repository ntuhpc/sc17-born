#ifndef WAVELET_RTM_3D_H
#define WAVELET_RTM_3D_H 1
#include <hypercube_float.h>
#include "wavefield_insert_3d.h"
#include<cassert>
class source_func : public wavefield_insert_3d {
public:
source_func(){
};
source_func(std::shared_ptr<SEP::genericIO> io,std::string tag);



void set_source_file(std::string tg){
	tag=tg;
}
~source_func(){
}
std::shared_ptr<hypercube_float> create_domain(int ishot);
virtual int get_points(bool e){
	if(e) ; return 0;
}
void set_compute_size(std::shared_ptr<SEP::hypercube> dom, float aper,int nbt,
	int nb,int nby, int fat, int blocksize);
int y_points(){
	return ay.n;
}
int x_points(){
	return ax.n;
}
int z_points(){
	return az.n;
}
virtual void get_source_func(std::shared_ptr<hypercube_float> domain,
	int ishot,int nts, std::vector<int> &ilocs, std::vector<float> &vals){
	if(domain==0) ; if(ishot==0) ; if(nts==0) ; if(ilocs.size()==0) ;
	assert(1==2);
}
float get_dt(){
	return dt;
}
float dt;
std::vector<float> sx,sz,sy;
int nx,nz,ny,jt,jts,jtd;
float aper;
SEP::axis az,ax,ay;
int nboundt, nbound, nbound_y;
std::string tag;
};

class wavelet_source_func : public source_func {
public:
wavelet_source_func(){
};
wavelet_source_func(std::shared_ptr<SEP::genericIO> io,std::string tag);
void set_sz(float s_z){
	sz.push_back(s_z);
}
virtual int get_points(bool e);
virtual void get_source_func(std::shared_ptr<hypercube_float> domain, int ishot,
	int nts, std::vector<int >&ilocs,  std::vector<float> &vals);
~wavelet_source_func(){
}
void set_sources_axes(float s_z, SEP::axis src_axis1, SEP::axis src_axis2);

std::shared_ptr<hypercube_float> wavelet;


};
class wavefield_source_func : public source_func {
public:
wavefield_source_func(){
};
wavefield_source_func(std::shared_ptr<SEP::genericIO> io, std::string tag);
virtual int get_points(bool e);
void set_sources_depth(float s_z){
	sz.push_back(s_z);
}
virtual void get_source_func(std::shared_ptr<hypercube_float> domain, int ishot,int nts, std::vector<int> &ilocs, std::vector<float> &vals);

std::shared_ptr<hypercube_float> wavefield;

};

#endif
