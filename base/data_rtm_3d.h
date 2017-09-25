#include "hypercube_float.h"
#ifndef DATA_RTM_3D
#define DATA_RTM_3D 1
#include "wavefield_insert_3d.h"
class data_rtm_3d : public wavefield_insert_3d {
public:
data_rtm_3d(){
};
data_rtm_3d(std::string tg, std::shared_ptr<SEP::genericIO> io);
void get_shot_loc(int ishot, float loc1, float loc2, float loc3);
void read_shot(int shot);
void add_data(int ishot,std::shared_ptr<hypercube_float> dat);

void set_shot_num(int ishot){
	shot_num=ishot;
	//  load_shot();
}
void load_shot();
void clear_shot_data(){
	if(data!=0) delete [] data;
	if(rec_locs!=0) delete [] rec_locs;
	rec_locs=0;
	data=0;

}
virtual int get_points();
virtual int get_points1();
virtual int get_points2();


void get_source_func(std::shared_ptr<hypercube_float> domain, int ishot,
	std::vector<float>& s_x, std::vector<float>&s_y, std::vector<float>& s_z,
	int nsinc,  int nts, std::shared_ptr<hypercube_float> time);
//void get_source_func_encode(hypercube_float *domain, int ishot, bool encode, int *rvec, float *s_z, float *s_x, float *s_y, int nsinc,  int nts, hypercube_float *time);
void set_source_file(std::shared_ptr<oc_float> pntr){
	myf=pntr;
}

int nshotsx(){
	return getAxis(4).n;
}
int nshotsy(){
	return getAxis(5).n;
}
int nshots(){
	return getAxis(4).n*getAxis(5).n;
}



~data_rtm_3d(){

}


float sz;
int shot_num;
std::shared_ptr<hypercube_float> shot;
float *rec_locs;
float *data;
float rec_depth;
int nrec;
int nt;
int n11;
std::shared_ptr<oc_float> myf;
float max_time,dt;

};
#endif
