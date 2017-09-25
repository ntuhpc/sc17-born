#ifndef MAP_DATA_3D_H
#define MAP_DATA_3D_H 1
#include "my_operator.h"
#include "source_func_3d.h"
#include "float_3d.h"
class map_data : public my_operator {
public:
map_data(){
};
map_data(int npt,std::vector<float> &s_x,std::vector<float >&s_y, std::vector<float >&s_z,
	std::vector<int> &locs,std::shared_ptr<hypercube_float> model,
	std::shared_ptr<hypercube_float> dom, std::shared_ptr<hypercube_float> ran,int ntbig);

virtual bool forward(bool add, std::shared_ptr<my_vector> model, std::shared_ptr<my_vector>data,int iter=0);
virtual bool adjoint(bool add, std::shared_ptr<my_vector>model, std::shared_ptr<my_vector>data,int iter=0);



std::vector<float >scale;
std::vector<int> map;
int ntb;

};
#endif
