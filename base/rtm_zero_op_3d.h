#ifndef RTM_OP_3D_H
#define RTM_OP_3D_H 1
#include "base_prop.h"
#include "float_1d.h"
#include "my_operator.h"
#include "data_rtm_3d.h"
#include "image_rtm_3d.h"
#include "fd_prop_3d.h"
#include "source_func_3d.h"
class rtm_zero_op : public my_operator, public fd_prop {
public:
rtm_zero_op(){
	basic_init_op();
}
rtm_zero_op(std::shared_ptr<SEP::paramObj> par,
	std::shared_ptr<baseProp> prop, std::shared_ptr<vel_fd_3d>vel,
	std::shared_ptr<source_func>source_func,
	std::shared_ptr<data_rtm_3d> data_insert, std::shared_ptr<image_rtm_3d>image,
	float aper, bool verb,
	bool encode, std::vector<int > rand_vec, bool do_src=true,bool redo_src=false);
void create_source_fields();

virtual bool forward(bool add, std::shared_ptr<my_vector> model, std::shared_ptr<my_vector> data, int iter);
virtual bool adjoint(bool add, std::shared_ptr<my_vector> model, std::shared_ptr<my_vector> data, int iter);
~rtm_zero_op(){
	delete_rtm_op();
}
void delete_rtm_op(){
};
void create_transfer_sinc_data( int nsinc);
void create_random_trace(int nshots, int *rand_vec);
bool encode;

private:
std::shared_ptr<data_rtm_3d>data;
std::shared_ptr<image_rtm_3d>image;
std::vector<int> rand_vec;

int jtd;
float dtd;
bool redo;
int base;
std::vector<float> slice_p0;

};
#endif
