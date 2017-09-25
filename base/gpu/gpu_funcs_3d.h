#ifndef GPU_FUNCS_3D_H
#define GPU_FUNCS_3D_H 1
//dont forget the img kernel bits later!
extern "C"{

void setup_cuda(int device, int argc, char **argv);//no change needed

void source_prop(int n1, int n2, int n3, bool damp, bool get_last, float *p0, float *p1, int jt, int npts,int nt);

void rtm_adjoint(int n1, int n2,  int n3, int j, float *p0_s_cpu, float *p1_s_cpu, float *img, int npts,int nt); //done
void set_ntblock(int nt);
void rtm_forward(int n1, int n2, int n3, int jt, float *img, float *dat, int npts_src,int nt,int nt_big, int nx, int ny);

void transfer_sinc_table_s(int nsinc, int ns,  float **tables); //cleaned up, no y dim needed

void transfer_sinc_table_d(int nsinc, int ns,float **tabled); //nothing to do

void transfer_source_func(int npts, int nt, int *locs, float *vals); //(think) nothing to do
void load_source(int it);
void transfer_receiver_func(int nx, int ny, int nt, int *locs, float *vals/*, float *s_z, float *s_x, float *s_y*/); //(think) nothing to do

void transfer_vel_func1(int nx, int ny, int nz, float *vel); //done

void transfer_vel_func2(int nx, int ny, int nz, float *vel); //done

void create_gpu_space(float d1, float d2, float d3, float bc_a, float bc_b, float bc_b_y, int nx, int ny, int nz); //done
};
#endif
