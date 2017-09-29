#include "gpu_prop.h"
#include <cstring>
#include <vector>
#include "gpu/gpu_funcs_3d.h"

gpuProp::gpuProp(std::shared_ptr<SEP::genericIO> io) {
  storeIO(io);
  std::shared_ptr<SEP::paramObj> pars = getParam();
  int n_gpus = pars->getInt("n_gpus", 1);
  // setup_cuda(n_gpus,argc,argv);
  setup_cuda(n_gpus);
}

void gpuProp::setNtblock(int nb) { set_ntblock(nb); }
void gpuProp::transferSincTableD(int nsinc, int jtd,
                                 std::vector<std::vector<float>> &table) {
  std::vector<float *> table_ptrs(table.size());
  for (size_t i = 0; i < table.size(); ++i)
    table_ptrs.at(i) = table.at(i).data();
  transfer_sinc_table_d(nsinc, jtd, table_ptrs.data());
}
void gpuProp::transferSourceFunc(int npts, int nt_big, std::vector<int> &locs,
                                 float *vals) {
  transfer_source_func(npts, nt_big, locs.data(), vals);
}
void gpuProp::transferVelFunc1(int nx, int ny, int nz, float *vloc) {
  transfer_vel_func1(nx, ny, nz, vloc);  // REMINDER: vloc->vals
}
void gpuProp::transferVelFunc2(int nx, int ny, int nz, float *vloc) {
  transfer_vel_func2(nx, ny, nz, vloc);  // REMINDER: vloc->vals
}
void gpuProp::sourceProp(int nx, int ny, int nz, bool damp, bool getLast,
                         float *p0, float *p1, int jts, int npts, int nt) {
  source_prop(nx, ny, nz, damp, getLast, p0, p1, jts, npts, nt);
  int _n123 = nx * ny * nz;
  if (nt % 2 == 1) {
    memcpy(p1, p0, sizeof(float) * _n123);
  }
}
void gpuProp::transferReceiverFunc(int nx, int ny, int nt,
                                   std::vector<int> &locs, float *rec) {
  // transfer_receiver_func(rec_nx,rec_ny,nt_big,locs,rec_func->vals/*,s_z,s_x,s_y*/);
  transfer_receiver_func(nx, ny, nt, locs.data(), rec /*,s_z,s_x,s_y*/);
}
void gpuProp::rtmForward(int n1, int n2, int n3, int jt, float *img, float *rec,
                         int npts, int nt, int nt_big, int rec_nx, int rec_ny) {
  rtm_forward(n1, n2, n3, jt, img, rec, npts, nt, nt_big, rec_nx, rec_ny);
}
void gpuProp::rtmAdjoint(int n1, int n2, int n3, int jtd, float *p0, float *p1,
                         float *img, int npts_s, int nt) {
  // rtm_adjoint(ad1.n,ad2.n,ad3.n,jtd,src_p0->vals,src_p1->vals,img->vals,npts_s,nt/*,src,recx*/);
  rtm_adjoint(n1, n2, n3, jtd, p0, p1, img, npts_s, nt /*,src,recx*/);
}
void gpuProp::transferSincTableS(int nsinc, int jts,
                                 std::vector<std::vector<float>> &table) {
  std::vector<float *> table_ptrs(table.size());
  for (size_t i = 0; i < table.size(); ++i)
    table_ptrs.at(i) = table.at(i).data();
  transfer_sinc_table_s(nsinc, jts, table_ptrs.data());
}
void gpuProp::createSpace(float d1, float d2, float d3, float bc_a, float bc_b,
                          float bc_y, int nx, int ny, int nz) {
  create_gpu_space(d1, d2, d3, bc_a, bc_b, bc_y, nx, ny, nz);
}
