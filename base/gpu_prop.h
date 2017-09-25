#pragma once
#include "base_prop.h"
class gpuProp : public baseProp {
public:
gpuProp(std::shared_ptr<SEP::genericIO> io);

virtual void setNtblock(int nb);
virtual void transferSincTableD(int nsinc, int jtd, float **table);
virtual void transferSourceFunc(int npts,int nt_big,int *locs, float *vals);
virtual void sourceProp(int nx, int ny, int nz, bool damp, bool getLast,
	float *p0, float *p1, int jts, int npts, int nt);
virtual void transferVelFunc1(int nx, int ny, int nz, float *vloc);
virtual void transferVelFunc2(int nx, int ny, int nz, float *vloc);
virtual void transferReceiverFunc(int nx, int ny, int nt, int *locs,
	float *rec);
virtual void rtmForward(int n1, int n2, int n3, int jt, float *img,
	float *rec, int npts, int nt, int nt_big, int rec_nx, int rec_my);
virtual void rtmAdjoint(int n1, int n2, int n3, int jtd, float *p0, float *p1,
	float *img, int npts_s, int nt);
virtual void transferSincTableS(int nsinc, int jts, float **table);
virtual void createSpace(float d1, float d2, float d3,float bc_a, float bc_b, float bc_y,
	int nx, int ny, int nz);
};
