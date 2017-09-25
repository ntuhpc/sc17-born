#pragma once
#include "base_prop.h"
class cpuProp : public baseProp {
public:
cpuProp(std::shared_ptr<SEP::genericIO> io);


virtual void rtmForward(int n1, int n2, int n3, int jt, float *img,
	float *rec, int npts, int nt, int nt_big, int rec_nx, int rec_my);
virtual void rtmAdjoint(int n1, int n2, int n3, int jtd, float *p0, float *p1,
	float *img, int npts_s, int nt);
virtual void sourceProp(int nx, int ny, int nz, bool damp, bool getLast,
	float *p0, float *p1, int jts, int npts, int nt);
virtual void transferSincTableD(int nsinc, int jtd,std::vector<std::vector<float>> &table);
virtual void transferSourceFunc(int npts,int nt_big,std::vector<int> &locs, float *vals);

virtual void transferVelFunc1(int nx, int ny, int nz, float *vloc);
virtual void transferVelFunc2(int nx, int ny, int nz, float *vloc);
virtual void transferReceiverFunc(int nx, int ny, int nt, std::vector<int> &locs,
	float * rec);
virtual void transferSincTableS(int nsinc, int jts, std::vector<std::vector<float>> &table);
virtual void createSpace(float d1, float d2, float d3,float bc_a, float bc_b, float bc_y,
	int nx, int ny, int nz);
void prop(float *p0, float *p1, float *vel);
void injectSource(int id, int ii, float *p1);
void damp(float *p0, float *p1);
void imageCondition(float *src, float *rec, float *image);
void injectReceivers(int id, int ii, float *p1);
void dataExtract(int id, int ii, float *p);
void imageAdd(float *img,  float *recField, float *srcField);
void stats(float *buf, std::string title);
private:
int _nptsS;
int _jtdD,_nsincD;
std::vector<int> _locsR,_locsS;
float *_rec,*_sourceV;
float *_vel1,*_vel2;
std::vector<std::vector<float>> _tableS,_tableD;
int _nsincS,_jtsS;
int _ntRec,_ntSrc;
int _nRecs;
int _dir;
int _nx,_ny,_nz;   // Number of samples
int _n12;
int _jt;
float _bcA,_bcB,_bcY;   //Boundary condition
std::vector<float>  coeffs,_bound;
long long _n123;
FILE *myf;
};
