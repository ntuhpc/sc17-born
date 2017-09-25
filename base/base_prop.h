#pragma once
#include <genericIO.h>
#include<cassert>
class baseProp {
public:
baseProp(){
	;
}

void storeIO(std::shared_ptr<SEP::genericIO> io){
	_io=io;
}
std::shared_ptr<SEP::paramObj> getParam(){
	return _io->getParamObj();
}
virtual void setNtblock(int nb){
	if(nb==0) ;

}
virtual void transferSincTableD(int nsinc, int jtd, std::vector<std::vector<float>>& table){
	if(nsinc==0 && jtd==0 ) ;
}
virtual void transferSourceFunc(int npts,int nt_big,std::vector<int> &locs, 
float * vals){
	if(npts==0 && nt_big==0) ;
}
virtual void transferVelFunc1(int nx, int ny, int nz, float *vloc){
	if(nx ==0 && ny==0 && nz==0 && vloc==0) ;

}
virtual void transferVelFunc2(int nx, int ny, int nz, float *vloc){
	if(nx ==0 && ny==0 && nz==0 && vloc==0) ;

}
virtual void sourceProp(int nx, int ny, int nz, bool damp, bool getLast,
	float *p0, float *p1, int jts, int npts, int nt){
	if(nx==ny && nz==0 && damp &&getLast && p0==0 && p1==0 && jts==0 && npts==0
	   && nt==0) ;
	   assert(1==2);
}
virtual void transferReceiverFunc(int nx, int ny, int nt, std::vector<int> &locs,
	float *ec){
	if(nx==0 && ny==0 && nt==0 ) ;
	assert(1==2);
}
virtual void rtmForward(int n1, int n2, int n3, int jt, float *img,
	float *rec, int npts, int nt, int nt_big, int rec_nx, int rec_ny){
	if(n1==0 && n2==0 && n3==0 && jt==0 && img==0 && rec==0 && npts==0
	   &&nt==0 && nt_big && rec_nx ==0 && rec_ny==0) ;

}
virtual void transferSincTableS(int nsinc, int jts, std::vector<std::vector<float>>& table){

	if(nsinc==0 && jts==0) ;

}
virtual void rtmAdjoint(int n1, int n2, int n3, int jtd, float *p0, float *p1,
	float *img, int npts_s, int nt){
	if(n1==0 && n2==0 && n3==0 && jtd==0 && p0==0 && p1==0 && img==0 && npts_s==0&&
	   nt==0) ;


}
virtual void createSpace(float d1, float d2, float d3,float bc_a, float bc_b, float bc_y,
	int nx, int ny, int nz){

	if(d1==0 && d2==0 && d3==0&& bc_a==0 && bc_b==0 && nx==0 && ny==0 && nz==0) ;

}

private:
std::shared_ptr<SEP::genericIO> _io;
};
