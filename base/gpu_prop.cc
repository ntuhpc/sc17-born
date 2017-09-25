#include "gpu_prop.h"

gpuProp::gpuProp(std::shared_ptr<SEP::genericIO> io){
	storeIO(io);
	std::shared_ptr<SEP::paramObj> pars=getParam();
	int n_gpus=pars->getInt("n_gpus",1);
	//setup_cuda(n_gpus,argc,argv);
}
void gpuProp::setNtblock(int nb){

//set_ntblock(50);

}
void gpuProp::transferSincTableD(int nsinc, int jtd, float **table){
// transfer_sinc_table_d(nsinc,jtd,myr.table);



}
void gpuProp::transferSourceFunc(int npts,int nt_big,int *locs, float *vals){
	//transfer_source_func(npts,nt_big,locs,vals);

}
void gpuProp::transferVelFunc1(int nx, int ny, int nz, float *vloc){
//	transfer_vel_func_1(nx,ny,nz,vloc->vals);
}
void gpuProp::transferVelFunc2(int nx, int ny, int nz, float *vloc){
//	transfer_vel_func_2(nx,ny,nz,vloc->vals);
}
void gpuProp::sourceProp(int nx, int ny, int nz, bool damp, bool getLast,
	float *p0, float *p1, int jts, int npts, int nt){
//source_prop(nx,ny,nz,damp,getLast,p0,p1,jts,npts,nt);
}
void gpuProp::transferReceiverFunc(int nx, int ny, int nt, int *locs,
	float *rec){
//transfer_receiver_func(rec_nx,rec_ny,nt_big,locs,rec_func->vals/*,s_z,s_x,s_y*/);
}
void gpuProp::rtmForward(int n1, int n2, int n3, int jt, float *img,
	float *rec, int npts, int nt, int nt_big, int rec_nx, int rec_ny){

//	rtm_forward(n1,n2,n3,jt,img,rec,npts,nt,nt_big,rec_nx,rec_ny);

}
void gpuProp::rtmAdjoint(int n1, int n2, int n3, int jtd, float *p0, float *p1,
	float *img, int npts_s, int nt){
//    rtm_adjoint(ad1.n,ad2.n,ad3.n,jtd,src_p0->vals,src_p1->vals,img->vals,npts_s,nt/*,src,recx*/);
}
void gpuProp::transferSincTableS(int nsinc, int jts, float **table){
//transfer_sinc_table_s(nsinc,jts,table);
}
void gpuProp::createSpace(float d1, float d2, float d3,float bc_a, float bc_b, float bc_y,
	int nx, int ny, int nz){

	//create_gpu_space(d1,d2,d3,bc_a,bc_b,bc_b_y,nx,ny,nz);
}
