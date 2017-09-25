#include "rtm_zero_op_3d.h"
#include "sinc_bob.h"
//#include "gpu_funcs_3d.h"
#include <math.h>
#include "deriv_3d.h"
#include "laplac_3d.h"
#include "map_data_3d.h"
rtm_zero_op::rtm_zero_op(std::shared_ptr<SEP::paramObj> par,
	std::shared_ptr<baseProp> prop,
	std::shared_ptr<vel_fd_3d>vel_3d, std::shared_ptr<source_func>source_func,
	std::shared_ptr<data_rtm_3d> dat, std::shared_ptr<image_rtm_3d>img,
	float ap,bool v,bool enc,std::vector<int> r_vec, bool do_src,bool redo_src){
	std::vector<float> ds;
	data=dat;
	setProp(prop);
	ds.push_back(dat->dt);

	ds.push_back(source_func->get_dt());

	set_vel(vel_3d);

	calc_stability(ds,data->nt);
	set_fd_basics(par,source_func,ap,v);
	_prop->setNtblock(50);

	image=img;

	//set_domain(image);
//	set_range(data);
	dtd=data->dt;
	jtd=(int)(dtd/dt);
	nt=(data->nt-1)*jtd+1;
	base=0;
	redo=redo_src;
	create_transfer_sinc_data(8);

	encode=enc;
	rand_vec=r_vec;
	//if(do_src && !redo) create_source_fields();
}

void rtm_zero_op::create_transfer_sinc_data(int nsinc){
	jtd=(int)((data->dt)/dt);
	sinc_bob myr(jtd,nsinc);
	_prop->transferSincTableD(nsinc,jtd,myr.table);

}

void rtm_zero_op::create_source_fields(){

	int nshots=data->nshots();

	int npts,nt_big;
	nt_big=9+data->nt;
	npts=source->get_points(encode);
	std::vector<int> locs(npts,0);
	std::vector<float> vals(npts*nt_big,0.);

	int seed=0;
	//nshots=1;

	for(int ishot=0; ishot< nshots; ishot++) {

		if(verb) fprintf(stderr,"Forward apropagating shot %d of %d \n",ishot,nshots);
		std::shared_ptr<hypercube_float>src_p0=source->create_domain(ishot);
		std::shared_ptr<hypercube_float>src_p1=src_p0->clone();
		if(verb) fprintf(stderr,"Forward bpropagating shot %d of %d \n",ishot,nshots);

		source->get_source_func(src_p0,ishot,nt_big,locs,vals);
		_prop->transferSourceFunc(npts, nt_big, locs, vals.data());

		if(verb) fprintf(stderr,"Forward 0propagating shot %d of %d \n",ishot,nshots);

		seed=locs[3] - ((int) locs[3]/10000)*10000;

		std::shared_ptr<hypercube_float> vloc=
			vel->rand_subspace(seed,nbound,nbound,nbound,dt,src_p0);

		if(verb) fprintf(stderr,"Forward 1propagating shot %d of %d \n",ishot,nshots);

		_prop->transferVelFunc1(nx,ny,nz,vloc->vals);
		time_t startwtimes, endwtimes;
		startwtimes = time(&startwtimes);
		if(verb) fprintf(stderr,"Forward2 propagating shot %d of %d \n",ishot,nshots);

		_prop->sourceProp(nx,ny,nz,false,true,src_p0->vals,src_p1->vals,jts,npts,nt);
		if(verb) fprintf(stderr,"Forward3 propagating shot %d of %d \n",ishot,nshots);

		endwtimes = time(&endwtimes);


	}

}

bool rtm_zero_op::forward(bool add, std::shared_ptr<my_vector>mvec,
	std::shared_ptr<my_vector> dvec, int iter){

	std::shared_ptr<data_rtm_3d> d=std::dynamic_pointer_cast<data_rtm_3d> (dvec);
	std::shared_ptr<image_rtm_3d> m=std::dynamic_pointer_cast<image_rtm_3d>( mvec);

	if(!add) dvec->zero();

	int npts_s=source->get_points(encode);

	//data->set_source_file(d);
	//image->set_source_file(m);

	int nt_big=8+data->nt;

	int npts=data->get_points();
	int rec_nx=data->get_points1();
	int rec_ny=data->get_points2();

	std::vector<int> locs(npts,0);
	std::vector<int> locs_s(npts_s,0);
	std::vector<float> vals(npts_s*nt_big,0.);
	std::vector<float> s_z(npts,0.),s_x(npts,0.),s_y(npts,0.);

	SEP::axis ax_pt1(rec_nx);
	SEP::axis ax_pt2(rec_ny);

	std::shared_ptr<float_3d> shot (new float_3d(data->getAxis(1),
			data->getAxis(2),data->getAxis(3)));
	std::shared_ptr<float_3d> shot_deriv(new float_3d(data->getAxis(1),
			data->getAxis(2),data->getAxis(3)));

	SEP::axis ax_t=SEP::axis(nt_big);
	std::shared_ptr<float_3d> rec_func(new float_3d(ax_t,ax_pt1,ax_pt2));
	int nshots=data->nshots();
	//nshots=1;
	for(int ishot=0; ishot<nshots; ishot++) {

		if(verb) fprintf(stderr,"Forward shot %d of %d\n",ishot,data->nshots());
		std::shared_ptr<hypercube_float>src_p0=source->create_domain(ishot);

		std::shared_ptr<hypercube_float>src_p1=src_p0->clone();

		SEP::axis ad1=src_p0->getAxis(1);
		SEP::axis ad2=src_p0->getAxis(2);
		SEP::axis ad3=src_p0->getAxis(3);

		source->get_source_func(src_p0,ishot,nt_big,locs_s,vals);


		_prop->transferSourceFunc(npts_s,nt_big,locs_s,vals.data());

		int seed=locs_s[3] - ((int) locs_s[3]/10000)*10000;
fprintf(stderr,"before rand \n");
		std::shared_ptr<hypercube_float> vrand=vel->rand_subspace(seed,nbound,nbound,nbound,dt,src_p0);
		fprintf(stderr,"before none \n");

		std::shared_ptr<hypercube_float>vnone=vel->rand_subspace(seed,0,0,0,dt,src_p0);
fprintf(stderr,"before extract image \n");

		std::shared_ptr<hypercube_float>img=image->extract_sub(ad1,ad2,ad3);

		_prop->transferVelFunc1(nx,ny,nz,vrand->vals);
		_prop->transferVelFunc2(nx,ny,nz,vnone->vals);
fprintf(stderr,"after transfer \n");


		data->get_source_func(src_p0,ishot,s_x,s_y,s_z,8,nt_big,shot);

		map_data mapit(npts,s_x,s_y,s_z,locs,src_p0,rec_func,shot_deriv,nt_big);

		rec_func->zero();


		_prop->transferReceiverFunc(rec_nx, rec_ny, nt_big, locs,
			rec_func->vals);

		std::shared_ptr<hypercube_float>img_lap=img->clone();

		laplac lap=laplac(img_lap,img);

		lap.forward(false,img_lap,img);

fprintf(stderr,"bevore forward \n");

		_prop->rtmForward(ad1.n,ad2.n,ad3.n,jtd,img->vals,rec_func->vals,npts_s,nt,nt_big,rec_nx,rec_ny);

		deriv der=deriv(shot_deriv,shot);

		mapit.forward(false,rec_func,shot_deriv);

		der.forward(false,shot_deriv,shot);




		data->add_data(ishot,shot);


	}



	return true;

}

bool rtm_zero_op::adjoint(bool add, std::shared_ptr<my_vector>mvec,
	std::shared_ptr<my_vector> dvec, int iter){



	if(!add) {
		mvec->zero();
	}

	std::shared_ptr<data_rtm_3d> d=std::dynamic_pointer_cast<data_rtm_3d> (dvec);
	std::shared_ptr<image_rtm_3d> m=std::dynamic_pointer_cast<image_rtm_3d>( mvec);
	int npts_s=source->get_points(encode);

	int nt_big=8+data->nt;
	int npts=data->get_points();
	int rec_nx=data->get_points1();
	int rec_ny=data->get_points2();

	float dtd=data->dt;
	int jtd=(int)(dtd/dt);
	int nt=(data->nt-1)*jtd+1;
	std::vector<int> locs(npts,0),locs_s(npts_s,0);
	std::vector<float> vals(npts_s*nt_big,0.),s_x(npts,0.),
	s_z(npts,0.),s_y(npts,0.);

	SEP::axis ax_pt1(rec_nx);
	SEP::axis ax_pt2(rec_ny);

	std::shared_ptr<float_3d> shot(new float_3d(data->getAxis(1),
			data->getAxis(2),data->getAxis(3)));
	std::shared_ptr<float_3d> shot_deriv(new float_3d(data->getAxis(1),
			data->getAxis(2),data->getAxis(3)));
	SEP::axis ax_t=SEP::axis(nt_big);

	std::shared_ptr<float_3d> rec_func(new float_3d(ax_t,ax_pt1,ax_pt2));
	int seed=0;

	int nts=(int)(nt/jts)+1;
	int nshots=data->nshots();
	int nptsS=source->get_points(false);

	//nshots=1;

	for(int ishot=0; ishot<nshots; ishot++) {
		if(verb) fprintf(stderr,"Adjoint shot %d of %d \n",ishot,data->nshots());
		std::shared_ptr<hypercube_float>  src_p0=source->create_domain(ishot);
		std::shared_ptr<hypercube_float> src_p1=src_p0->clone();
		SEP::axis ad1=src_p0->getAxis(1);
		SEP::axis ad2=src_p0->getAxis(2);
		SEP::axis ad3=src_p0->getAxis(3);


		source->get_source_func(src_p0,ishot,nt_big,locs_s,vals);
		data->get_source_func(src_p0,ishot,s_x,s_y,s_z,8,nt_big,shot);

		_prop->transferSourceFunc(npts_s,nt_big,locs_s,vals.data());


		seed=locs_s[3] - ((int) locs_s[3]/10000)*10000;

		std::shared_ptr<hypercube_float>vrand=vel->rand_subspace(seed,nbound,nbound,nbound,dt,src_p0);
		//hypercube_float *vnone=vel->rand_subspace(seed,0,0,0,dt,src_p0);
		std::shared_ptr<hypercube_float>img=src_p0->clone();
        std::shared_ptr<hypercube_float> vnone=vel->rand_subspace(seed,0,0,0,dt,src_p0);
		_prop->transferVelFunc1(nx,ny,nz,vrand->vals);
		_prop->transferVelFunc2(nx,ny,nz,vnone->vals);

		deriv der=deriv(shot_deriv,shot);
		der.adjoint(false,shot_deriv,shot);
		map_data mapit(npts,s_x,s_y,s_z,locs,src_p0,rec_func,shot_deriv,nt_big);
		mapit.adjoint(false,rec_func,shot_deriv);

		_prop->transferReceiverFunc(rec_nx,rec_ny,nt_big,locs,rec_func->vals);
		_prop->sourceProp(ad1.n,ad2.n,ad3.n,false,true,src_p0->vals,src_p1->vals,jts,nptsS,nt);

 
		time_t startwtimem, endwtimem;
		startwtimem = time(&startwtimem);

		_prop->rtmAdjoint(ad1.n,ad2.n,ad3.n,jtd,src_p1->vals,src_p0->vals,img->vals,npts_s,nt);


		endwtimem = time(&endwtimem);

		fprintf(stderr,"Adjoint, wall clock time = %f \n", difftime(endwtimem, startwtimem));

		std::shared_ptr<hypercube_float> img_lap=img->clone();
		laplac lap=laplac(img_lap,img);
		lap.adjoint(false,img_lap,img);
		//lap.dot_test(true);

		//image->add_image(img);
		image->add_image(img_lap);


	}




	return true;
}
