#include <vel_fd_3d.h>
#include <data_rtm_3d.h>
#include <image_rtm_3d.h>
#include <ioModes.h>
#include "source_func_3d.h"
#include "rtm_zero_op_3d.h"
#include "cpu_prop.h"

main(int argc, char **argv){


	SEP::ioModes modes(argc,argv);



	std::shared_ptr<SEP::genericIO>  io=modes.getDefaultIO();
	std::shared_ptr<SEP::paramObj> pars=io->getParamObj();
	std::shared_ptr<data_rtm_3d> data(new data_rtm_3d("data",io));
	std::shared_ptr<wavelet_source_func> wavelet(new wavelet_source_func(io,"wavelet"));
	float src_depth=pars->getFloat("src_depth",0.);
	std::shared_ptr<vel_fd_3d> vel(new vel_fd_3d(io,"velocity"));
	std::shared_ptr<image_rtm_3d> image(new image_rtm_3d(io,"image",vel));
	SEP::axis asx=data->getAxis(4);
	SEP::axis asy=data->getAxis(5);


	wavelet->set_sources_axes(src_depth,asx,asy);




	float aper=pars->getFloat("aper",8.);

	std::vector<int> rand_vec(1.0);

	bool encode=false;
	std::shared_ptr<cpuProp> prop(new cpuProp(io));

	std::shared_ptr<rtm_zero_op> op(new
		rtm_zero_op(pars,prop,vel,wavelet,data,image,aper,true,encode,rand_vec,true));

   

	op->adjoint(false,image,data,1);
    image->write_final_volume();
	return 0;
}
