#include "image_rtm_3d.h"
#include "deriv_3d.h"

image_rtm_3d::image_rtm_3d(std::shared_ptr<SEP::genericIO> io,std::string tg,std::shared_ptr<vel_fd_3d>v){
	imgHyper=v->_vFile->getHyper();
	image.reset( new hypercube_float(imgHyper));
	std::vector<SEP::axis> axes;
	initNewFile(tg,io,imgHyper,SEP::usageInOut);

 fprintf(stderr,"in init \n");
	image->zero();
	fprintf(stderr,"should have zeroed \n");
//  hypercube_float *mute=(hypercube_float*)image->clone_zero();
//  sreed("mute",mute->vals,4*get_axis(1).n*get_axis(2).n*get_axis(3).n);
}

void image_rtm_3d::set_source_file(std::shared_ptr<oc_float> ptr){
//	myf=ptr;
	//myf->_file->seekTo(0,0);
//	myf->_file->readFloatStream(image->vals,imgHyper->getN123());


}

void image_rtm_3d::add_image(std::shared_ptr<hypercube_float>img){
	SEP::axis a1o=imgHyper->getAxis(1);
	SEP::axis a2o=imgHyper->getAxis(2);
	SEP::axis a3o=imgHyper->getAxis(3);
	SEP::axis a1i=img->getAxis(1);
	SEP::axis a2i=img->getAxis(2);
	SEP::axis a3i=img->getAxis(3);

	int f3=(int)((a3i.o-a3o.o)/a3o.d);
	int f2=(int)((a2i.o-a2o.o)/a2o.d);
	int f1=(int)((a1i.o-a1o.o)/a1o.d);

	for(int i3=0; i3 < a3i.n; i3++) {
		if(i3+f3 >=0 && i3+f3 < a3o.n) {
			for(int i2=0; i2 < a2i.n; i2++) {
				if(i2+f2 >=0 && i2+f2 < a2o.n) {
					for(int i1=0; i1 < a1i.n; i1++) {
						if(i1+f1>=0 && i1+f1 <a1o.n) {
							image->vals[i1+f1+(i2+f2)*a1o.n+(i3+f3)*a1o.n*a2o.n]+=img->vals[i1+i2*a1i.n+i3*a1i.n*a2i.n];
						}
					}
				}
			}
		}
	}

}

std::shared_ptr<float_3d> image_rtm_3d::extract_sub(SEP::axis a1i, SEP::axis a2i,
	SEP::axis a3i){
	SEP::axis a1o=imgHyper->getAxis(1);
	SEP::axis a2o=imgHyper->getAxis(2);
	SEP::axis a3o=imgHyper->getAxis(3);
	_file->seekTo(0,0);
	image->zero();
	_file->readFloatStream(image->vals,image->getN123());

	std::shared_ptr<float_3d> img(new float_3d(a1i,a2i,a3i));
	img->zero();
/*
   hypercube_float *mute=(hypercube_float*)image->clone_zero();
   sreed("mute",mute->vals,4*a1o.n*a2o.n*a3o.n);
   sseek("mute",0,0);
   for(int h=0; h<image->get_n123(); h++){
    image->vals[h]=image->vals[h]*mute->vals[h];
   }
 */

	int f3=(int)((a3i.o-a3o.o)/a3o.d);
	int f2=(int)((a2i.o-a2o.o)/a2o.d);
	int f1=(int)((a1i.o-a1o.o)/a1o.d);

	for(int i3=0; i3 < a3i.n; i3++) {
		if(i3+f3 >=0 && i3+f3 < a3o.n) {
			for(int i2=0; i2 < a2i.n; i2++) {
				if(i2+f2 >=0 && i2+f2 < a2o.n) {
					for(int i1=0; i1 < a1i.n; i1++) {
						if(i1+f1>=0 && i1+f1 <a1o.n) {
							img->vals[i1+(i2)*a1i.n+(i3)*a1i.n*a2i.n]=image->vals[i1+f1+(i2+f2)*a1o.n+(i3+f3)*a1o.n*a2o.n];
						}
					}
				}
			}
		}
	}

	return img;
}

void image_rtm_3d::write_volume(){
	// hypercube_float *mute=(hypercube_float*)image->clone_zero();
	//sreed("mute",mute->vals,4*image->get_axis(1).n*image->get_axis(2).n*image->get_axis(3).n);
	//sseek("mute",0,0);
//  for(int h=0; h<image->get_n123(); h++){
	//  image->vals[h]=image->vals[h]*mute->vals[h];
	// }
	return;
	std::stringstream ss;
	_file->getHyper()->infoStream(ss);
	image->infoStream(ss);
	std::cerr<<ss.str()<<std::endl;
	_file->seekTo(0,0);
	_file->writeFloatStream(image->vals,image->getN123());

}

void image_rtm_3d::write_final_volume(){
fprintf(stderr,"in write final volume \n");
	_file->seekTo(0,0);
	_file->writeFloatStream(image->vals,image->getN123());

}
