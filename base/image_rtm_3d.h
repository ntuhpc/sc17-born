#ifndef IMAGE_RTM_3D_H
#define IMAGE_RTM_3D_H 1
#include "vel_fd_3d.h"
#include "float_3d.h"
#include "oc_float.h"

class image_rtm_3d : public oc_float {
public:
image_rtm_3d();
image_rtm_3d(std::shared_ptr<SEP::genericIO> io, std::string tag,std::shared_ptr<vel_fd_3d> v);


void write_volume();
void write_final_volume();
void add_image(std::shared_ptr<hypercube_float> img);
std::shared_ptr<float_3d> extract_sub(SEP::axis a1, SEP::axis a2, SEP::axis a3);

void set_source_file(std::shared_ptr<oc_float> ptr);
std::shared_ptr<hypercube_float> image, mute;
std::shared_ptr<SEP::hypercube> imgHyper;

};
#endif
