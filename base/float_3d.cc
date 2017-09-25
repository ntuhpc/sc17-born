#include <float_3d.h>

float_3d::float_3d(SEP::axis a1, SEP::axis a2, SEP::axis a3)
{
	base_init_3df(a1,a2,a3);
}
float_3d::float_3d(SEP::axis a1, SEP::axis a2, SEP::axis a3, float *array)
{
	this->base_init_3df(a1,a2,a3);
	this->set(array);
}
void float_3d::base_init_3df(SEP::axis a1, SEP::axis a2, SEP::axis a3)
{
	std::vector<SEP::axis> axes;
	axes.push_back(a1);
	axes.push_back(a2);
	axes.push_back(a3);
	this->initNd(axes);
	this->allocate();
}
float_3d::float_3d(int n1, int n2, int n3)
{
	SEP::axis a1=SEP::axis(n1);
	SEP::axis a2=SEP::axis(n2);
	SEP::axis a3=SEP::axis(n3);
	this->base_init_3df(a1,a2,a3);
}
float_3d::float_3d(int n1, int n2, int n3,float *array)
{

	SEP::axis a1=SEP::axis(n1);
	SEP::axis a2=SEP::axis(n2);
	SEP::axis a3=SEP::axis(n3);
	this->base_init_3df(a1,a2,a3);
	this->set(array);
}
