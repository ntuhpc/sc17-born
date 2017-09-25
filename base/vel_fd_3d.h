#include "hypercube_float.h"
#include <genericFile.h>
#include <genericIO.h>
#ifndef VEL_RTM_3D_H
#define VEL_RTM_3D_H 1
/**
   Class for medium velocity
 */
class vel_fd_3d {
public:
/**
   Default constructor should never be called
 */
vel_fd_3d() : _vel(nullptr),_dens(nullptr){
	fprintf(stderr,"default constructor \n");
}
/** constructor
   \param io class handling parameters and iO
   \param tag the name of the file containing the velocity values
 */
vel_fd_3d(std::shared_ptr<SEP::genericIO> io,
	std::string tag);
/** constructor
   \param io class handling parameters and iO
   \param tagV the name of the file containing the velocity values
         \param tagD the name of the file containing the densitiy values
 */
vel_fd_3d(std::shared_ptr<SEP::genericIO> io,std::string tagV, std::string tagD);
/**
   Return hypercube_float with random values around the edge
   \param irand  initialize random engine
   \param nrandt number of random on top
   \param nrandbldr
   \param randblr2
   \param dt propagation rate in time
   \space space describing output
 */
std::shared_ptr<hypercube_float> rand_subspace(const int irand, const int nrandt,
	const int nrandblr, const int randblr2, const float dt, std::shared_ptr<hypercube_float> space);
/** get_min_max - get min sampling in all directions
 */
float get_min_samp(){
	float dx=_vel->getAxis(1).d;
	float dy=_vel->getAxis(2).d;
	float dz=_vel->getAxis(3).d;
	if(dz < dx && dz < dy ) return dz;
	if(dy < dx ) return dz;
	return dx;
}
/*get max sampling along all axes*/
float get_max_samp(){
	float dx=_vel->getAxis(1).d;
	float dy=_vel->getAxis(2).d;
	float dz=_vel->getAxis(3).d;
	if(dz > dx && dz > dy ) return dz;
	if(dy > dx) return dz;
	return dx;
}
void set_old(){
	_old=true;
}
void set_mid(){
	_mid=true;
}
bool has_dens(){
	if(!_dens) return false;
	return true;
}
float max_vel(){
	return (float) _vel->my_max();
}
float min_vel(){
	return (float) _vel->my_min();
}
SEP::axis getAxis(int iax){
	return _vel->getAxis(iax);
}
void set_zero(){
	_zero=true;

}
~vel_fd_3d(){
	fprintf(stderr,"in delete 1 \n");


}
/** io being used*/
std::shared_ptr<SEP::genericIO> _io;
/**
   values
 */
std::shared_ptr<hypercube_float>_vel;
/** Class for density
 */
std::shared_ptr<hypercube_float>_dens;
/** Class handling IO*/
std::shared_ptr<SEP::genericRegFile>_vFile,_dFile;
bool _old,_mid,_zero;




};
#endif
