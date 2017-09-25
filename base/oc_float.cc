#include <oc_float.h>
#include <math.h>
#include <cstdlib>
#include <string.h>
#define BUFSZ 1000000
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )

oc_float::oc_float(std::shared_ptr<SEP::genericIO> io, std::string tag,std::vector<SEP::axis> axes,bool alloc){
	_io=io;
	_file=_io->getRegFile(tag,SEP::usageInOut);
	fprintf(stderr,"should have gotten regular file \n");
	std::shared_ptr<SEP::hypercube> hyper(new SEP::hypercube(axes));
	_file->setHyper(hyper);



	if(alloc) zeroFile();

	name="oc_float";
	_file->writeDescription();
	temp=false;
}
void oc_float::zeroFile(){
	set_val(0.);

}
oc_float::oc_float(std::shared_ptr<SEP::genericIO> io,std::string tag,std::shared_ptr<SEP::hypercube> hyper){

	_io=io;

	temp=false;
	_file=io->getRegFile(tag,SEP::usageInOut);
	_file->setHyper(hyper);
	_file->writeDescription();
	name="oc_float";
}
oc_float::oc_float(std::shared_ptr<SEP::genericIO> io,std::string tag,std::shared_ptr<oc_float> fle){

	_io=io;

	temp=false;
	_file=io->getRegFile(tag,SEP::usageInOut);
	_file->setHyper(fle->_file->getHyper());
	_file->writeDescription();
	name="oc_float";
}
oc_float::oc_float(std::shared_ptr<SEP::genericIO> io,std::string tag){
	_io=io;
	_file= io->getRegFile(tag,SEP::usageIn);
	name="oc_float";
	temp=false;

}
void oc_float::initNewFile(std::string tag, std::shared_ptr<SEP::genericIO> io,
	std::shared_ptr<SEP::hypercube> hyper, const SEP::usage_code usage){


	_io=io;
	_tag=tag;
	_file=_io->getRegFile(tag,usage);
	_file->setHyper(hyper);
	_file->writeDescription();

}
void oc_float::scale(double r){

	long long ndo=_file->getHyper()->getN123();
	long long nbuf=1000*1000;
	float *buf=new float[nbuf];
	long long ndone=0;
	_file->seekTo(0,0);
	while(ndone!=ndo) {
		long long nbuf=std::min(ndo-ndone,nbuf);
		_file->readFloatStream(buf,nbuf);
		for(long long i=0; i< nbuf; i++) buf[i]*=r;
		_file->seekTo(-nbuf*4,1);
		_file->writeFloatStream(buf,nbuf);
		ndone+=nbuf;
	}

	delete [] buf;
}
void oc_float::allocate(){
	zeroFile();
}
void oc_float::deallocate(){


}
std::shared_ptr<oc_float> oc_float::clone(bool alloc, std::string tag){
	int ndims=this->getHyper()->getNdim();

	std::vector<SEP::axis> axes;

	std::string val;

	if(tag=="NONE") val=make_temp();
	else val=tag;
	std::shared_ptr<oc_float> tmp;
	if(alloc) {
		tmp=this->clone(true,val);
	}
	else tmp=this->clone(false,val);

	return tmp;
}
void oc_float::set_val(double val){
_file->seekTo(0,0);
	long long ndo=_file->getHyper()->getN123();
	long long nbuf=1000*1000;
	float *zero=new float[nbuf];
	for(int i=0; i < nbuf; i++) zero[i]=val;
	long long ndone=0;

	while(ndone!=ndo) {
		long long n=std::min(ndo-ndone,nbuf);
		_file->writeFloatStream(zero,n);
		ndone+=n;
	}

	delete [] zero;

}
void oc_float::normalize(float val){
	double mx=my_max();
	double sc=val/mx;
	scale(sc);
}
void oc_float::info(char *str,int level){
	double sm=0.,mymin,mymax;
	int imin=0,imax=0;

	_file->seekTo(0,0);


	long long sz=getHyper()->getN123(), done=0,block;
	int blk;
	float *buf=new float[BUFSZ],*hold=new float[BUFSZ];
	while(done < sz) {
		block=MIN(sz-done,BUFSZ); blk=(int)block;
		_file->readFloatStream(buf,block);
		if(done==0) memcpy(hold,buf,BUFSZ*4);
		if(done==0) {mymin=mymax=buf[0];}
		for(int i=0; i < blk; i++) {
			if(mymin > buf[i]) {mymin=buf[i]; imin=i;}
			if(mymax < buf[i]) {mymax=buf[i]; imax=i;}
			sm+=buf[i]*buf[i];
		}
		//for(int i=0; i < blk; i++) buf[i]=buf[i]*r;
		done+=block;

	}
	delete [] buf;
	if(1==3) {
		fprintf(stderr,"N123 %d \n",(int)getHyper()->getN123());
		fprintf(stderr,"    NAME=%s TYPE=%s \n",str,name.c_str());
		std::shared_ptr<SEP::hypercube> hyper;
		for(int i=0; i < hyper->getNdim(); i++) fprintf(stderr,"    n%d=%d",i+1,getAxis(i+1).n);
		fprintf(stderr,"\n       N=%d min(%d)=%g max(%d)=%g RMS=%g   \n",
			(int)(hyper->getN123()), imin, mymin,imax, mymax, sqrt(sm)/(1.0*hyper->getN123()));
		long long print;
		if(level!=0) {
			if(level<1) print=getHyper()->getN123();
			else print=MIN(level,BUFSZ);
			int ic=0;
			for(long long i=0; i <MIN(BUFSZ,sz); i++) {
				if(fabs(hold[i])>.0000001 && ic <print) {
					// fprintf(stderr,"val %d %f \n",(int)i,hold[i]);
					ic+=1;
				}
			}
		}
	}
	delete [] hold;
}
std::shared_ptr<my_vector>oc_float::clone_vec(){
	//std::shared_ptr<my_vector>m=(my_vector*) this->clone();
	std::shared_ptr<oc_float> m=this->clone();
	return m;
}
std::shared_ptr<my_vector>oc_float::clone_space(){
	std::shared_ptr<oc_float> m=this->clone(false);
	return m;
}
void oc_float::random(){

	long long ndo=_file->getHyper()->getN123();
	long long nbuf=1000*1000;
	float *buf=new float[nbuf];
	long long ndone=0;
	_file->seekTo(0,0);
	while(ndone!=ndo) {
		long long nbuf=std::min(ndo-ndone,nbuf);
		_file->readFloatStream(buf,nbuf);
		for(long long i=0; i< nbuf; i++) buf[i]=(float)rand()/(float)RAND_MAX-.5;
		_file->seekTo(-nbuf*4,1);
		_file->writeFloatStream(buf,nbuf);
		ndone+=nbuf;
	}

}
double oc_float::dot(std::shared_ptr<my_vector>other){
	double ret=0;
	check_same(other);
	std::shared_ptr<oc_float> o = std::dynamic_pointer_cast<oc_float>( other);
	long long ndo=_file->getHyper()->getN123();
	long long nbuf=1000*1000;
	float *buf1=new float[nbuf];
	float *buf2=new float[nbuf];

	long long ndone=0;
	_file->seekTo(0,0);
	o->_file->seekTo(0,0);

	while(ndone!=ndo) {
		long long nbuf=std::min(ndo-ndone,nbuf);
		_file->readFloatStream(buf1,nbuf);
		o->_file->readFloatStream(buf2,nbuf);
		for(long long i=0; i< nbuf; i++) ret+=buf1[i]*buf2[i];
		ndone+=nbuf;
	}
	delete[] buf1;
	delete [] buf2;
	return ret;


}
void oc_float::mult(std::shared_ptr<my_vector>other){

	check_same(other);
	std::shared_ptr<oc_float> o = std::dynamic_pointer_cast<oc_float>( other);

	long long ndo=_file->getHyper()->getN123();
	long long nbuf=1000*1000;
	float *buf1=new float[nbuf];
	float *buf2=new float[nbuf];

	long long ndone=0;
	_file->seekTo(0,0);
	o->_file->seekTo(0,0);

	while(ndone!=ndo) {
		long long nbuf=std::min(ndo-ndone,nbuf);
		_file->readFloatStream(buf1,nbuf);
		o->_file->readFloatStream(buf2,nbuf);
		for(long long i=0; i< nbuf; i++) buf1[i]*=buf2[i];
		_file->seekTo(-nbuf,1);
		_file->writeFloatStream(buf1,nbuf);
		ndone+=nbuf;
	}
	delete[] buf1;
	delete [] buf2;


}
void oc_float::scale_add(double sc1, std::shared_ptr<my_vector>v1, double sc2, std::shared_ptr<my_vector>v2){

	check_same(v1);
	check_same(v2);
	std::shared_ptr<oc_float> o1 = std::dynamic_pointer_cast<oc_float>( v1);
	std::shared_ptr<oc_float> o2 = std::dynamic_pointer_cast<oc_float>( v2);

	long long ndo=_file->getHyper()->getN123();
	long long nbuf=1000*1000;
	float *buf1=new float[nbuf];
	float *buf2=new float[nbuf];
	float *buf=new float[nbuf];

	long long ndone=0;
	_file->seekTo(0,0);
	o1->_file->seekTo(0,0);
	o2->_file->seekTo(0,0);

	while(ndone!=ndo) {
		long long nbuf=std::min(ndo-ndone,nbuf);
		o1->_file->readFloatStream(buf1,nbuf);
		o2->_file->readFloatStream(buf2,nbuf);

		for(long long i=0; i< nbuf; i++) buf[i]=buf1[i]*sc1+buf2[i]*sc2;
		_file->writeFloatStream(buf,nbuf);
		ndone+=nbuf;
	}
	delete[] buf1;
	delete [] buf2;
	delete [] buf;
}
double oc_float::sum(){
	double ret=0;

	long long ndo=_file->getHyper()->getN123();
	long long nbuf=1000*1000;
	float *buf1=new float[nbuf];
\
	long long ndone=0;
	_file->seekTo(0,0);

	while(ndone!=ndo) {
		long long nbuf=std::min(ndo-ndone,nbuf);
		_file->readFloatStream(buf1,nbuf);
		for(long long i=0; i< nbuf; i++) ret+=buf1[i];
		ndone+=nbuf;
	}
	delete[] buf1;
	return ret;
}

double oc_float::my_min(){
	double ret=1e31;

	long long ndo=_file->getHyper()->getN123();
	long long nbuf=1000*1000;
	float *buf1=new float[nbuf];

	long long ndone=0;
	_file->seekTo(0,0);

	while(ndone!=ndo) {
		long long nbuf=std::min(ndo-ndone,nbuf);
		_file->readFloatStream(buf1,nbuf);
		for(long long i=0; i< nbuf; i++) ret=std::min(ret,(double)buf1[i]);
		ndone+=nbuf;
	}
	delete[] buf1;
	return ret;
}

double oc_float::my_max(){
	double ret=-1e31;

	long long ndo=_file->getHyper()->getN123();
	long long nbuf=1000*1000;
	float *buf1=new float[nbuf];

	long long ndone=0;
	_file->seekTo(0,0);

	while(ndone!=ndo) {
		long long nbuf=std::min(ndo-ndone,nbuf);
		_file->readFloatStream(buf1,nbuf);
		for(long long i=0; i< nbuf; i++) ret=std::max(ret,(double)buf1[i]);
		ndone+=nbuf;
	}
	delete[] buf1;
	return ret;
}
void oc_float::scale_add(const double sc1, std::shared_ptr<my_vector>other, const double sc2){

	check_same(other);
	std::shared_ptr<oc_float> o1 = std::dynamic_pointer_cast<oc_float>( other);


	long long ndo=_file->getHyper()->getN123();
	long long nbuf=1000*1000;
	float *buf1=new float[nbuf];
	float *buf=new float[nbuf];

	long long ndone=0;
	_file->seekTo(0,0);
	o1->_file->seekTo(0,0);

	while(ndone!=ndo) {
		long long nbuf=std::min(ndo-ndone,nbuf);
		o1->_file->readFloatStream(buf1,nbuf);
		_file->readFloatStream(buf,nbuf);
		for(long long i=0; i< nbuf; i++) buf[i]=buf[i]*sc1+buf1[i]*sc2;
		_file->seekTo(-nbuf,1);
		_file->writeFloatStream(buf,nbuf);
		ndone+=nbuf;
	}
	delete[] buf1;
	delete [] buf;
}

void oc_float::add(std::shared_ptr<my_vector>other){
	scale_add(1.,other,1.);
}
bool oc_float::check_match(const std::shared_ptr<my_vector>v2){
	if(-1==v2->name.find("oc_float")) {
		fprintf(stderr,"vector not oc_float");
		return false;
	}
	std::shared_ptr<oc_float> h2 = std::dynamic_pointer_cast<oc_float>( v2);

	if(_file->getHyper()->getNdim()!=h2->_file->getHyper()->getN123()) {
		fprintf(stderr,"vectors not the same number of dimensions\n");
		return false;
	}

	for(int i=0; i < _file->getHyper()->getN123(); i++) {
		SEP::axis a1=getAxis(i+1),a2=h2->getAxis(i+1);
		if(a1.n!=a2.n) {
			fprintf(stderr,"vectors axis=%d not the same number of samples %d,%d \n",
				i+1,a1.n,a2.n);
			return false;
		}

		if(fabs((a1.o-a2.o)/a1.d) > .01) {
			fprintf(stderr,"vectors axis=%d not the same origin %f,%f \n",
				i+1,a1.o,a2.o);
			return false;
		}  if(a1.n!=a2.n) {
			fprintf(stderr,"vectors axis=%d not the same sampling %f,%f \n",
				i+1,a1.d,a2.d);
			return false;
		}
	}

	return true;
}

std::string oc_float::make_temp(){
	char temp_file[4096];
	strcpy(temp_file,"TEMP_XXXXXX"); mkstemp(temp_file);
	std::string tmp=temp_file;
	return tmp;

}
void oc_float::tagInit(std::string tag){
	_file=_io->getRegFile(tag,SEP::usageInOut);

}
void oc_float::readAll(std::shared_ptr<hypercube_float> dat){
	std::shared_ptr<SEP::hypercube> hyper=_file->getHyper();
	_file->readFloatStream(dat->vals,hyper->getN123());
}

void oc_float::init_ndf(std::vector<SEP::axis> ax){
	std::shared_ptr<SEP::hypercube> hyper(new SEP::hypercube(ax));
	_file=_io->getRegFile(_tag,SEP::usageIn);
	_file->setHyper(hyper);
	_file->writeDescription();
	//Need to fill this in
	//initNd(ax); allocate();

}
