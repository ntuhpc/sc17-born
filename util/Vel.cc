#include <ioModes.h>
using namespace SEP;

int main(int argc, char **argv){

	ioModes modes(argc,argv);



	std::shared_ptr<genericIO>  io=modes.getDefaultIO();
	std::shared_ptr<paramObj> par=io->getParamObj();





	std::string out=par->getString(std::string("out"));

	std::vector<SEP::axis> axes;
	int nx=par->getInt("nx",400); float dx=par->getFloat("dx",.01); axes.push_back(SEP::axis(nx,0.,dx));
	int ny=par->getInt("ny",400); float dy=par->getFloat("dy",.01); axes.push_back(SEP::axis(ny,0.,dy));
	int nz=par->getInt("nz",400); float dz=par->getFloat("dz",.01); axes.push_back(SEP::axis(nz,0.,dz));

	std::shared_ptr<hypercube> hyp(new hypercube(axes));

	std::shared_ptr<genericRegFile> outp=io->getRegFile(out,usageOut);
	outp->setHyper(hyp);
	outp->writeDescription();

	float vconst=par->getFloat("vconst",2.0);
	std::vector<float> val(hyp->getN123(),vconst);


	outp->writeFloatStream(val.data(),hyp->getN123());


}
