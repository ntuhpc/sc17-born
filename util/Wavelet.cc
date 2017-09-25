#include <ioModes.h>
using namespace SEP;

int main(int argc, char **argv){

	ioModes modes(argc,argv);



	std::shared_ptr<genericIO>  io=modes.getDefaultIO();
	std::shared_ptr<paramObj> par=io->getParamObj();





	std::string out=par->getString(std::string("out"));


	int nt=par->getInt("nt",128);
	float dt=par->getFloat("dt",.004);
	float fund=par->getFloat("fund",10.);

	SEP::axis a=axis(nt,0.,dt);
	std::shared_ptr<hypercube> hyp(new hypercube(a));

	std::shared_ptr<genericRegFile> outp=io->getRegFile(out,usageOut);
	outp->setHyper(hyp);
	outp->writeDescription();

	float pi=atan(1.f)*4;	
	std::vector<float> val(nt,0.);
	for(int it=0; it < nt; it++) {
		float tm=-.064+it*dt;
		val[it]=(1.-2.*(pi*pi)*fund*fund*tm*tm) *expf(-pi*pi*fund*fund*tm*tm);
	}

	outp->writeFloatStream(val.data(),nt);


}
