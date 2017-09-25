#include <ioModes.h>
using namespace SEP;

int main(int argc, char **argv){

	ioModes modes(argc,argv);



	std::shared_ptr<genericIO>  io=modes.getDefaultIO();
	std::shared_ptr<paramObj> par=io->getParamObj();





	std::string out=par->getString(std::string("out"));

	std::vector<SEP::axis> axes;
	int nt=par->getInt("nt",400); float dt=par->getFloat("dt",.008); 
     axes.push_back(SEP::axis(nt,0.,dt));
	int nx=par->getInt("nx",300);
	float dx=par->getFloat("dx",.01);
	float ox=par->getFloat("ox",.5);
	axes.push_back(SEP::axis(nx,ox,dx));
	int ny=par->getInt("ny",300);
	float dy=par->getFloat("dy",.01);
	float oy=par->getFloat("oy",.5);
	axes.push_back(SEP::axis(ny,oy,dy));
        axes.push_back(SEP::axis(1,ox+dx*nx/2,dx));
        axes.push_back(SEP::axis(1,oy+dy*ny/2,dy));
 

	std::shared_ptr<hypercube> hyp(new hypercube(axes));

	std::shared_ptr<genericRegFile> outp=io->getRegFile(out,usageOut);
	outp->setHyper(hyp);
	outp->writeDescription();

 float *val=new float[hyp->getN123()];
for(auto i=0; i < hyp->getN123(); i++) val[i]=0.;


for(int i3=20; i3 < ny; i3+=20){
  for(int i2=20; i2 < nx; i2+=20){
   for(int i1=30; i1 < nt; i1+=50){
      val[i1+i2*nt+i3*nx*nt]=1.;
   }
} 
}

      val[nt/2+nx/2*nt+ny/2*nx*nt]=1.;

	outp->writeFloatStream(val,hyp->getN123());


}
