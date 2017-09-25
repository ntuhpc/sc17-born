#include<ioModes.h>
#include<cassert>
using namespace SEP;
int main(int argc, char **argv){

  ioModes modes(argc,argv);



  std::shared_ptr<genericIO>  io=modes.getDefaultIO();
  std::shared_ptr<paramObj> par=io->getParamObj();



  std::string in1=par->getString(std::string("in1"));
  std::string in2=par->getString(std::string("in2"));

  std::string out=par->getString(std::string("out"));

  std::shared_ptr<genericRegFile> inp1=io->getRegFile(in1,usageIn);
  std::shared_ptr<genericRegFile> inp2=io->getRegFile(in2,usageIn);


  std::shared_ptr<hypercube> hyperIn1=inp1->getHyper();
  std::shared_ptr<hypercube> hyperIn2=inp2->getHyper();
  assert(hyperIn2->sameSize(hyperIn1));
  std::shared_ptr<genericRegFile> outp=io->getRegFile(out,usageOut);

  outp->setHyper(hyperIn1);
  outp->writeDescription();


  long long blockSize=100*1024*1024;
  float *inf1=new float[blockSize];
  float *inf2=new float[blockSize];

  long long n123=hyperIn1->getN123();

  long long done=0;

  while(done!=n123){
    long pass=std::min(n123-done,blockSize);
    inp1->readFloatStream(inf1,pass);
    inp2->readFloatStream(inf2,pass);
    for(long long i=0;i < pass; i++) inf1[i]+=inf2[i]; 
    outp->writeFloatStream(inf1,pass);
    done+=pass;
  }
}
