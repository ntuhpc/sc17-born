#!/usr/bin/env python
import commands
x=[]
y=[]

def write_file(xx,yy):
      fileName="data.big.%d.%d.json"%(int(xx),int(yy));
      x=open(fileName,"w")
      x.write('{\n')
      x.write('"d1" : 0.004,\n')
      x.write('"d2" : 25,\n')
      x.write('"d3" : 25,\n')
      x.write('"d4" : 25,\n')
      x.write('"d5" : 25,\n')
      x.write('"filename" : "/data/sep/bob/test/data.big.json.%d.%d.dat",\n'%(int(xx),int(yy)))
      x.write('"label1" : "Undefined",\n')
      x.write('"label2" : "Undefined",\n')
      x.write('"label3" : "Undefined",\n')
      x.write('"label4" : "Undefined",\n')
      x.write('"label5" : "Undefined",\n')
      x.write('"n1" : 1500,\n')
      x.write('"n2" : 240,\n')
      x.write('"n3" : 240,\n')
      x.write('"n4" : 1,\n')
      x.write('"n5" : 1,\n')
      x.write('"o1" : 0,\n')
      x.write('"o2" : 0,\n')
      x.write('"o3" : 0,\n')
      x.write('"o4" : %d,\n'%xx)
      x.write('"o5" : %d\n}\n'%yy)
      x.close()
      par="model.%d.%d.P"%(xx,yy);
      y=open(par,"w");
      y.write("{\n")
      y.write('"data": "/data/sep/bob/test/cpp/born2/examp/data.big.%d.%d.json",\n'%(xx,yy))
      y.write('"image": "/data/sep/bob/test/cpp/born2/examp/refl.big.json\",\n')
      y.write('"velocity": "/data/sep/bob/test/cpp/born2/examp/vel.big.json\",\n')
      y.write('"wavelet": "/data/sep/bob/test/cpp/born2/examp/wavelet.json"\n}\n')
      y.close()
  
def copy_file(x,y):
  print "hould copy"
  commands.getoutput("cp /data/sep/bob/test/data.big.json.dat /data/sep/bob/test/data.big.%d.%d.json.dat"%(x,y));
  
  
def run_job(x,y):
   stat,out=commands.getstatusoutput("Window3d < srcloci.H >srcloci.%d.H f2=%d n2=1"%(i,i));
   f=open("sendit.%d.%d.sh"%(x,y),"w")
   f.write( "#!/bin/tcsh\n");
   f.write("#PBS -N 3dsyn.%d.%d\n"%(x,y))
   f.write("#PBS -l nodes=1:ppn=16\n");
   f.write("#PBS -q sep\n");
   f.write("#PBS -V\n");
   f.write("#PBS -e ./test%d.%d.err\n"%(x,y));
   f.write("#PBS -o ./test%d.%d.out\n"%(x,y));
   f.write("#$ -cwd\n");
   f.write("#\n");
   f.write("date\n");
   f.write( "/data/sep/bob/test/Born/bin/Model3D json=/data/sep/bob/test/cpp/born2/examp/model.%d.%d.P\n"%(x,y))
   f.write("date\n");
   f.close();
   commands.getstatusoutput("qsub sendit.%d.%d.sh"%(x,y))
#  commands.getoutput("/data/sep/bob/test/Born/bin/Model3D json=model.%d.%d.P"%(x,y))
	
  




for ibigOut in range(4):
  for ibigIn in range(4):
    for i2 in range(5,10):
      for i1 in range(10):
        x.append(i2*800.+ibigOut*200);
        y.append(i1*800.+ibigIn*200);
     



for i in range(len(x)):
   print "writing file",x[i],y[i]
   write_file(x[i],y[i]);
   print "copying file",x[i],y[i]
   copy_file(int(x[i]),int(y[i]))
   run_job(int(x[i]),int(y[i]));
   
