
Pre-reqs 

You need  a c++ compiler with  c++11 support and tbb.

You need to install two small libraries before installing Born.

Install instructions

1. Install hypercube
  1. git clone http://zapad.Stanford.EDU/bob/hypercube.git   /opt/hypercube/src 
  1. mkdir /opt/hypercube/build 
  1.  cd /opt/hypercube/build 
  1.    cmake -DCMAKE_INSTALL_PREFIX=/opt/hypercube ../src 
  1.    make install 
  1.    rm -rf /opt/hypercube/build

1. Install genericIO
  1. git clone http://zapad.Stanford.EDU/bob/genericIO.git /opt/genericIO/src 
  1. mkdir /opt/genericIO/build
  1. cd /opt/genericIO/build
  1. cmake  -Dhypercube_DIR=/opt/hypercube/lib  -DCMAKE_INSTALL_PREFIX=/opt/genericIO ../src
  1. make install 
  1. rm -rf /opt/genericIO/build

1. Install Born
  1. git clone http://zapad.Stanford.EDU/SEP-external/Born.git /opt/born/src 
  1. mkdir /opt/born/build 
  1. cd /opt/born/build 
  1. cmake  -Dhypercube_DIR=/opt/hypercube/lib -DgenericIO_DIR=/opt/genericIO/lib  -DCMAKE_INSTALL_PREFIX=/opt/genericIO ../src 
  1. make install 
  1. rm -rf /opt/born/build



