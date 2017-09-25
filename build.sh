#!/bin/bash
BORN_INSTALL_DIR=/home/public/born/install
rm -rf build
mkdir build
cd build
CXX=icpc CC=icc cmake -DCMAKE_LIBRARY_PATH=$TBBROOT/lib/intel64/gcc4.7 -DgenericIO_DIR=$BORN_INSTALL_DIR/lib -Dhypercube_DIR=$BORN_INSTALL_DIR/lib -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-xHost" -DUSE_GPU=ON .. 
make -j 
