#!/bin/bash
rm -rf build
mkdir build
cd build
export CXX=g++
cmake -DgenericIO_DIR=/sep/bob/genericIO/noSEP/lib -DCMAKE_INSTALL_PREFIX=/sep/bob/Born2 ..
make
make install
