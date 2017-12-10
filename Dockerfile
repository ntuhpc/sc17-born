From centos:7
MAINTAINER Bob Clapp <bob@sep.stanford.edu>
RUN yum -y install gcc-gfortran gcc-c++ cmake make git t
RUN git clone http://zapad.Stanford.EDU/bob/hypercube.git /opt/hypercube/src
RUN git clone http://zapad.Stanford.EDU/bob/genericIO.git /opt/genericIO/src
RUN mkdir -p /opt/hypercube/build
RUN mkdir -p /opt/genericIO/build
RUN cd
RUN cd /opt/genericIO/build; cmake -DCMAKE_INSTALL_PREFIX=/opt/genericIO  ..; make install;
