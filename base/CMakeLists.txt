cmake_minimum_required(VERSION 2.8)
project(born-base)

include_directories(${GENERICIOINCDIR})

#include_directories(/sep/bob/genericIO/include)
set(SOURCES my_operator.cc hypercube_float.cc float_1d.cc float_3d.cc i_op.cc
  cgstep.cc lin_solver.cc super_vector.cc combo_oper.cc laplac_3d.cc
  source_func_3d.cc wavefield_insert_3d.cc oc_float.cc sinc_bob.cc
  deriv_3d.cc vel_fd_3d.cc tmute.cc data_rtm_3d.cc image_rtm_3d.cc
  fd_prop_3d.cc map_data_3d.cc rtm_zero_op_3d.cc cpu_prop.cc gpu_prop.cc

)
set(INCLUDES my_vector.h my_operator.h hypercube_float.h float_1d.h float_3d.h
  cgstep.h lin_solver.h i_op.h step.h super_vector.h combo_oper.h laplac_3d.h
  source_func_3d.h wavefield_insert_3d.h oc_float.h deriv_3d.h vel_fd_3d.h
  rtm_zero_op_3d.h fd_prop_3d.h map_data_3d.h
  sinc_bob.h tmute.h data_rtm_3d.h image_rtm_3d.h
  cpu_prop.h gpu_prop.h base_prop.h
)


add_library(base ${SOURCES} ${INCLUDES})
