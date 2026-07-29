/*
 * Split M=16,K=4096 INT8 activation into two packed signed-INT4 planes.
 */
#define PYPTO_PROJECTION_K 4096
#define PYPTO_PROJECTION_BLOCKS 2
#include "split_common.cpp"
