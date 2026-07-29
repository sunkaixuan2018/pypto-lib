/*
 * Split M=16,K=2048 INT8 activation into two packed signed-INT4 planes.
 */
#define PYPTO_PROJECTION_K 2048
#define PYPTO_PROJECTION_BLOCKS 4
#include "split_common.cpp"
