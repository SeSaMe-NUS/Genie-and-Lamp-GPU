/*
 * DistFunc.h
 *
 *  Created on: Feb 24, 2014
 *      Author: zhoujingbo
 */

#ifndef DISTFUNC_H_
#define DISTFUNC_H_
#include <stdio.h>
#include <unistd.h>
#include <device_functions.h>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


//template <class T>
//class DistFunc {
//public:
//	__host__ __device__ DistFunc();
//	__host__ __device__ virtual ~DistFunc();
template<class T>
__host__ __device__ T dtw(T* Q, uint sq, T* C, uint sc, uint cq_len);

template<class T>
__host__ __device__ T dtw_SCBand(T* Q, uint sq, T* C, uint sc, uint cq_len, uint r);

template<class T>
__host__ __device__ T eu(T* Q, uint sq, T* C, uint sc, uint cq_len);

template<class T>
__host__ __device__ T dtw_compressDP(T* Q, uint q_len, T*C, uint c_len);

template<class T>
__host__ __device__ T dtw_DP_SCBand(T* Q, uint q_len, T* C, uint c_len, uint r);

template<class T>
__host__ __device__ T dtw_recur(T*Q, uint q_len, T* C, uint c_len);

template<class T>
__host__ __device__ T dtw_AuxRecur(T* Q, uint qi, T* C, uint cj);

template<class T>
__host__ __device__ T dtw_recur_SCBand(T*Q, uint q_len, T* C, uint c_len, uint r);

template<class T>
__host__ __device__ T dtw_AuxRecur_SCBand(T* Q, uint qi, T* C, uint cj, uint r);
//};



template <class T>
struct Eu_Func{

	__host__ __device__ T dist ( T Q[], uint sq, T C[], uint sc, uint cq_len){

		T d = 0;

		d = eu(Q,sq,C,sc,cq_len);

		return d;
	}

};

template <class T>
struct Dtw_SCBand_Func{

	uint sc_band;

	__host__ __device__ T dist ( T Q[], uint sq, T C[], uint sc, uint cq_len){
		return dtw_SCBand( Q, sq,  C,  sc, cq_len, sc_band);
	}
};

#include "DistFunc.inc"
#endif /* DISTFUNC_H_ */
