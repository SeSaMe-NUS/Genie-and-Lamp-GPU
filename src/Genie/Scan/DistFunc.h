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
__host__ __device__ T eu(const T* Q, uint sq,const T* C, uint sc, uint cq_len);

template <class T>
__host__ __device__ T dtw_DP_SCBand_modulus(const T* Q, uint q_len,const T* C, uint c_len, uint r);

template<class T>
__host__ __device__ T dtw(const T* Q, uint sq, const T* C, uint sc, uint cq_len);

template<class T>
__host__ __device__ T depressed_dtw_SCBand(const T* Q, uint sq, const T* C, uint sc, uint cq_len, uint r);

template <class T>//this is auxiliary function of depressed_dtw_SCBand();
__host__ __device__ T depressed_dtw_DP_SCBand(const T* Q, uint q_len,const T* C, uint c_len, uint r);

template<class T>
__host__ __device__ T dtw_compressDP(const T* Q, uint q_len, const T*C, uint c_len);

template<class T>
__host__ __device__ T dtw_recur(const T*Q, uint q_len, const T* C, uint c_len);

template<class T>
__host__ __device__ T dtw_AuxRecur(const T* Q, uint qi, const T* C, uint cj);

template<class T>
__host__ __device__ T dtw_recur_SCBand(const T*Q, uint q_len, const T* C, uint c_len, uint r);

template<class T>
__host__ __device__ T dtw_AuxRecur_SCBand(const T* Q, uint qi, const T* C, uint cj, uint r);
//};

//for LB_Keogh
template<class T>
__host__ __device__ T LowerBound_keogh_byQ(const T* Q, int sq, const T* C, int sc, int cq_len, int sc_band);



template <class T>
struct Eu_Func{


	__host__ __device__ Eu_Func(){

	}

	__host__ __device__ T dist ( const T* Q, uint sq, const T* C, uint sc, uint cq_len){

		T d = 0;

		d = eu(Q,sq,C,sc,cq_len);

		return d;
	}

};

//this function can only run in CPU and cannot work for large scale GPU, but the program is corrected.
//we keep this function for experiment and debug purpose
template <class T>
struct depressed_Dtw_SCBand_Func_old{

	uint sc_band;

	__host__ __device__ depressed_Dtw_SCBand_Func_old(uint sc_band){
		this->sc_band = sc_band;
	}

	__host__ __device__ T dist ( const T* Q, uint sq, const T* C, uint sc, uint cq_len){
		return depressed_dtw_SCBand( Q, sq,  C,  sc, cq_len, sc_band);
	}
};

template <class T>
struct Dtw_SCBand_Func_modulus{

	uint sc_band;

	__host__ __device__ Dtw_SCBand_Func_modulus(uint sc_band){
		this->sc_band = sc_band;
	}

	__host__ __device__ T dist ( const T* Q, uint sq, const T* C, uint sc, uint cq_len){
		return dtw_DP_SCBand_modulus( Q+sq, cq_len,C+sc, cq_len, sc_band);
	}

};



template<class T>
struct Dtw_SCBand_LBKeogh {
	int sc_band;

	__host__ __device__ Dtw_SCBand_LBKeogh(uint sc_band) {
		this->sc_band = sc_band;
	}
	__host__ __device__ T LowerBound_keogh_byQuery(const T* Q, int sq,
			const T* C, int sc, int cq_len) {

		return LowerBound_keogh_byQ(Q, sq, C, sc, cq_len, sc_band);

	}

	__host__ __device__ T LowerBound_keogh_byData(const T* Q, int sq,
			const T* C, int sc, int cq_len) {

		return LowerBound_keogh_byQ(C, sc, Q, sq, cq_len, sc_band);

	}
};

#include "DistFunc.inc"
#endif /* DISTFUNC_H_ */
