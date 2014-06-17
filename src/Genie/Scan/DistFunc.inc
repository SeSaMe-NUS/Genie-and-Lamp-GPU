/*
 * DistFunc.cu
 *
 *  Created on: Feb 24, 2014
 *      Author: zhoujingbo
 */


#include <math.h>
#include <limits.h>
#include <iostream>
#include "stdio.h"

#include "DistFunc.h"

//
//template <class T>
//__host__ __device__ DistFunc<T>::DistFunc()
//{
//	// TODO Auto-generated constructor stub
//
//}
//
//template <class T>
//__host__ __device__ DistFunc<T>::~DistFunc()
//{
//	// TODO Auto-generated destructor stub
//}
//

/*
 * sq:starting point of Q
 * sc:starint point of C
 * cq_len: length of Q and C
 */
template <class T>
__host__ __device__ T dtw( const T* Q, uint sq, const T* C, uint sc, uint cq_len){


	T* iq = new T[cq_len+1];
	T* ic = new T[cq_len+1];

	iq[0] = INT_MAX-1;
	ic[0] = INT_MAX-1;

	for(uint i=0;i<cq_len;i++){

		iq[i+1]=Q[sq+i];
		ic[i+1]=C[sc+i];

	}

	T d1 = dtw_compressDP(iq,cq_len,ic,cq_len);
	//T d2 = dtw_recur(iq,cq_len,ic,cq_len);
	//printf("DTW is dtw_compressDP:%i dtw_recur:%d \n",d1, d2);
	//if(d1!=d2){
	//	printf("error: the DP_DTW and Recursive_DTW are unequal, something is wrong!\n");
	//}

	delete[] iq;
	delete[] ic;
	return d1;
}


/**
 * sq: starting position of q
 * sc: starting position of C
 */
template <class T>
__host__ __device__ T dtw_SCBand( const T* Q, uint sq, const T* C, uint sc, uint cq_len, uint r){


	T* iq = new T[cq_len+1];
	T* ic = new T[cq_len+1];

	iq[0] = INT_MAX-1;
	ic[0] = INT_MAX-1;

	for(uint i=0;i<cq_len;i++){

		iq[i+1]=Q[sq+i];
		ic[i+1]=C[sc+i];

	}

	T d1 = dtw_DP_SCBand(iq,cq_len,ic,cq_len,r);
	//T d2 = dtw_recur_SCBand(iq,cq_len,ic,cq_len,r);
	//printf("DTW is dtw_compressDP:%i dtw_recur:%d \n",d1, d2);
	//if(d1!=d2){
	//	printf("error: the DP_DTW and Recursive_DTW are unequal, something is wrong!\n");
	//}

	delete[] iq;
	delete[] ic;
	return d1;
}

/**
 * the data points of Q and C is from 1 to cq_len
 * Q[0]=C[0]=infinity
 * refer to paper "Accelerating dynamic time warping subsequence search with GPUs and FPGAs, ICDM, but the Algorithm in Table2 is wrong (or confused). This is the correct one
 *
 * Algorithm:
 * s = 0
 * for i=0 to m:
 * 	   d(i,s) =  infinity
 *
 * s = s XOR 1//Xor operation
 *
 * for j = 1 to n
 * 	   for i = 1 to m
 * 	       d(i,s) = |C(j)-Q(i)| + min(d(i-1),s), d(i, s XOR 1), d(i-1, s XOR 1))
 * 	   s = s XOR 1
 *
 * return d(n,s XOR 1)
 *
 */
template <class T>
__host__ __device__ T dtw_compressDP(const T* Q, uint q_len, const T* C, uint c_len){

	uint s=0;
	T** d;
	d = new T*[q_len+1];
	for(uint i=0;i<q_len+1;i++){
		d[i] =new T[2];
		for(uint j=0;j<2;j++){
			d[i][j] = 0;
		}
	}


	d[0][0] = 0;// d[0][0]=c(1)-q(0)
	d[0][1] = INT_MAX;
	for(uint i=1;i<q_len+1;i++){
		d[i][s]=INT_MAX;//(C[1]-Q[i])*(C[1]-Q[i]);
	}

	s=1-s;
	uint s_xor;
	T dist = 0;
	for(uint j=1;j<c_len+1;j++){
		d[0][s] = INT_MAX;
		for(uint i=1;i<q_len+1;i++){
			s_xor = 1-s;
			dist = (C[j]-Q[i])*(C[j]-Q[i]);
			d[i][s] = dist + min(d[i-1][s],min(d[i][s_xor],d[i-1][s_xor]));
		}
		s = 1-s;
	}

	s_xor = 1-s;
	dist = d[q_len][s_xor];

	for(uint i=0;i<c_len+1;i++){
		delete[] d[i];
	}
	delete[] d;

	return dist;
}




/**
 * r:  Sakoe-Chiba Band
 * TODO:
 * 1. the data points of Q and C is from 1 to cq_len
 * Q[0]=C[0]=infinity
 * refer to paper "Accelerating dynamic time warping subsequence search with GPUs and FPGAs, ICDM, but the Algorithm in Table2 is wrong (or confused). This is the correct one
 *
 *
 * 2. Add Sakoe-Chiba Band to compute the DTW
 * refer to paper: Sakoe, Hiroaki, and Seibi Chiba. "Dynamic programming algorithm optimization for spoken word recognition."
 *  Acoustics, Speech and Signal Processing, IEEE Transactions on 26, no. 1 (1978): 43-49.
 *
 * 3. =======================================
 * pseudo code  Algorithm:
 * s = 0
 * for i=0 to m:
 * 	   d(i,s) =  infinity
 *
 * s = s XOR 1//Xor operation
 *
 * for j = 1 to n
 * 	   for i = j-r to j+r
 * 	       d(i,s) = |C(j)-Q(i)| + min(d(i-1),s), d(i, s XOR 1), d(i-1, s XOR 1)) //j - r <= i <= j + r
 * 	   s = s XOR 1
 *
 * return d(n,s XOR 1)
 *
 *
 *
 */
template <class T>
__host__ __device__ T dtw_DP_SCBand(const T* Q, uint q_len,const T* C, uint c_len, uint r){

	uint s=0;
	T** d = new T*[q_len+1];//create matrix for wrap path
	for(uint i=0;i<q_len+1;i++){
		d[i] =new T[2];
		for(uint j=0;j<2;j++){
			d[i][j] = 0;
		}
	}


	d[0][0] = 0;// d[0][0]=c(1)-q(0)
	d[0][1] = INT_MAX;
	for(uint i=1;i<q_len+1;i++){
		d[i][s]=INT_MAX;//(C[1]-Q[i])*(C[1]-Q[i]);
	}

	s=1-s;
	uint s_xor;
	T dist = 0;

	for(uint j=1;j<c_len+1;j++){

		uint start_i = j<r+1 ? 1 : (j-r);
		uint end_i = j+r>(q_len)? q_len:(j+r);

		d[start_i-1][s] = INT_MAX;
		s_xor = 1-s;
		if(j+r<=(q_len)){
		d[j+r][s_xor] = INT_MAX;
		}

		for(uint i = start_i;i<=end_i;i++){

			dist = (C[j]-Q[i])*(C[j]-Q[i]);
			d[i][s] = dist + min(d[i-1][s],min(d[i][s_xor],d[i-1][s_xor]));
		}

		s = 1-s;
	}

	s_xor = 1-s;
	dist = d[q_len][s_xor];

	for(uint i=0;i<c_len+1;i++){
		delete[] d[i];
	}
	delete[] d;

	return dist;
}


/**
 * the data points of Q and C is from 1 to cq_len
 * Q[0]=C[0]=infinity
 * recursive implementation
 */
template <class T>
__host__ __device__ T dtw_recur(const T*Q, uint q_len, const T* C, uint c_len){

	return dtw_AuxRecur(Q,q_len,C,c_len);
}


/**
 * auxiliary function for recursive function of DTW
 */
template <class T>
__host__ __device__ T dtw_AuxRecur(const T* Q, uint qi, const T* C, uint cj){



	if(qi==0&&cj==0){
		return 0;
	}

	if(qi==0||cj==0){
		return INT_MAX;
	}

	T di = Q[qi]-C[cj];
	di=di*di;



	T d_i_1_j_1 = dtw_AuxRecur(Q,qi-1,C,cj-1);
	T d_i_1_j = dtw_AuxRecur(Q,qi-1,C,cj);
	T d_i_j_1 = dtw_AuxRecur(Q,qi,C,cj-1);

	return di+min(d_i_1_j_1,min(d_i_1_j,d_i_j_1));

}



/**
 * the data points of Q and C is from 1 to cq_len
 * Q[0]=C[0]=infinity
 * recursive implementation for SCBand
 */
template <class T>
__host__ __device__ T dtw_recur_SCBand(const T*Q, uint q_len,const T* C, uint c_len, uint r){

	return dtw_AuxRecur_SCBand(Q,q_len,C,c_len,r);
}

/**
 * auxiliary function for recursive function of DTW_SCBand
 */
template <class T>
__host__ __device__ T dtw_AuxRecur_SCBand(const T* Q, uint qi, const T* C, uint cj,uint r){



	if(qi==0&&cj==0){
		return 0;
	}

	if(qi==0||cj==0){
		return INT_MAX;
	}

	if(qi>r+cj||cj>r+qi){
		return INT_MAX;
	}


	T di = Q[qi]-C[cj];
	di=di*di;



	T d_i_1_j_1 = dtw_AuxRecur_SCBand(Q,qi-1,C,cj-1,r);
	T d_i_1_j = dtw_AuxRecur_SCBand(Q,qi-1,C,cj,r);
	T d_i_j_1 = dtw_AuxRecur_SCBand(Q,qi,C,cj-1,r);

	return di+min(d_i_1_j_1,min(d_i_1_j,d_i_j_1));

}



template <class T>
__host__ __device__ T eu( const T* Q, uint sq, const T* C, uint sc, uint cq_len){

	T d = 0;
	for(uint i=0;i<cq_len;i++){
		d+=(C[i+sc]-Q[i+sq])*(C[i+sc]-Q[i+sq]);
	}
	return d;
}

//template class DistFunc<int>;
//template class DistFunc<float>;
