/*
 * GPUFunctions.h
 *
 *  Created on: Jun 14, 2014
 *      Author: zhoujingbo
*/

#ifndef GPUFUNCTIONS_H_
#define GPUFUNCTIONS_H_
#include <stdlib.h>
#include "../../lib/bucket_topk/bucket_topk.h"
#include "../UtlScan.h"

#include <vector>
using namespace std;



template<class T, class DISTFUNC>
__global__ void computeScanDist(
	const T* queryData, const int* queryData_endIdx, const int* query_blade_data_id,
	const T* blade_data, const int* data_endIdx,
	DISTFUNC distFunc,
	T* result_vec,//output
	int* result_vec_endIdx
 	 );


void inline getMinMax(device_vector<float>& d_data_vec, float & min, float& max);

template <class DISTFUNC>
 void GPU_computeTopk(vector<vector<float> >& query, vector<int>& query_blade_id_vec,
 		 vector<vector<float> >& bladeData, vector<int>& topk,
 		DISTFUNC distFunc,
 		 vector<vector<topNode> >& _topk_result_idx);



#include "GPUScanFunctions.inc"

#endif  /* GPUFUNCTIONS_H_ */
