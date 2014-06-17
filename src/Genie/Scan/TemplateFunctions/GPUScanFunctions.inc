/*
 * GPUScanFunctions.cpp
 *
 *  Created on: Jun 14, 2014
 *      Author: zhoujingbo
 */

#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>


using namespace thrust;
#define THREAD_PER_BLK 256  // must be greater or equal to MAX_DIM and not exceed 1024


/*
GPUScanFunctions::GPUScanFunctions() {
	// TODO Auto-generated constructor stub

}

GPUScanFunctions::~GPUScanFunctions() {
	// TODO Auto-generated destructor stub
}*/





/*
 * shared memory size: the maximum dimension of one query
*/
 template<class T, class DISTFUNC>
__global__ void computeScanDist(
	const T* query_vec, const int* query_vec_endIdx, const int* query_blade_data_id,
	const T* blade_data_vec, const int* blade_data_vec_endIdx,
	DISTFUNC distFunc,
	T* _result_vec,
	int* result_vec_endIdx
 	 ){
	 int bid = blockIdx.x;
	 extern __shared__ T sm_query[];
	 int query_start_id = (bid == 0)? 0 : query_vec_endIdx[bid - 1];
	 int query_end_id = query_vec_endIdx[blockIdx.x];

	 int queryLen = query_end_id-query_start_id;
	 int round = queryLen/blockDim.x + (queryLen%blockDim.x!=0);

	 for(int i=0;i<round;i++){
		 int idx = i*blockDim.x + threadIdx.x;

		 if(idx<queryLen){
			 sm_query[idx] = query_vec[idx+query_start_id];
		 }

	 }
	 __syncthreads();

	 int blade_id = query_blade_data_id[blockIdx.x];

	 int blade_start_idx = (blade_id == 0)? 0:blade_data_vec_endIdx[blade_id-1];
	 int blade_end_idx = blade_data_vec_endIdx[blade_id];
	 int blade_len = blade_end_idx - blade_start_idx;

	 int result_start_idx = (blockIdx.x == 0)? 0:result_vec_endIdx[blockIdx.x - 1];

	 int query_round = (blade_len)/blockDim.x + ((blade_len)%blockDim.x!=0);

	 for(int i=0;i<query_round;i++){
		 int idx = i*blockDim.x + threadIdx.x;
		 if(idx<blade_len-queryLen+1){
			 T res = distFunc.dist(sm_query, 0, blade_data_vec, blade_start_idx+idx, queryLen);
			 _result_vec[result_start_idx + idx] = res;

		 }
	 }

 }



template <class T>
struct ValueOfSeletion{
__host__ __device__ T valueOf(T data)
{
   		return data;
	}
};



void inline getMinMax(device_vector<float>& d_data_vec, float & min, float& max) {

 	device_vector<float>::iterator minData =
 			thrust::min_element(d_data_vec.begin(), d_data_vec.end());

 	device_vector<float>::iterator maxData = thrust::max_element(d_data_vec.begin(), d_data_vec.end());
 	device_vector<float> d_min(1), d_max(1);
 	thrust::copy(minData, minData + 1, d_min.begin());
 	thrust::copy(maxData, maxData + 1, d_max.begin());
 	host_vector<float> h_min,h_max;
 	h_min = d_min, h_max = d_max;

 	min = h_min[0];
 	max = h_max[0];
 }



/**
 * TODO:
 *
 * @parameters:
 * query_vec: a set of queries
 * query_blade_id_vec: the query is corresponding to which data blade
 * bladeData_vec: several data blades, the mapping from query to corresponding blades is recorded in query_blade_id_vec
 */
  template <class DISTFUNC>
  void GPU_computeTopk(vector<vector<float> >& query_vec, vector<int>& query_blade_id_vec,
 		 vector<vector<float> >& bladeData_vec, vector<int>& topk_vec ,
 		 DISTFUNC distFunc,
 		 vector<vector<topNode> >& _topk_result_idx//output
 		 ){

  	vector<float> std_query_vec;
  	vector<int> std_query_vec_idx;
  	int maxQueryLen = 0;
  	for(int i=0;i<query_vec.size();i++){
  		for(int j=0;j<query_vec[i].size();j++){

  			std_query_vec.push_back(query_vec[i][j]);

  		}

  		if(maxQueryLen<query_vec[i].size()){maxQueryLen = query_vec[i].size();}

  		std_query_vec_idx.push_back(query_vec[i].size());
  	}

  	host_vector<float> h_query_vec(std_query_vec);
  	host_vector<int> h_query_vec_idx(std_query_vec_idx);
  	host_vector<int> h_query_blade_id(query_blade_id_vec);


  	device_vector<float> d_query_vec = h_query_vec;
  	device_vector<int> d_query_vec_endIdx =h_query_vec_idx;
  	device_vector<int> d_query_blade_id = h_query_blade_id;
  	thrust::inclusive_scan(d_query_vec_endIdx.begin(),d_query_vec_endIdx.end(),d_query_vec_endIdx.begin());

  	vector<float> std_blade_data_vec;
  	vector<int> std_blade_data_idx;
  	for(int i=0;i<bladeData_vec.size();i++){
  		for(int j=0;j<bladeData_vec[i].size();j++){
  			std_blade_data_vec.push_back(bladeData_vec[i][j]);
  		}
  		std_blade_data_idx.push_back(bladeData_vec[i].size());
  	}

  	host_vector<float> h_blade_data_vec(std_blade_data_vec);
  	host_vector<int> h_blade_data_vec_size(std_blade_data_idx);

  	device_vector<float> d_blade_data_vec = h_blade_data_vec;
  	device_vector<int> d_blade_data_vec_size = h_blade_data_vec_size;
  	device_vector<int> d_blade_data_vec_endIdx(h_blade_data_vec_size.size());

  	thrust::inclusive_scan(d_blade_data_vec_size.begin(),d_blade_data_vec_size.end(),d_blade_data_vec_endIdx.begin());


  	host_vector<int> h_result_vec_size(query_vec.size());
  	for(int i=0;i<query_vec.size();i++){
  		int queryItem_blade_id = query_blade_id_vec[i];
  		h_result_vec_size[i] = bladeData_vec[queryItem_blade_id].size()-query_vec[i].size()+1;
  	}

  	device_vector<int> d_distComp_vec_size = h_result_vec_size;
	device_vector<int> d_distComp_vec_startIdx(h_result_vec_size.size());
	device_vector<int> d_distComp_vec_endIdx(h_result_vec_size.size());
  	thrust::exclusive_scan(d_distComp_vec_size.begin(),d_distComp_vec_size.end(),d_distComp_vec_startIdx.begin());
  	thrust::inclusive_scan(d_distComp_vec_size.begin(),d_distComp_vec_size.end(),d_distComp_vec_endIdx.begin());

  	device_vector<float> d_distComp_vec(d_distComp_vec_endIdx[d_distComp_vec_endIdx.size()-1]);



  	computeScanDist<<<query_vec.size(),THREAD_PER_BLK, maxQueryLen*sizeof(float)>>>(
  		raw_pointer_cast(d_query_vec.data()),
  		raw_pointer_cast(d_query_vec_endIdx.data()),
  		raw_pointer_cast(d_query_blade_id.data()),
  		raw_pointer_cast(d_blade_data_vec.data()),
  		raw_pointer_cast(d_blade_data_vec_endIdx.data()),
  		distFunc,
  		raw_pointer_cast(d_distComp_vec.data()),
  		raw_pointer_cast(d_distComp_vec_endIdx.data())
  	 	);

  	float min,max;
  	getMinMax(d_distComp_vec, min, max);


  	device_vector<int> d_topk_vec(topk_vec);
  	device_vector<int> d_topk_result_idx;



  	bucket_topk(&d_distComp_vec,
  			ValueOfSeletion<float>(), min, max,
  			&d_topk_vec,
  			&d_distComp_vec_startIdx, &d_distComp_vec_endIdx,
  			query_vec.size(),
  			&d_topk_result_idx);

     host_vector<int> h_topk_result_idx = d_topk_result_idx;


     host_vector<float> h_distComp_vec = d_distComp_vec;
     int res_count_idx = 0;
     _topk_result_idx.clear();
     _topk_result_idx.resize(topk_vec.size());
     for(int i=0;i<topk_vec.size();i++){
     	for(int j=0;j<topk_vec[i];j++){
     		int id = h_topk_result_idx[res_count_idx];
     		float dis = h_distComp_vec[id];
     		topNode tn(dis,id);
     		_topk_result_idx[i].push_back(tn);
     		res_count_idx++;
     	}
     }

  }


