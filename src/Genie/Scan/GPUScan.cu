/*
 * GPUScan.cpp
 *
 *  Created on: Jun 16, 2014
 *      Author: zhoujingbo
*/

#include "GPUScan.h"


#include <iostream>
using namespace std;

#include "TemplateFunctions/GPUScanFunctions.h"
#include "DistFunc.h"
#include "UtlScan.h"

#include <thrust/extrema.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <sys/time.h>

using namespace thrust;

GPUScan::GPUScan() {
	// TODO Auto-generated constructor stub

}

GPUScan::~GPUScan() {
	// TODO Auto-generated destructor stub
}


void convterIntToFloat(vector<vector<int> >& query,vector<vector<int> >& bladeData,
		vector<vector<float> >& query_flt,vector<vector<float> >& bladeData_flt){
	//convert to float
	bladeData_flt.resize(bladeData.size());
	for(int i=0;i<bladeData.size();i++){
		bladeData_flt[i].resize(bladeData[i].size());
		for(int j=0;j<bladeData[i].size();j++){
			bladeData_flt[i][j] = (float)bladeData[i][j];
		}
	}

	query_flt.resize(query.size());
	for(int i=0;i<query.size();i++){
		query_flt[i].resize(query[i].size());
		for(int j=0;j<query[i].size();j++){
			query_flt[i][j] = (float)query[i][j];
		}
	}

}

void GPUScan::computTopk_int_eu(vector<vector<int> >& query, int k,
		vector<int> & data) {


	vector<int> query_blade_id(query.size(),0);

	vector<vector<int> > bladeData;
	bladeData.push_back(data);
	vector<vector<topNode> > topk_result_idx;

	vector<int> topk_vec(query.size(),k);


	vector<vector<float> > query_flt;
	vector<vector<float> > bladeData_flt;

	 convterIntToFloat(query,bladeData, query_flt, bladeData_flt);

	struct timeval start;
	struct timeval end;

	cudaEvent_t start_t, stop_t;
	float elapsedTime;
	cudaEventCreate(&start_t);
	cudaEventCreate(&stop_t);
	cudaEventRecord(start_t, 0);

	GPU_computeTopk(query_flt, query_blade_id, bladeData_flt, topk_vec, Eu_Func<float>(), topk_result_idx );

	cudaEventRecord(stop_t, 0);
	cudaEventSynchronize(stop_t);
	cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
	cout << "GPU SCAN Time used: " << elapsedTime/1000 << endl;

	for(int i=0;i<topk_result_idx.size();i++){
		cout<<"query item ["<<i<<"]"<<endl;
		for(int j=0;j<topk_result_idx[i].size();j++){
			cout<<"query item ["<<i<<"] result "<< j<<":"<<topk_result_idx[i][j].idx<<" dist:"<<topk_result_idx[i][j].dis<<endl;
		}
		cout<<endl;
	}


}



void GPUScan::computTopk_int_dtw_scBand(vector<vector<int> >& query, int k,
		vector<int> & data, int sc_band) {



		vector<int> query_blade_id(query.size(),0);

		vector<vector<int> > bladeData;
		bladeData.push_back(data);
		vector<vector<topNode> > topk_result_idx;

		vector<int> topk_vec(query.size(),k);


		vector<vector<float> > query_flt;
		vector<vector<float> > bladeData_flt;

		convterIntToFloat(query,bladeData, query_flt, bladeData_flt);
		struct timeval start;
		struct timeval end;

		gettimeofday(&start, NULL);
		GPU_computeTopk(query_flt, query_blade_id, bladeData_flt, topk_vec, Dtw_SCBand_Func_modulus<float>(sc_band),topk_result_idx);
		gettimeofday(&end, NULL);
		float delta_time = (end.tv_sec* 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
		//cout << "GPU SCAN Time used: " << delta_time << endl;
		for (int i = 0; i < topk_result_idx.size(); i++) {
		//cout << "query item [" << i << "]" << endl;
		for (int j = 0; j < topk_result_idx[i].size(); j++) {
			//cout<<"query item ["<<i<<"] result "<< j<<":"<<topk_result_idx[i][j].idx<<" dist:"<<topk_result_idx[i][j].dis<<endl;
		}
		//cout << endl;
	}
}
