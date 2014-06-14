/*
 * CPUScan.h
 *
 *  Created on: Jun 11, 2014
 *      Author: zhoujingbo
 */

#ifndef CPUSCAN_H_
#define CPUSCAN_H_
#include <stdio.h>
#include <unistd.h>

#include <vector>
#include <iostream>
#include <limits.h>
#include <algorithm>
using namespace std;
#include "UtlScan.h"
#include "DistFunc.h"


class CPUScan {
public:
	__host__ CPUScan();
	__host__ virtual ~CPUScan();


	template<class T, class DISTFUNC>
	//=============for cmputing top-k
	__host__ void  computTopk(vector< vector <T> >& query, int k, vector<T> & data,DISTFUNC distFunc){
		long t=0;

			vector<vector<topNode> > resVec(query.size());

			for(uint i=0;i<query.size();i++){
				long start = clock();
				//vector<int> qr = convertQuery(*query[i]);
				compTopkItem(query[i],  k,  data, resVec[i], distFunc);//indexVec[i], distVec[i]);

				long end=clock();
				t+=(end-start);
			}


			cout<<"the time of top-"<< k <<" in CPU version is:"<< (double)t / CLOCKS_PER_SEC <<endl;
			cout<<"the result is:"<<endl;
			for(int i=0;i<query.size();i++){
				//cout<<"print result of Query["<<i<<"]"<<endl;
				//cout<<"start ================================"<<endl;
				for(int j=0;j<k;j++){
					//resVec[i][j].print();//<<endl;
				}
				//cout<<"end =================================="<<endl;
			}
	}

	template<class T, class DISTFUNC>
	__host__ void  compTopkItem(vector<T>& q, int k, vector<T> & data,
				vector<topNode>& res,DISTFUNC distFunc){

		int dim = q.size();
		res.clear();
		res.resize(k);
		for(int r=0;r<k;r++){
			res[r].idx = 0;
			res[r].dis = (double)INT_MAX;
		}

		for(uint i=0;i<data.size()- dim ;i++){

			T di=0;

			di = distFunc.dist(q.data(),0,data.data(),i,dim);


			//compute the max value in this array
			double maxd=-1; int idx=-1;
			for(int r=0;r<k;r++){
				if(res[r].dis >= maxd){

					maxd=res[r].dis;
					idx=r;

				}
			}

			//if smaller than maxd, replace
			if(di<=maxd){
				res[idx].dis = di; //_dist[idx]=di;
				res[idx].idx = i; ////_index[idx]=i;
			}
		}

		std::sort(res.begin(),res.end());

	}

	__host__ void  computTopk_int_eu(vector< vector <int> >& query, int k, vector<int> & data){


		Eu_Func<int> eu_func;

		computTopk( query,  k,  data, eu_func);
	}
	__host__ void  computTopk_int_dtw_scBand(vector< vector <int> >& query, int k, vector<int> & data, int sc_band){


		Dtw_SCBand_Func<int> dtw_func;
		dtw_func.sc_band = sc_band;

		computTopk( query,  k,  data, dtw_func);
	}

};

#endif /* CPUSCAN_H_ */
