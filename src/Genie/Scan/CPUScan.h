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

struct CompareTopNode {
    bool operator()(const topNode& x, const topNode& y) const
    {
        return x.dis < y.dis;
    }
};

class CPUScan {
public:
	CPUScan();
	 virtual ~CPUScan();


	 template<class T, class DISTFUNC>
	 	//=============for cmputing top-k
	 void  CPU_computTopk(vector< vector <T> >& query, int k, vector<T> & data,DISTFUNC distFunc){
	 		long t=0;

	 			vector<vector<topNode> > resVec(query.size());

	 			for(uint i=0;i<query.size();i++){
	 				long start = clock();
	 				//vector<int> qr = convertQuery(*query[i]);
	 				CPU_compTopkItem(query[i],  k,  data, resVec[i], distFunc);//indexVec[i], distVec[i]);

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
	 void  CPU_compTopkItem(vector<T>& q, int k, vector<T> & data,
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
	 			double maxd=res[0].dis; int idx=0;
//	 			for(int r=0;r<k;r++){
//	 				if(res[r].dis >= maxd){
//
//	 					maxd=res[r].dis;
//	 					idx=r;
//
//	 				}
//	 			}

	 			//if smaller than maxd, replace
	 			if(di<=maxd){
	 				res[idx].dis = di; //_dist[idx]=di;
	 				res[idx].idx = i; ////_index[idx]=i;
	 				make_heap (res.begin(), res.end(), CompareTopNode());
	 			}
	 		}

	 		std::sort(res.begin(),res.end());


	 	}




	 void  computTopk_int_eu(vector< vector <int> >& query, int k, vector<int> & data){

	 	//	Eu_Func<int> eu_func;

	 		CPU_computTopk( query,  k,  data, Eu_Func<int> ());
	 	//	void  CPU_computTopk(vector< vector <T> >& query, int k, vector<T> & data,DISTFUNC distFunc);

	 }


	 void  computTopk_int_dtw_scBand(vector< vector <int> >& query, int k, vector<int> & data, int sc_band){


	 	//Dtw_SCBand_Func<int> dtw_func(sc_band);

	 	CPU_computTopk( query,  k,  data, Dtw_SCBand_Func<int>(sc_band));
	 }

};

#endif /* CPUSCAN_H_ */
