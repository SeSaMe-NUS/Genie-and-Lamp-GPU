/*
 * CudaDTW.h
 *
 *  Created on: Apr 1, 2014
 *      Author: zhoujingbo
 */

#ifndef CUDADTW_H_
#define CUDADTW_H_
#include <vector>

 int* vec2Ddata(vector< vector<int> >& data);

template <class T>
void selectMinK(int k, const T* data, int s, int e, std::vector<int>& _index){
	vector<T> dist;
	selectMinK(k, data, s, e, _index,  dist);
}



template<class T>
/**function: select the minimum value in an array (with start and end position), return the index of the values in the array
 * k: number of point to select
 * data:
 * s:
 *
 * output:
 * _index
 * _dist:
 */
void selectMinK(int k, const T* data, int s, int e, vector<int>& _index,
		vector<T>& _dist) {

	//int* index = new int[k];
	_index.resize(k,-1);
	_dist.resize(k,100000);

	//T* dist = new T[k];

	//for (int j = 0; j < k; j++) {
	//	_index[j] = -1;
	//	_dist[j] = 100000;
	//}

	for (int i = s; i < e; i++) {
		T d = data[i];

		//select the max item from buffer
		T maxd = -1;
		int idx = -1;
		for (int r = 0; r < k; r++) {

			if (_dist[r] >= maxd) {
				maxd = _dist[r];
				idx = r;
			}


		}

		if (d < maxd) {
			_dist[idx] = d;
			_index[idx] = i;
			//cout<<"with debug purpose test selectMinK maxd:"<<maxd<<" d:"<<d<< " id:"<<i<<" _dist[idx]:"<<_dist[idx]<<endl;
		}

	}

}


int startGPU(int* ts, int ts_len, int* tq, int tq_num, int dim, int* res_buff);

//template<class T>
//void selectQueryRes(int* res_buff, int topk,int tq_num, int ts_len, int dim, vector<vector<int> >& _resIdx);

template<class T>
void selectQueryRes(int* res_buff, int topk,int tq_num, int ts_len, int dim,
		vector<vector<int> >&  _resIdx, vector<vector<T> >& _dist){

	_resIdx.clear();
	_dist.clear();
	_resIdx.resize(tq_num);
	_dist.resize(tq_num);

	for(int i=0;i<tq_num;i++){

		selectMinK(topk,res_buff,i*(ts_len-dim+1),(i+1)*(ts_len-dim+1), _resIdx[i],_dist[i]);


		printf("# queryId %d\n",i);
		printf("fId	dist\n");
		for(uint j=0;j<topk;j++){//convert to one dimension array
			_resIdx[i][j]-=i*(ts_len-dim+1);//from one D array to two D array.
			printf("%d	%d \n",_resIdx[i][j], _dist[i][j]);
		}
		printf("\n");

		//printIntArray(print_resIdx,topk);

	}
}


#endif /* CUDADTW_H_ */
