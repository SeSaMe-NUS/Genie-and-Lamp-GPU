/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

//==include cuda file and lib
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
 # error printf is only supported on devices of compute capability 2.0 and
 higher, please compile with -arch=sm_20 or higher
#endif



//==include c++ library file
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <time.h>
using namespace std;


 //===inlucde customized file
#include "DistFunc.h"
#include "GPUScan.h"

//Cuda code


 const int ShareMemSize = 1024;



 __host__ __device__ void printIntArray(int* data,int len){

 	printf("# print int array\n");
 	for(int i=0;i<len;i++){
 		printf(" %d",data[i]);
 	}
 	printf("\n");

 }

 __host__ __device__ void printFloatArray(float* data,int len){

 	printf("# print float array\n");
 	for(int i=0;i<len;i++){

 		printf(" %f",data[i]);

 	}
 	printf("\n");

 }


 __global__ void ScanQuery(const int* dev_tq,const int* dev_ts,int* dev_res_buff,int dim,int ts_len, int tq_num, int slides_per_thread, int query_per_thread){

	 __shared__ int s_tq[ShareMemSize];



	 //load query into shared memory
	 int qI = threadIdx.x*query_per_thread;
	 int tq_len = tq_num*dim;

	 for(uint j=0;j<query_per_thread;j++){
		 for(uint i=(qI+j)*dim;i<(qI+j+1)*dim && i<tq_len;i++){
			 s_tq[i]=dev_tq[i];
		 }
	 }


	 __syncthreads();

	 int tIdx = blockDim.x*blockIdx.x+threadIdx.x;

	 int I = tIdx*slides_per_thread;

	 //skip if idx is outside the boundary of array
	 if(I>ts_len-dim-slides_per_thread+1){
		 return;
	 }


	 //load the data within this thread
	 int* ts_thread = new int[dim+slides_per_thread];
	 for(int s=I;s<I+dim+slides_per_thread;s++){
		 //
		 ts_thread[s-I] = dev_ts[s];
	 }

	//DistFunc<int> df;
	int sum = 0;
	//scan every possible time series
	for (int i = 0; i < slides_per_thread; i++) {
		//scan every query
		for (int k = 0; k < tq_num; k++) {

			//sum = df.dtw(s_tq,k*dim,ts_thread,i,dim);
			//sum = eu(s_tq,k*dim,ts_thread,i,dim);
			sum = dtw_SCBand(s_tq,k*dim,ts_thread,i,dim,4);//for debug


			dev_res_buff[k * (ts_len - dim + 1) + i + I] = sum;//store the DTW
		}

	}
	delete [] ts_thread;
	 return;
 }



int* vec2Ddata(vector<vector<int> >& data){
	uint row = data.size();
	uint col = data.at(0).size();

	int *res = new int[row*col];

	for(uint i=0;i<row;i++){
		vector<int> di = data.at(i);
		for(uint j =0;j<col;j++){
			res[i*col+j] = di.at(j);
		}
	}

	return res;
}




template<class T>
void printVector(vector<T>& data){
	cout<<"print vetor. len:"<<data.size()<<endl;
	for(int i=0;i<data.size();i++){
		cout<<"id: "<<i<<" value:"<<data[i]<<endl;
	}
	cout<<"element of vec printed:"<<data.size()<<endl;
	cout<<endl;
}


int startGPU(int* ts, int ts_len, int* tq, int tq_num, int dim, int* res_buff){
	//copy data to gpu
	int* dev_ts;
	cudaMalloc((void **) &dev_ts, ts_len*sizeof(int));
	cudaMemcpy(dev_ts, ts, ts_len*sizeof(int),cudaMemcpyHostToDevice);

	//copy query to gpu
	int* dev_tq;
	int tq_len = dim*tq_num;
	cudaMalloc((void **)&dev_tq,tq_len*sizeof(int));
	cudaMemcpy(dev_tq, tq,tq_len*sizeof(int),cudaMemcpyHostToDevice);

	//start kernal
	//number of sliding windows per thread
	int gridDim = 5;
	int blockDim = 256;
	int numThreads = gridDim*blockDim;
	int slides_per_thread = ceil(((double)(ts_len-dim+1))/numThreads);
	int query_per_thread = ceil(((double)tq_num)/blockDim);
	cout<<"slides per thread per block:"<<slides_per_thread<<endl;
	cout<<"query per thread per block:"<<query_per_thread<<endl;


	//define result buffer
	//define the buffer for distance result
	int res_buff_len = (tq_num)*(ts_len-dim+1);
	cout<<"res_buff_len:"<<res_buff_len<<endl;
	int* dev_res_buff;

	cudaMalloc((void **)&dev_res_buff,res_buff_len*sizeof(int));


	ScanQuery<<<gridDim,blockDim>>>(dev_tq,dev_ts,dev_res_buff, dim,ts_len, tq_num,slides_per_thread,query_per_thread);


	cudaDeviceSynchronize();
	cudaMemcpy(res_buff,dev_res_buff,res_buff_len*sizeof(int),cudaMemcpyDeviceToHost);

	//printFloatArray(res_buff,slides_per_thread*2);

	cudaFree(dev_ts);
	cudaFree(dev_tq);
	cudaFree(dev_res_buff);

	return 0;
}



