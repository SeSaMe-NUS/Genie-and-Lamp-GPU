/*
 * WrapperDTW.cpp
 *
 *  Created on: Apr 1, 2014
 *      Author: zhoujingbo
 */

#include "WrapperScan.h"
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <time.h>

#include "../AppManager/BladeLoader.h"
#include "../AppManager/IndexBuilder/DataProcess.h"
#include "Scan/GPUScan.h"
#include "Scan/CPUScan.h"


using namespace std;




WrapperScan::WrapperScan()
{
	// TODO Auto-generated constructor stub

}

WrapperScan::~WrapperScan()
{
	// TODO Auto-generated destructor stub
}


void WrapperScan::DTWQuery(vector<int>& data, vector<vector<int> >& qdata, int dim, int topk,
		vector<vector<int> >& _resIdx, vector<vector<int> >& _dist){


	//copy time series into GPU
	int ts_len = data.size();//
	int* ts= (data).data();//get raw data

	int tq_num = qdata.size();
	int  res_buff_len = tq_num*(ts_len-dim+1);
	int* res_buff = new int[res_buff_len];

	//int** resIdx = new int*[tq_num];//the return result
	_resIdx.resize(tq_num);

	//prepare the query file
	int tq_len = 0;
	tq_len = dim*tq_num;
    int* tq =vec2Ddata(qdata);//convert query data into one dimesnion array


	startGPU(ts,  ts_len,  tq,  tq_num,  dim, res_buff);//use GPU to compute DTW
	//selectQueryRes(int* res_buff, int topk,int tq_num, int ts_len, int dim, vector<vector<int> >&  resIdx)
	selectQueryRes(res_buff, topk, tq_num,  ts_len,  dim, _resIdx, _dist);//select top-k query

	delete[] res_buff;
	//delete ts;
	delete[] tq;

	//return resIdx;
}


int WrapperScan::runDTWQueryInt(string inputFilename, string queryFile,int columnAv, int dim, int tq_num, int topk){

	DataProcess dp;
	//load data
	vector<int> data;
	dp.ReadFileInt(inputFilename.c_str(),columnAv,data);
	cout<<"load data item:"<<data.size()<<endl;
	//load query
	vector<vector<int> > qdata;
	dp.ReadFileInt(queryFile.c_str(),qdata);
	cout<<"load query items:"<<qdata.size()<<endl;

	long time=0;
	long start = clock();
	vector<vector<int> > resIdx;
	vector<vector<int> > dist;
	DTWQuery(data,  qdata,  dim,  topk, resIdx, dist);
	long end=clock();
	time = end-start;
	cout<<"the running time of GPU scan is:"<<(double)time /CLOCKS_PER_SEC <<endl;

	return 0;

}

int WrapperScan::runDTWQueryByInput(){


	//load time series data
	string inputFilename;
	cout << "Please enter input filename" << endl;
	cin >> inputFilename;

	cout << "Which column is going to be used? (Start with column 0)" << endl;
	int columnAv;
	cin >> columnAv;

	string queryFile;
	cout<< "Please enter query filename(note: if selecting data from datafile, please input *";
	cin >> queryFile;

	//define the query
	cout<<"Please input query number"<<endl;
	int tq_num=1;
	cin>>tq_num;

	int dim=32;
	cout<<"Please input dimensions:"<<endl;
	cin>>dim;

	int k = 8;
	cout<<"please input the top-k (NN):"<<endl;
	cin>>k;

	runDTWQueryInt(inputFilename, "", columnAv,  dim,  tq_num, k);

	return 0;
}




int WrapperScan::runGPUDTW(){
	//runQueryByInput();
	//runDTWQueryInt( "data/calit2/CalIt2_7.csv", "data/calit2/CalIt2_7_d8_q16_dir.query", 3,  8,  16,  3);
	runDTWQueryInt( "data/test/sequenceTest.csv", "data/test/sequenceTest_ql16_gqn2_group.query", 0,  16,  2,  10);
	return 0;
}

void loadData(const string queryFile,const string dataFile,int dataFile_col,vector<vector<int> >& qdata,vector<int>& data){

		DataProcess dp;
		//dp.ReadFileInt("data/calit2/CalIt2_7.csv",3, data);
		dp.ReadFileInt(dataFile.c_str(),dataFile_col, data);
		cout<<"data size:"<<data.size()<<endl;
		dp.printMaxMin();

		//dp.ReadFileInt("data/calit2/CalIt2_7_d8_q16_dir.query", qdata);
		dp.ReadFileInt(queryFile.c_str(), qdata);

		cout<<"query size:"<<qdata.size()<<endl;


}

int WrapperScan::runCPUEu(){
		int topk = 5;
		int dimensionNum = 32;
		int queryNum = 16;
		string dataFileHolder = "data/Dodgers/Dodgers";
		int dataFile_col  = 1;

		vector<vector<int> > qdata;
		vector<int> data;

		stringstream ssDataFile;
		ssDataFile<<dataFileHolder<<".csv";
		string dataFile = ssDataFile.str();
		cout<<dataFile<<endl;


		stringstream ssQueryFile;
		ssQueryFile<<dataFileHolder<<"_d"<< dimensionNum<<"_q"<<queryNum <<"_dir.query";
		string queryFile = ssQueryFile.str();
		cout<<queryFile<<endl;


		loadData(queryFile,dataFile,dataFile_col, qdata,data);

		CPUScan cscan;

		cscan.computTopk_int_eu(qdata, topk, data);

		return 0;

}

int WrapperScan::runCpuDtw_scBand(){

			int topk = 10;
			int sc_band = 2;
			vector<vector<int> > qdata;
			vector<int> data;
			string dataFile = "data/Dodgers/Dodgers.csv";
			int dataFile_col = 1;
			string queryFile = "data/Dodgers/Dodgers_d32_q16_dir.query";

			//string dataFile = "data/test/sequenceTest.csv";
			//int dataFile_col  = 0;
			//string queryFile = "data/test/sequenceTest_ql16_gqn2_group.query";



			loadData(queryFile,dataFile,dataFile_col, qdata,data);


			CPUScan cscan;
			long start = clock();
			cscan.computTopk_int_dtw_scBand(qdata, topk, data,sc_band);
			long end = clock();

			cout<<"the time of top-"<< topk <<" in CPU version is:"<< (double)(end-start) / CLOCKS_PER_SEC <<endl;

			return 0;
}

