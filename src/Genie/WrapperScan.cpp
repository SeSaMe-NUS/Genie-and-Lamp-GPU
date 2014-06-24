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
#include "../CONSTANT.h"


using namespace std;




WrapperScan::WrapperScan()
{
	// TODO Auto-generated constructor stub

}

WrapperScan::~WrapperScan()
{
	// TODO Auto-generated destructor stub
}



void loadData(const string queryFile,const string dataFile,int dataFile_col,vector<vector<int> >& qdata,vector<int>& data){

		DataProcess dp;
		//dp.ReadFileInt("data/calit2/CalIt2_7.csv",3, data);
		dp.ReadFileInt(dataFile.c_str(),dataFile_col, data);
		cout<<"data size:"<<data.size()<<endl;
		//dp.printMaxMin();

		//dp.ReadFileInt("data/calit2/CalIt2_7_d8_q16_dir.query", qdata);
		dp.ReadFileInt(queryFile.c_str(), qdata);

		cout<<"query size:"<<qdata.size()<<endl;


}

int WrapperScan::runCPUEu(){
		int topk = TOPK;
		int dimensionNum = DIMENSIONNUM;
		int queryNum = QUERYNUM;
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

int WrapperScan::runGPUEu(){
			int topk = TOPK;
			int dimensionNum = DIMENSIONNUM;
			int queryNum = QUERYNUM;
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

			GPUScan gscan;
			gscan.computTopk_int_eu(qdata,topk,data);

			return 0;
}

int WrapperScan::runCpu_Dtw_scBand(){

			int sc_band = 8;
				int topk = TOPK;
				int dimensionNum = DIMENSIONNUM;
				int queryNum = QUERYNUM;
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
			long start = clock();
			cscan.computTopk_int_dtw_scBand(qdata, topk, data,sc_band);
			long end = clock();

			cout<<"the time of top-"<< topk <<" in CPU version is:"<< (double)(end-start) / CLOCKS_PER_SEC <<endl;

			return 0;
}



int WrapperScan::runGpu_Dtw_scBand(){
			int sc_band = 8;
			int topk = TOPK;
			int dimensionNum = DIMENSIONNUM;
			int queryNum = QUERYNUM;
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

			GPUScan gscan;
			gscan.computTopk_int_dtw_scBand(qdata,topk,data,sc_band);

			return 0;
}

