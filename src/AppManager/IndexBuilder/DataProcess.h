/*
 * DataProcess.h
 *
 *  Created on: Dec 23, 2013
 *      Author: zhoujingbo
 */

#ifndef DATAPROCESS_H_
#define DATAPROCESS_H_

#include <iostream>
#include <fstream>
#include <limits>
#include <cstring>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include "UtlIndexBuilder.h"
#include "../../Genie/Scan/UtlScan.h"

using namespace std;

class DataProcess {

private:

	double min;
	double max;

public:
	DataProcess();
	virtual ~DataProcess();

private:
	vector<string> split(string& str, const char* c);
	string eraseSpace(string origin);
	int rangPartition(double v, double dw);

public:
	//vector<int>* Bucketized(vector<double>* data, double dw);
	void Bucketized(vector<double>& data, double dw, vector<int>& _dbRes);
	double getBukWidth(int bukNum);

//read from file
public:
	//used for reading data file
	template<typename F>
	void ReadFile(const char* fname, int fcol, F (*atoX)(const char *),vector<F>& _data);

	void ReadFileInt(const char* fname, int fcol,vector<int>& _data );
	void ReadFileDouble(const char* fname, int fcol, vector<double>& _data);

	//used for reading query file
	template<typename E>
	void ReadFile(const char * fname, E (*atoX)(const char *), vector<vector<E> >& _data);
	void ReadFileInt(const char * fname, vector<vector<int> >& _data);



	//vector<int>* ReadFileWidBucket(const char * fname, int fcol, double dw);
	void ReadFileWidBucket(const char * fname, int fcol, double dw, vector<int>& _dbRes);
	//vector<int>* ReadFileBucket(const char * fname, int fcol, int bukNum);
	void ReadFileBucket(const char * fname, int fcol, int bukNum, vector<int> _dbRes);

//write into binary file
public:
	void writeBinaryFile(const char * outfile, int numDim, int maxValuePerDim, map<uint, vector<int> >& im);
	//keyLow: the nuber of bits for value, the rest of the bits are left for dimensions.
	void writeInvListQueryFile(const char * outfile,
			map<uint, vector<int> > &query, int keyLow);
	void writeBinaryQueryFile(const char* outfile,map<uint, vector<int>*> &query);
	//void writeQueryFile(const char* outfile, map<uint, vector<int>*> &query);
	void writeQueryFile(const char* outfile,	map<uint, vector<int> > &query);

	void run();


	void printMaxMin(){
		std::cout<<" max ="<<max<<" min="<<min<<std::endl;
	}
};



#endif /* DATAPROCESS_H_ */
