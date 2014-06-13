/*
 * DataProcess.cpp
 *
 *  Created on: Dec 23, 2013
 *      Author: zhoujingbo
 */

#include "DataProcess.h"
#include <limits.h>
#include <algorithm>

DataProcess::DataProcess() {
	// TODO Auto-generated constructor stub

	max = numeric_limits<double>::min();
	min = numeric_limits<double>::max();

}

DataProcess::~DataProcess() {
	// TODO Auto-generated destructor stub
}

vector<string> DataProcess::split(string& str, const char* c) {
	char *cstr, *p;
	vector<string> res;
	cstr = new char[str.size() + 1];
	strcpy(cstr, str.c_str());
	p = strtok(cstr, c);
	while (p != NULL) {
		res.push_back(p);
		p = strtok(NULL, c);
	}
	delete[] cstr;
	return res;
}

string DataProcess::eraseSpace(string origin) {
	int start = 0;
	while (origin[start] == ' ')
		start++;
	int end = origin.length() - 1;
	while (origin[end] == ' ')
		end--;
	return origin.substr(start, end - start + 1);
}

template<typename F>
/**
 * _data:output result
 */
void DataProcess::ReadFile(const char* fname, int fcol,
		F (*atoX)(const char*), vector<F>& _data) {


	string line;
	ifstream ifile(fname);

	_data.clear();

	if (ifile.is_open()) {
		while (getline(ifile, line)) {

			vector<string> nstring = split(line, ",");
			string myString = eraseSpace(nstring[fcol]);

			//cout<<"my string"<<myString<<endl;
			F value = (*atoX)(myString.c_str());

			if (value < min&&value>=0)
				min = value;
			if (value > max)
				max = value;

			//if the missing value is -1, ignore it
			if ( value>=0) {
				_data.push_back(value);
			}
		}
	}

	ifile.close();
}


/**
 * direct read the time series data, without preprocessing, and the data type is integer
 */
void DataProcess::ReadFileInt(const char * fname, int fcol,vector<int>& _data) {

	ReadFile(fname, fcol, &atoi, _data);

}

void DataProcess::ReadFileDouble(const char * fname, int fcol, vector<double>& _data) {

	return ReadFile(fname, fcol,&atof,_data);

}


template<typename E>
/**
 * _data: output result
 */
void DataProcess::ReadFile(const char * fname,
		E (*atoX)(const char *), vector<vector<E> >& _data) {

	//vector<vector<E>*>* data = new vector<vector<E>*>;
	string line;
	ifstream ifile(fname);

	_data.clear();

	if (ifile.is_open()) {

		while (getline(ifile, line)) {

			vector<string> nstring = split(line, ", ");
			vector<E> lv;

			for (int j = 0; j < nstring.size(); j++) {
				E lvi = (*atoX)(nstring[j].c_str());
				lv.push_back(lvi);
			}

			_data.push_back(lv);
		}
	}

	ifile.close();
	//return data;
}

/**
 * _data:output result
 */
void DataProcess::ReadFileInt(const char * fname, vector<vector<int> >& _data) {
	return ReadFile(fname, &atoi,_data);
}


/**
 *
 */
void DataProcess::ReadFileWidBucket(const char * fname, int fcol,
		double dw, vector<int>& _dbRes) {

	vector<double> data;
	ReadFileDouble(fname, fcol, data);
	Bucketized(data, dw, _dbRes);

}

void DataProcess::ReadFileBucket(const char * fname, int fcol,
		int bukNum, vector<int> _dbRes) {

	vector<double> data;
	ReadFileDouble(fname, fcol,data);

	double dw = getBukWidth(bukNum);

	Bucketized(data, dw,_dbRes);

}

void DataProcess::Bucketized(vector<double>& data, double dw, vector<int>& _dbRes) {

	_dbRes.clear();
	_dbRes.resize(data.size());
	for (uint i = 0; i < data.size(); i++) {
		int wi = rangPartition(data.at(i), dw);
		_dbRes[i] = wi;

	}

}

double DataProcess::getBukWidth(int bukNum) {

	double dw = (max - min) / (bukNum - 1);
	return dw;

}

int DataProcess::rangPartition(double v, double dw) {

	int vi = (int) (v / dw);
	return vi;
}

/**
 *
 *
 * keyLow: the nuber of bits for value, the rest of the bits are left for dimensions.
 * In this method, we write file with the bit shift compared with method writeQueryFile(const char* outfile, map<uint,vector<int>* > &query)
 */
void DataProcess::writeInvListQueryFile(const char * outfile,
		map<uint, vector<int> > &query, int keyLow) {
	ofstream outf;
	outf.open(outfile, ios::binary | ios::out);

	for (map<uint, vector<int> >::iterator it = query.begin();
			it != query.end(); ++it) {
		vector<int> v = it->second;

		for (int i = 0; i < v.size(); i++) {
			int vit = v.at(i);
			int d = (i << keyLow) + vit; //make the composite key
			outf << d << " ";
		}

		outf << endl;
	}

	outf.close();
}

/**
 * in this method, we do not write the file with bit shift,
 * compared with writeInvListQueryFile(const char * outfile,map<uint,vector<int>* > &query, int keyLow)
 */
void DataProcess::writeQueryFile(const char* outfile,
		map<uint, vector<int> > &query) {

	ofstream outf;
	outf.open(outfile, ios::binary | ios::out);

	for (map<uint, vector<int> >::iterator it = query.begin();
			it != query.end(); ++it) {
		vector<int> v = it->second;

		for (int i = 0; i < v.size(); i++) {
			int vit = v.at(i);
			outf << vit << " ";
			//cout<< vit << " ";
		}
		outf << endl;
		// cout<<endl;
	}

	outf.close();

}

void DataProcess::writeBinaryQueryFile(const char* outfile,
		map<uint, vector<int>*> &query){
	ofstream outf;
	outf.open(outfile, ios::binary | ios::out);

	for (map<uint, vector<int>*>::iterator it = query.begin();
			it != query.end(); ++it) {
		vector<int> *v = it->second;

		for (int i = 0; i < v->size(); i++) {
			int vit =  v->at(i);
			outf.write((char*) &vit, sizeof(vit));
		}
		outf << endl;
		// cout<<endl;
	}
	outf.close();

}



void DataProcess::writeBinaryFile(const char * outfile, int numOfDim, int maxValuePerDim,
		 map<uint, vector<int> >& im) {

	ofstream outf;
	outf.open(outfile, ios::binary | ios::out);

	int countK = 0;
	int countE = 0;

	//write number of dimension and number of features

	outf.write((char*) &numOfDim, sizeof(uint));
	outf.write((char*) &maxValuePerDim, sizeof(uint));

	for (map<uint, vector<int> >::iterator it = im.begin(); it != im.end();
			++it) {
		uint key = it->first;
		outf.write((char*) &key, sizeof(uint));
		countK++;

		vector<int> v = it->second;
		uint s = v.size();

		outf.write((char*) &s, sizeof(s));

		for (int vi = 0; vi < v.size(); vi++) {
			countE++;
			int vit = (v)[vi];
			outf.write((char*) &vit, sizeof(vit));
		}
	}

	cout << "The total number of keys in idx is:" << countK << endl;
	cout << "The total number of elements in idx is:" << countE << endl;
	outf.close();

}




void DataProcess::run() {

}

