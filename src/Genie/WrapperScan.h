/*
 * WrapperDTW.h
 *
 *  Created on: Apr 1, 2014
 *      Author: zhoujingbo
 */

#ifndef WRAPPERSCAN_H_
#define WRAPPERSCAN_H_
#include <vector>
#include <string>
#include "Scan/UtlScan.h"

using namespace std;



class WrapperScan {
public:
	WrapperScan();
	virtual ~WrapperScan();






	//====
	int runCPUEu();
	int runCpuDtw_scBand();

	int runGPUEu();

};

/*
 *
 *
 * private:
	int* vec2data(std::vector<int> data);
	int runDTWQueryInt(std::string inputFilename, std::string queryFile,int columnAv, int dim, int tq_num, int k);
	int runDTWQueryByInput();

public:
	int runGPUDTW();
 * void DTWQuery(vector<int>& data, vector<vector<int> >& qdata, int dim, int topk,
			vector<vector<int> >& _resIdx, vector<vector<int> >& _dist);
 *
 */

#endif /* WRAPPERSCAN_H_ */
