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
	int runGPUEu();

	int runCpu_Dtw_scBand();
	int runGpu_Dtw_scBand();


};



#endif /* WRAPPERSCAN_H_ */
