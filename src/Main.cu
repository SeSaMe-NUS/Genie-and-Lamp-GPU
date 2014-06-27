/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>


#include "AppManager/WrapperIndexBuilder.h"
#include "AppManager/deviceDetector/deviceDetector.h"
#include "Genie/WrapperGPUKNN.h"
#include "Genie/WrapperScan.h"
#include "CONSTANT.h"
#include <vector>
using namespace std;

#include <thrust/device_vector.h>
using namespace thrust;

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char* argv[]) {
	for(int i=1; i<argc; i++)
	{
		if((strcmp(argv[i], "--topk")==0 || strcmp(argv[i], "-k")==0) && argv[i+1]!=NULL)
			sscanf(argv[i+1], "%d", &TOPK);
		if(strcmp(argv[i], "--dimension")==0 || strcmp(argv[i], "-d")==0)
			sscanf(argv[i+1], "%d", &DIMENSIONNUM);
		if(strcmp(argv[i], "--query")==0 || strcmp(argv[i], "-q")==0)
			sscanf(argv[i+1], "%d", &QUERYNUM);
	}
	cout << "TOPK = " << TOPK << ", DIMENSIONNUM = " << DIMENSIONNUM << ", QUERYNUM = " << QUERYNUM << endl;
	WrapperIndexBuilder wrapperIndexBuilder;
	//wrapperIndexBuilder.runBuilderIndex();

	WrapperGPUKNN wrapperGpuKnn;
	cout<<endl;
	cout<<"run GPU inverted index ==================="<<endl;
	//wrapperGpuKnn.runGPUKNN();



	WrapperScan wscan;
	cout<<endl;
	cout<<"run CPU Scan ==================="<<endl;
	//wscan.runCPUEu();
	//wscan.runCpu_Dtw_scBand();

	cout<<endl;
	cout<<"run GPU Scan ==================="<<endl;
	wscan.runGPUEu();
	//wscan.runGpu_Dtw_scBand();

	//runDevicDetector();



	return 0;
}
