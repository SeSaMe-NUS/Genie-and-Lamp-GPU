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
#include <iostream>


#include "AppManager/WrapperIndexBuilder.h"
#include "AppManager/deviceDetector/deviceDetector.h"
#include "Genie/WrapperGPUKNN.h"
#include "Genie/WrapperScan.h"
#include <vector>
using namespace std;

#include <thrust/device_vector.h>
using namespace thrust;

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {



	WrapperIndexBuilder wrapperIndexBuilder;
	wrapperIndexBuilder.runBuilderIndex();

	WrapperGPUKNN wrapperGpuKnn;
	cout<<endl;
	cout<<"run GPU inverted index ==================="<<endl;
	wrapperGpuKnn.runGPUKNN();



	WrapperScan wscan;
	cout<<endl;
	cout<<"run CPU Scan ==================="<<endl;
	wscan.runCPUEu();

	cout<<endl;
	cout<<"run GPU Scan ==================="<<endl;
	wscan.runGPUEu();



	//runDevicDetector();



	return 0;
}
