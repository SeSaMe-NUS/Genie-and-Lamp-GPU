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

using namespace std;


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {

	WrapperIndexBuilder wrapperIndexBuilder;
	wrapperIndexBuilder.runBuilderIndex();

	WrapperGPUKNN wrapperGpuKnn;
	wrapperGpuKnn.runGPUKNN();

	WrapperScan wscan;
	wscan.runCPUEu();

	//runDevicDetector();



	return 0;
}
