/*
 * WrapperIndexBuilder.cpp
 *
 *  Created on: Jun 13, 2014
 *      Author: zhoujingbo
 */

#include "WrapperIndexBuilder.h"
#include "IndexBuilder/invListBuilder.h"
#include "IndexBuilder/DataProcess.h"
#include "BladeLoader.h"
#include "../CONSTANT.h"

#include <sstream>
using namespace std;


WrapperIndexBuilder::WrapperIndexBuilder() {
	// TODO Auto-generated constructor stub

}

WrapperIndexBuilder::~WrapperIndexBuilder() {
	// TODO Auto-generated destructor stub
}


int WrapperIndexBuilder:: runSingleIndexBuilder(string dataHolder,int fcol,int dimensionNumber, int queryNum, string winType,int bits_for_value){


	std::stringstream queryFileStream;
	queryFileStream << dataHolder<<"_d" << dimensionNumber
			<< "_q"<< queryNum<<"_dir.query";

	//queryFileStream<<"data/Dodgers/Dodgers_d"<<totalDimension<<"_q"<<queryNum<<"_dir.query";
	string queryFile = queryFileStream.str().c_str();

	std::stringstream idxPath;
	idxPath << dataHolder<<"_d" << dimensionNumber << "_" << winType
			<<"w.idx";
	//idxPath << "data/Dodgers/Dodgers_d" << totalDimension<< "_sw.idx";
	string invertedListPath = idxPath.str().c_str();
	string dataFile = dataHolder+".csv";

	invListBuilder ilB;


	ilB.runBuild_IdxAndQuery(dataFile,fcol, "i",invertedListPath,queryFile,queryNum,dimensionNumber,bits_for_value, winType);
	return 0;
}

int WrapperIndexBuilder::runSingleIndexBuilder(void) {

	//============this is for seting the index and query
	string dataHolder = "data/Dodgers/Dodgers";
	int fcol = 1;

	//string dataHolder = "data/calit2/CalIt2_7";
	//int fcol = 3;

	int dimensionNumber = DIMENSIONNUM;
	int queryNum = QUERYNUM;


	string winType = "s";
	int bits_for_value = 7;
	//===================== end for seting the index and query

	runSingleIndexBuilder(dataHolder, fcol, dimensionNumber,  queryNum,  winType, bits_for_value);


	return 0;
}


int WrapperIndexBuilder::runBuilderIndex(){
	runSingleIndexBuilder();
}

