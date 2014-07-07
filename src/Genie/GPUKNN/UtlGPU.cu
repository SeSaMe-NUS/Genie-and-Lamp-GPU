#include <climits>
#include <limits>
#include "UtlGPU.h"


//__host__ __device__ QueryInfo::QueryInfo( )
//{
//	// apply a for the sift case
//	new (this)QueryInfo(128);
//}

GpuQuery::GpuQuery(int queryId, int nds){
	setDefaultPara(queryId, nds);
}


GpuQuery::GpuQuery( const GpuQuery& other )
{
	queryID = other.queryID;
	numOfDimensionToSearch = other.numOfDimensionToSearch;
	topK = other.topK;
	aggregateFunction = other.aggregateFunction;

	keywords.resize(numOfDimensionToSearch);
	upwardSearchBound.resize(numOfDimensionToSearch);
	downwardSearchBound.resize(numOfDimensionToSearch);
	upwardDistanceBound.resize(numOfDimensionToSearch);
	downwardDistanceBound.resize(numOfDimensionToSearch);
	dimensionSet.resize(numOfDimensionToSearch);

	for (int i = 0; i < numOfDimensionToSearch; i++)
	{
		keywords[i] = other.keywords[i];
		dimensionSet[i].dimension = other.dimensionSet[i].dimension;
		dimensionSet[i].distanceFunctionPerDimension = other.dimensionSet[i].distanceFunctionPerDimension;
		dimensionSet[i].weight =  other.dimensionSet[i].weight;

		upwardDistanceBound[i] = other.upwardDistanceBound[i];
		downwardDistanceBound[i] = other.downwardDistanceBound[i];

		upwardSearchBound[i] = other.upwardSearchBound[i];
		downwardSearchBound[i] = other.downwardSearchBound[i];
	}
}


GpuQuery& GpuQuery::operator =(GpuQuery other)
{
	numOfDimensionToSearch = other.numOfDimensionToSearch;
	topK = other.topK;
	aggregateFunction = other.aggregateFunction;

	keywords.resize(numOfDimensionToSearch);
	upwardSearchBound.resize(numOfDimensionToSearch);
	downwardSearchBound.resize(numOfDimensionToSearch);
	upwardDistanceBound.resize(numOfDimensionToSearch);
	downwardDistanceBound.resize(numOfDimensionToSearch);
	dimensionSet.resize(numOfDimensionToSearch);

	for (int i = 0; i < numOfDimensionToSearch; i++)
	{
		keywords[i] = other.keywords[i];
		dimensionSet[i].dimension = other.dimensionSet[i].dimension;
		dimensionSet[i].distanceFunctionPerDimension = other.dimensionSet[i].distanceFunctionPerDimension;

		upwardDistanceBound[i] = other.upwardDistanceBound[i];
		downwardDistanceBound[i] = other.downwardDistanceBound[i];

		upwardSearchBound[i] = other.upwardSearchBound[i];
		downwardSearchBound[i] = other.downwardSearchBound[i];
	}

	return *this;
}

void GpuQuery::setDefaultPara(int queryId, int nds){

	this->queryID = queryId;
	this->numOfDimensionToSearch = nds;

	topK = 5;
	aggregateFunction = 1;

	keywords.resize(nds, 0);
	upwardSearchBound.resize(nds, INT_MAX);
	downwardSearchBound.resize(nds, INT_MAX);
	upwardDistanceBound.resize(nds, 0);
	downwardDistanceBound.resize(nds, 0);
	dimensionSet.resize(nds);

	//default setting for one GPU query
	for (int i = 0; i < numOfDimensionToSearch; i++) {

		dimensionSet[i].dimension = i;
		dimensionSet[i].distanceFunctionPerDimension = 2; //L-p normal distance, the default distance is L2
		dimensionSet[i].weight = 1;
	}

}

__host__ __device__ QueryInfo::QueryInfo( int numOfDimensionToSearch )
{
	this->numOfDimensionToSearch = numOfDimensionToSearch;
	aggregateFunc = 1;
	topK = 10;

	// initialization
	initMemberArrays(numOfDimensionToSearch);


	for(int i = 0; i < numOfDimensionToSearch; i++)
	{
		keyword[i] = -1;	// init with invalid value
		lastPos[i].x = 0;	// init with invalid value
		lastPos[i].y = 0;	// init with invalid value

		searchDim[i] = -1;	// init with invalid value
		dimWeight[i] = 1;	// init with equal weight
		distanceFunc[i] = 2;// init with default distance function

		upperBoundDist[i] = 0;
		lowerBoundDist[i] = 0;

		// init with maximal possible value, they are all reletive position according to query keyword
		upperBoundSearch[i] = INT_MAX;
		lowerBoundSearch[i] = INT_MAX;
	}
}


__host__ __device__ QueryInfo::QueryInfo( const QueryInfo& other )
{
	numOfDimensionToSearch = other.numOfDimensionToSearch;

	this->topK = other.topK;

	this->aggregateFunc = other.aggregateFunc;

	// initialization
	initMemberArrays(numOfDimensionToSearch);

	// copy keywords
	for (int i = 0; i < numOfDimensionToSearch; i++)
	{
		// copy keywords
		this->keyword[i] = other.keyword[i];

		lastPos[i].x = other.lastPos[i].x;
		lastPos[i].y = other.lastPos[i].y;

		this->searchDim[i] = other.searchDim[i];
		this->dimWeight[i] = other.dimWeight[i];
		this->distanceFunc[i] =	other.distanceFunc[i];


		this->upperBoundDist[i] = other.upperBoundDist[i];
		this->lowerBoundDist[i] = other.lowerBoundDist[i];


		// NEED TO ADD LATER: search lower bound and upper bound of different dimension.
		this->upperBoundSearch[i] = other.upperBoundSearch[i];
		this->lowerBoundSearch[i] = other.lowerBoundSearch[i];
	}

}

__host__ __device__ void QueryInfo::reset(const QueryInfo& other){
		numOfDimensionToSearch = other.numOfDimensionToSearch;

		this->topK = other.topK;

		this->aggregateFunc = other.aggregateFunc;


		// copy keywords
		for (int i = 0; i < numOfDimensionToSearch; i++)
		{
			// copy keywords
			this->keyword[i] = other.keyword[i];

			lastPos[i].x = other.lastPos[i].x;
			lastPos[i].y = other.lastPos[i].y;

			this->searchDim[i] = other.searchDim[i];
			this->dimWeight[i] = other.dimWeight[i];
			this->distanceFunc[i] =	other.distanceFunc[i];


			this->upperBoundDist[i] = other.upperBoundDist[i];
			this->lowerBoundDist[i] = other.lowerBoundDist[i];


			// NEED TO ADD LATER: search lower bound and upper bound of different dimension.
			this->upperBoundSearch[i] = other.upperBoundSearch[i];
			this->lowerBoundSearch[i] = other.lowerBoundSearch[i];
		}
}


__host__ __device__ QueryInfo& QueryInfo::operator= (QueryInfo other)
{
	if ( numOfDimensionToSearch != other.numOfDimensionToSearch )
	{
		delete[] keyword;
		delete[] lastPos;

		delete[] searchDim;
		delete[] dimWeight;
		delete[] distanceFunc;

		delete[] upperBoundDist;
		delete[] lowerBoundDist;

		delete[] upperBoundSearch;
		delete[] lowerBoundSearch;

		// initialization
		initMemberArrays(numOfDimensionToSearch);
	}

	numOfDimensionToSearch = other.numOfDimensionToSearch;
	this->topK = other.topK;
	this->aggregateFunc = other.aggregateFunc;

	// copy keywords
	for (int i = 0; i < numOfDimensionToSearch; i++)
	{
		// copy keywords
		this->keyword[i] = other.keyword[i];

		lastPos[i].x = other.lastPos[i].x;
		lastPos[i].y = other.lastPos[i].y;

		this->searchDim[i] = other.searchDim[i];
		this->dimWeight[i] = other.dimWeight[i];
		this->distanceFunc[i] =	other.distanceFunc[i];


		this->upperBoundDist[i] = other.upperBoundDist[i];
		this->lowerBoundDist[i] = other.lowerBoundDist[i];


		// NEED TO ADD LATER: search lower bound and upper bound of different dimension.
		this->upperBoundSearch[i] = other.upperBoundSearch[i];
		this->lowerBoundSearch[i] = other.lowerBoundSearch[i];
	}

	return *this;
}


__host__ QueryInfo::QueryInfo( const GpuQuery& query )
{
	this->numOfDimensionToSearch = query.getNumOfDimensionToSearch();
	this->topK = query.topK;
	this->aggregateFunc = query.aggregateFunction;

	// initialization
	initMemberArrays(numOfDimensionToSearch);

	// copy keywords
	for (int i = 0; i < numOfDimensionToSearch; i++)
	{
		// copy keywords
		this->keyword[i] = query.keywords[i];

		lastPos[i].x = 0;	// init with invalid value
		lastPos[i].y = 0;	// init with invalid value

		this->searchDim[i] = query.dimensionSet[i].dimension;
		this->dimWeight[i] = query.dimensionSet[i].weight;
		this->distanceFunc[i] =	query.dimensionSet[i].distanceFunctionPerDimension;


		this->upperBoundDist[i] = query.upwardDistanceBound[i];
		this->lowerBoundDist[i] = query.downwardDistanceBound[i];


		// NEED TO ADD LATER: search lower bound and upper bound of different dimension.
		this->upperBoundSearch[i] = query.upwardSearchBound[i];
		this->lowerBoundSearch[i] = query.downwardSearchBound[i];
	}
}



/**
 * delete all the dynamic allocated arrays
 */
__host__ __device__ QueryInfo::~QueryInfo()
{
	delete[] keyword;
	delete[] lastPos;
	delete[] searchDim;
	delete[] dimWeight;
	delete[] distanceFunc;
	delete[] upperBoundDist;
	delete[] lowerBoundDist;
	delete[] upperBoundSearch;
	delete[] lowerBoundSearch;
}

__host__ __device__ void QueryInfo::initMemberArrays(int nds) {

	keyword = new float[nds];
	lastPos = new int2[nds];

	searchDim = new int[nds];
	dimWeight = new float[nds];
	distanceFunc = new float[nds];

	upperBoundDist = new float[nds];
	lowerBoundDist = new float[nds];

	upperBoundSearch = new int[nds];
	lowerBoundSearch = new int[nds];

	keyword_indexMapping = new int[nds];

}

/**
 * TODO: need to be further checked
 */
__host__ __device__ bool QueryInfo::checkInitialization()
{
	for(int i = 0; i < numOfDimensionToSearch; i++)
	{
		if ( keyword[i] == -1 || searchDim[i] == -1 ||
				upperBoundSearch[i] == INT_MAX || upperBoundSearch[i] <=0||
				lowerBoundSearch[i] == INT_MAX || lowerBoundSearch[i] <=0||
				dimWeight[i]<0||
				upperBoundDist[i]<0||lowerBoundDist[i]<0) {

			printf ("%s \n", "QueryInfo initialization error: The query is wrongly initialized, please check QueryInfo");
			return false;
		}
		if(lowerBoundDist[i]>upperBoundDist[i]){
			printf ("%s \n", "QueryInfo initialization error: Please make sure the min distance function bound is lower and max distance function is higher");
			return false;
		}


	}

	return true;
}


/**
 * int topK; 	// number of results to be returned
	int numOfDimensionToSearch; // number of dimension to search
	int aggregateFunc;			// aggregation function

	float *keyword;		// keyword (value of each dimension) for this query
	int2 *lastPos;		// lastPos[i].x means the down moved step, lastPos[i].y means the up moved step, all elements are valid
						// before total dimension in case the query change the search dimension in mid of search

	// all the array below, the entries are only valid up to index smaller than numOfDimensionToSearch
	int *searchDim;		// index represent the dimension to search
	float *dimWeight;	// weight of this perticular dimension
	int *distanceFunc;	// distance function for specific dimension ???

	float *upperBoundDist;		// upper bounding function to compute distance function
	float *lowerBoundDist;		// lower bounding function to compute distance function

	int *upperBoundSearch;	// search bound when going up
	int *lowerBoundSearch;	// search boudn when going down
 */


__host__ __device__ void QueryInfo::print()
{

	printf("Begin QueryInfo::===================================================== \n");
	printf("numOfDimensionToSearch: %d \n", numOfDimensionToSearch);
	printf("topK: %d \n", topK);
	printf("aggregateFunc: %d \n", aggregateFunc);

	//	float *keyword;		// keyword (value of each dimension) for this query
	//int2 *lastPos;		// lastPos[i].x means the down moved step, lastPos[i].y means the up moved step, all elements are valid
						// before total dimension in case the query change the search dimension in mid of search
	printf("print keyword+++++++++++++++++++++++++++++++++++++\n");
	for(int i=0;i<numOfDimensionToSearch;i++){
		printf(" keyword[%d] = %9.1f",i,keyword[i]);
	}
	printf("\n end print keyword++++++++++++++++++++++++++++\n");

	printf("print upperBoundDist+++++++++\n");
	for(int i=0;i<this->numOfDimensionToSearch;i++){
		printf(" upperBoundDist[%d] =%f ", i, this->upperBoundDist[i]);
	}
	printf("\n end print upperBoundDist+++++++++");

	printf("print lowerBoundDist+++++++++\n");
	for(int i=0;i<this->numOfDimensionToSearch;i++){
		printf(" lowerBoundDist[%d]=%f ",i,this->lowerBoundDist[i]);
	}


	printf("print lastPos+++++++++++++++++++++++++++++++++++\n");
	for(int i=0;i<numOfDimensionToSearch;i++){
		printf(" lastPos[%d] x =%d y =%d", i, lastPos[i].x,lastPos[i].y);
	}
	printf("\n end print lastPos++++++++++++++++++++++++++++\n");



	printf("print *searchDim++++++\n");
	for (int i = 0; i < numOfDimensionToSearch; i++) {
		printf("searchDim[%d]=%d   ", i, searchDim[i]);
	}
	printf("\n");

	printf("print *dimWeight++++++\n");
	for (int i = 0; i < numOfDimensionToSearch; i++) {
		printf("dimWeight[%d]=%f   ", i, dimWeight[i]);
	}
	printf("\n");

	printf("print *upperBoundSearch++++++\n");
	for (int i = 0; i < numOfDimensionToSearch; i++) {
		printf("upperBoundSearch[%d]=%d   ", i, upperBoundSearch[i]);
	}
	printf("\n");

	printf("print *lowerBoundSearch++++++\n");
	for (int i = 0; i < numOfDimensionToSearch; i++) {
		printf("lowerBoundSearch[%d]=%d   ", i, lowerBoundSearch[i]);
	}
	printf("\n");

	printf(
			"\n end QueryInfo::=======================================================================\n");

//	// initialization
//	keyword = new float[numOfDimensionToSearch];
//	lastPos = new int2[numOfDimensionToSearch];
//
//	searchDim = new int[numOfDimensionToSearch];
//	dimWeight = new float[numOfDimensionToSearch];
//	distanceFunc = new int[numOfDimensionToSearch];
//
//	upperBoundDist = new float[numOfDimensionToSearch];
//	lowerBoundDist = new float[numOfDimensionToSearch];
//
//	upperBoundSearch = new int[numOfDimensionToSearch];
//	lowerBoundSearch = new int[numOfDimensionToSearch];
//
//	// copy keywords
//	for (int i = 0; i < numOfDimensionToSearch; i++)
//	{
//		// copy keywords
//		this->keyword[i] = query.keywords[i];
//
//		lastPos[i].x = 0;	// init with invalid value
//		lastPos[i].y = 0;	// init with invalid value
//
//		this->searchDim[i] = query.dimensionSet[i].dimension;
//		this->dimWeight[i] = query.dimensionSet[i].weight;
//		this->distanceFunc[i] =	query.dimensionSet[i].distanceFunctionPerDimension;
//
//
//		this->upperBoundDist[i] = query.upwardDistanceBound[i];
//		this->lowerBoundDist[i] = query.downwardDistanceBound[i];
//
//
//		// NEED TO ADD LATER: search lower bound and upper bound of different dimension.
//		this->upperBoundSearch[i] = query.upwardSearchBound[i];
//		this->lowerBoundSearch[i] = query.downwardSearchBound[i];
//	}


}


InvertListSpecGPU::InvertListSpecGPU()
{
	totalDimension = 0;

	indexDimensionEntry = NULL;

	numOfQuery = 0;
	maxFeatureID = 0;
	numOfDocToExpand= 0;
}


InvertListSpecGPU::InvertListSpecGPU(int numberOfDimensions)
{
	totalDimension = numberOfDimensions;

	indexDimensionEntry = new GpuIndexDimensionEntry[totalDimension];

	numOfQuery = 0;
	maxFeatureID = 0;
	numOfDocToExpand= 0;
}


void InvertListSpecGPU::init_InvertListSpecGPU(int numberOfDimensions)
{
	totalDimension = numberOfDimensions;

	if(indexDimensionEntry!=NULL) delete []indexDimensionEntry;
	indexDimensionEntry = new GpuIndexDimensionEntry[totalDimension];

	numOfQuery = 0;
	maxFeatureID = 0;
	numOfDocToExpand= 0;
}



InvertListSpecGPU::~InvertListSpecGPU()
{
	delete[] indexDimensionEntry;
}

