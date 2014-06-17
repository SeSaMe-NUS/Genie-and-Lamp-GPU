/*
 * Util.h
 *
 *  Created on: Feb 22, 2014
 *      Author: zhoujingbo
 */

#ifndef UTIL_H_
#define UTIL_H_


#include <vector>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <assert.h>
using namespace std;




#ifndef NULL
#define NULL 0
#endif

#ifndef uint
typedef unsigned int uint;
#endif

#define OUTPUT
#define INPUT


template <class T>
void printVector(vector<T>& data);

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)





template <class T>
class UtlDTW {
public:
	UtlDTW();
	virtual ~UtlDTW();

	int* selectMinK(int k, T* data, int s, int e);
	void printIntArray(int* data,int len);

};



class topNode {
public:
	topNode() {
		dis = 0.0;
		idx = 0;
	}

	topNode(float dist, int id) :
		dis(dist), idx(id) {
	}
	float dis;
	int idx;

	bool operator <(const topNode &m) const {
		return dis < m.dis;
	}

	void print() {
		std::cout << "(" << idx << "," << dis << ")" <<std::endl;
	}
};


#endif /* UTIL_H_ */
