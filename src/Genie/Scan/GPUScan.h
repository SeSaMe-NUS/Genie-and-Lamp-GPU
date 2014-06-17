/*
 * GPUScan.h
 *
 *  Created on: Jun 16, 2014
 *      Author: zhoujingbo
*/

#ifndef GPUSCAN_H_
#define GPUSCAN_H_


#include <vector>
using namespace std;


class GPUScan {
public:
	GPUScan();
	virtual ~GPUScan();

	 void computTopk_int_eu(vector<vector<int> >& query, int k,
			vector<int> & data);
	 void computTopk_int_dtw_scBand(vector<vector<int> >& query, int k,
			vector<int> & data, int sc_band);


};

#endif  /* GPUSCAN_H_*/

