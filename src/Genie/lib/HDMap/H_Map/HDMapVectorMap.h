#ifndef VECTORMAP_H_
#define VECTORMAP_H_

#include <thrust/fill.h>
#include <vector>
#include "../HDMapConfig.h"
#include "../HDMapException.h"
#include "HDMapAdaptor.h"

#define VECTORMAP_H_EMPTY -1

template<typename Key=int>
class HDMapVectorMap : public HDMapAdaptor<Key>
{
  std::vector<int> h_map;
  bool frozen;
  int last;
public:
  HDMapVectorMap(int size)
    : h_map((int)(size*HDMAP_SIZE_RATIO)), frozen(false), last(0)
  {
    thrust::fill(h_map.begin(), h_map.end(), VECTORMAP_H_EMPTY);
  }

  void
  freeze();

  bool
  is_frozen();

  HDMapVectorMap&
  map(Key key, int add);

  bool
  hit(Key key);

  int
  get(Key key);

  int
  size();

  float
  ratio();

  ~HDMapVectorMap(){}
};

#include "HDMapVectorMap.inl"

#endif
