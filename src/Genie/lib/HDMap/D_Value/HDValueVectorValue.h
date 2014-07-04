#ifndef VECTOR_VALUE_H_
#define VECTOR_VALUE_H_

#include <thrust/device_vector.h>
#include <vector>
#include "../HDMapConfig.h"
#include "../HDMapException.h"
#include "HDValueAdaptor.h"

template<typename Value>
class HDValueVectorValue : public HDValueAdaptor<Value>
{
  thrust::device_vector<Value> d_value;
  std::vector<Value> h_value;
  bool frozen;
  int position;
public:
  HDValueVectorValue(int size)
    : h_value(size), frozen(false), position(0)
  {}

  void
  freeze();

  bool
  is_frozen();

  int
  push(Value value);

  bool
  hit(int index);

  Value
  get(int index);

  thrust::device_vector<Value>&
  raw_value();

  int
  size();

  float
  ratio();

  ~HDValueVectorValue(){}
};

#include "HDValueVectorValue.inl"
#endif
