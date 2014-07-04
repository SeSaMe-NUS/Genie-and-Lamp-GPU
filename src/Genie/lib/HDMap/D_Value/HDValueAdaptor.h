#ifndef DVALUEADAPTOR_H_
#define DVALUEADAPTOR_H_

#include <thrust/device_vector.h>
#include "../HDMapException.h"

template<typename Value>
class HDValueAdaptor
{
public:
  virtual
  void
  freeze()=0;

  virtual
  bool
  is_frozen()=0;

  virtual
  int
  push(Value value)=0;

  virtual
  bool
  hit(int index)=0;

  virtual
  Value
  get(int index)=0;

  virtual
  thrust::device_vector<Value>&
  raw_value(){ throw HDMAP_METHOD_NOT_IMPLEMENTED; };

  virtual
  int
  size(){ return -1; }

  virtual
  float
  ratio(){ return 0; }

  virtual
  ~HDValueAdaptor(){};
};


#endif
