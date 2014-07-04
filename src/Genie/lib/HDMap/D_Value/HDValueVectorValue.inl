#include <stdio.h>
#include <thrust/copy.h>
#include "HDValueVectorValue.h"

template<typename Value>
void
HDValueVectorValue<Value>::freeze()
{
  d_value.resize(position);
  thrust::copy(h_value.begin(), h_value.begin()+position, d_value.begin());
  frozen = true;
}

template<typename Value>
bool
HDValueVectorValue<Value>::is_frozen()
{
  return frozen;
}

template<typename Value>
int
HDValueVectorValue<Value>::push(Value value)
{
  frozen ? throw HDMAP_OBJECT_IS_FROZEN : NULL;

  int index = position;
  if(position >= h_value.size())
    {
      int size = h_value.size();
      h_value.resize((int)(HDMAP_SIZE_RATIO*size));
    }
  h_value[position] = (value);
  position++;
  return index;
}

template<typename Value>
bool
HDValueVectorValue<Value>::hit(int index)
{
  return index < position ? true : false;
}

template<typename Value>
Value
HDValueVectorValue<Value>::get(int index)
{
  hit(index) ? NULL : throw HDMAP_ITEM_NOT_FOUND;
  return frozen ? d_value[index] : h_value[index];
}

template<typename Value>
thrust::device_vector<Value>&
HDValueVectorValue<Value>::raw_value()
{
  frozen ? NULL : throw HDMAP_OBJECT_IS_NOT_FROZEN;
  return d_value;
}

template<typename Value>
int
HDValueVectorValue<Value>::size()
{
  return frozen ? d_value.size() : h_value.size();
}

template<typename Value>
float
HDValueVectorValue<Value>::ratio()
{
  return 1.0*position/size();
}


