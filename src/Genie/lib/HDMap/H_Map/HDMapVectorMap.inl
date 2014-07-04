#include <stdio.h>
#include <thrust/host_vector.h>
#include "HDMapVectorMap.h"

template<typename Key>
void
HDMapVectorMap<Key>::freeze()
{
  frozen = true;
}

template<typename Key>
bool
HDMapVectorMap<Key>::is_frozen()
{
  return frozen;
}

template<typename Key>
HDMapVectorMap<Key>&
HDMapVectorMap<Key>::map(Key key, int add)
{
  frozen ? throw HDMAP_OBJECT_IS_FROZEN : NULL;

  if(key >= h_map.size())
    {
      h_map.resize(key+1);
      thrust::fill(h_map.begin()+last+1, h_map.end(), VECTORMAP_H_EMPTY);
    }
  h_map[key] = add;
  last = last > key ? last : key;
  return *this;
}

template<typename Key>
bool
HDMapVectorMap<Key>::hit(Key key)
{
  return key < h_map.size() && h_map[key] >= 0 ? true : false;
}

template<typename Key>
int
HDMapVectorMap<Key>::get(Key key)
{
  hit(key) ? NULL : throw HDMAP_ITEM_NOT_FOUND;
  return h_map[key];
}

template<typename Key>
int
HDMapVectorMap<Key>::size()
{
  return h_map.size();
}

template<typename Key>
float
HDMapVectorMap<Key>::ratio()
{
  int count = 0;
  for(int i=0; i<h_map.size(); i++)
    if(h_map[i] >= 0)
      count++;
  return 1.0*count/size();
}
