#ifndef HDMapArray_H_
#define HDMapArray_H_

#include <thrust/device_vector.h>
#include "H_Map/H_Map.h"
#include "D_Value/D_Value.h"
#include "HDMapConfig.h"
#include "HDMapException.h"

template<typename K, typename V>
class HDMap {

  HDMapAdaptor<K>* h_map;
  HDValueAdaptor<V>* d_value;
  bool frozen;

public:
  HDMap();
  HDMap(int size);
  ~HDMap();

  void
  show_info(char *tag);

  void
  freeze();

  bool
  is_frozen();

  HDMap<K, V>&
  map(K key, V value);

  V
  get(K key);

  int
  get_index(K key);

  K
  get_fuzzy_lower_key(K key);

  K
  get_fuzzy_upper_key(K key);

  thrust::device_vector<V>&
  raw_value();
};

#include "HDMap.inl"
#endif
