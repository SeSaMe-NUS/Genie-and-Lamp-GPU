#ifndef MAPMAP_H_
#define MAPMAP_H_

#include <map>
#include "../HDMapConfig.h"
#include "../HDMapException.h"
#include "HDMapAdaptor.h"

template<typename Key>
class HDMapMapMap : public HDMapAdaptor<Key>
{
  std::map<Key, int> h_map;
  bool frozen;
public:
  HDMapMapMap() : frozen(false)
  {}
  ~HDMapMapMap()
  {}

  void
  freeze();

  bool
  is_frozen();

  HDMapMapMap<Key>&
  map(Key key, int add);

  bool
  hit(Key key);

  int
  get(Key key);

  Key
  fuzzy_lower(Key key);

  Key
  fuzzy_upper(Key key);

  int
  size();

};

#include "HDMapMapMap.inl"

#endif
