#include "HDMapMapMap.h"

template<typename Key>
void
HDMapMapMap<Key>::freeze()
{
  frozen = true;
}

template<typename Key>
bool
HDMapMapMap<Key>::is_frozen()
{
  return frozen;
}

template<typename Key>
HDMapMapMap<Key>&
HDMapMapMap<Key>::map(Key key, int add)
{
  frozen ? throw HDMAP_OBJECT_IS_FROZEN : NULL;

  h_map[key] = add;
  return *this;
}

template<typename Key>
bool
HDMapMapMap<Key>::hit(Key key)
{
  if(h_map.find(key) == h_map.end())
    return false;
  else
    return true;
}

template<typename Key>
int
HDMapMapMap<Key>::get(Key key)
{
  hit(key) ? NULL : throw HDMAP_ITEM_NOT_FOUND;
  return h_map[key];
}

template<typename Key>
Key
HDMapMapMap<Key>::fuzzy_lower(Key key)
{
  typename std::map<Key, int>::iterator lower;
  lower = h_map.lower_bound(key);
  if(lower != h_map.begin())
    lower--;
  else
    lower = h_map.end();
  return lower->first;
}

template<typename Key>
Key
HDMapMapMap<Key>::fuzzy_upper(Key key)
{
  typename std::map<Key, int>::iterator upper;
  upper = h_map.upper_bound(key);
  return upper->first;
}

template<typename Key>
int
HDMapMapMap<Key>::size()
{
  return h_map.size();
}
