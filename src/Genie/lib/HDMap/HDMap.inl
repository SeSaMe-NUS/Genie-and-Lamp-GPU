#include <stdio.h>
#include "HDMap.h"

template<typename K, typename V>
HDMap<K, V>::HDMap()
{
  h_map = new HDMapMapMap<K>();
  d_value = new HDValueVectorValue<V>(10);
  frozen = false;
}

template<typename K, typename V>
HDMap<K, V>::HDMap(int size)
{
  h_map = new HDMapMapMap<K>();
  d_value = new HDValueVectorValue<V>(size);
  frozen = false;
}

template<typename K, typename V>
HDMap<K, V>::~HDMap()
{
  delete(h_map);
  delete(d_value);
}

template<typename K, typename V>
void
HDMap<K, V>::show_info(char* tag)
{
  fprintf(stdout, "[HDMapArray Tag: %s] CPU Map Size: %d Ratio: %.2f%% Frozen? %d; GPU Value Size: %d Ratio %.2f%% Frozen? %d\n",
      tag, h_map->size(), h_map->ratio()*100, h_map->is_frozen(), d_value->size(), d_value->ratio()*100, d_value->is_frozen());
}

template<typename K, typename V>
void
HDMap<K, V>::freeze()
{
  h_map->freeze();
  d_value->freeze();
  frozen = true;
}

template<typename K, typename V>
HDMap<K, V>&
HDMap<K, V>::map(K key, V value)
{
  frozen ? throw HDMAP_OBJECT_IS_FROZEN : NULL;
  h_map->hit(key) ? throw HDMAP_KEY_EXISTS : NULL;

  h_map->map(key, d_value->push(value));
  return *this;
}

template<typename K, typename V>
V
HDMap<K, V>::get(K key)
{
  return d_value->get(get_index(key));
}

template<typename K, typename V>
int
HDMap<K, V>::get_index(K key)
{
  h_map->hit(key) ? NULL : throw HDMAP_ITEM_NOT_FOUND;
  return h_map->get(key);
}

template<typename K, typename V>
K
HDMap<K, V>::get_fuzzy_lower_key(K key)
{
  return h_map->fuzzy_lower(key);
}

template<typename K, typename V>
K
HDMap<K, V>::get_fuzzy_upper_key(K key)
{
  return h_map->fuzzy_upper(key);
}

template<typename K, typename V>
thrust::device_vector<V>&
HDMap<K, V>::raw_value()
{
  return d_value->raw_value();
}
