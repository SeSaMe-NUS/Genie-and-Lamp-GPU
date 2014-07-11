#include <stdio.h>
#include "HDMap.h"

template<typename K, typename V>
HDMap<K, V>::HDMap()
{
  h_map = new HDMapMapMap<K>();
  d_value = new HDValueVectorValue<V>(10);
  frozen = false;
  timer = 0;
}

template<typename K, typename V>
HDMap<K, V>::HDMap(int size)
{
  h_map = new HDMapMapMap<K>();
  d_value = new HDValueVectorValue<V>(size);
  frozen = false;
  timer = 0;
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
  fprintf(stdout, "[HDMapArray Tag: %s] Timer: %f; CPU Map Size: %d Ratio: %.2f%% Frozen? %d; GPU Value Size: %d Ratio %.2f%% Frozen? %d\n",
      tag, timer, h_map->size(), h_map->ratio()*100, h_map->is_frozen(), d_value->size(), d_value->ratio()*100, d_value->is_frozen());
}

template<typename K, typename V>
void
HDMap<K, V>::freeze()
{
  clock_t start = clock();
  h_map->freeze();
  d_value->freeze();
  frozen = true;
  timer += (double) (clock() - start) / CLOCKS_PER_SEC;
}

template<typename K, typename V>
HDMap<K, V>&
HDMap<K, V>::map(K key, V value)
{
  clock_t start = clock();
  frozen ? throw HDMAP_OBJECT_IS_FROZEN : NULL;
  h_map->hit(key) ? throw HDMAP_KEY_EXISTS : NULL;

  h_map->map(key, d_value->push(value));
  timer += (double) (clock() - start) / CLOCKS_PER_SEC;
  return *this;
}

template<typename K, typename V>
V
HDMap<K, V>::get(K key)
{
  clock_t start = clock();
  V value = d_value->get(get_index(key));
  timer += (double) (clock() - start) / CLOCKS_PER_SEC;
  return value;
}

template<typename K, typename V>
int
HDMap<K, V>::get_index(K key)
{
  clock_t start = clock();
  h_map->hit(key) ? NULL : throw HDMAP_ITEM_NOT_FOUND;
  int index = h_map->get(key);
  timer += (double) (clock() - start) / CLOCKS_PER_SEC;
  return index;
}

template<typename K, typename V>
K
HDMap<K, V>::get_fuzzy_lower_key(K key)
{
  clock_t start = clock();
  K k = h_map->fuzzy_lower(key);
  timer += (double) (clock() - start) / CLOCKS_PER_SEC;
  return k;
}

template<typename K, typename V>
K
HDMap<K, V>::get_fuzzy_upper_key(K key)
{
  clock_t start = clock();
  K k = h_map->fuzzy_upper(key);
  timer += (double) (clock() - start) / CLOCKS_PER_SEC;
  return k;
}

template<typename K, typename V>
thrust::device_vector<V>&
HDMap<K, V>::raw_value()
{
  clock_t start = clock();
  thrust::device_vector<V>& dv = d_value->raw_value();
  timer += (double) (clock() - start) / CLOCKS_PER_SEC;
  return dv;
}
