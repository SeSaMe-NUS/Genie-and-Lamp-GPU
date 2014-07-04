#ifndef HDMAPADAPTOR_H_
#define HDMAPADAPTOR_H_

template<typename Key>
class HDMapAdaptor
{
public:
  virtual
  void
  freeze()=0;

  virtual
  bool
  is_frozen()=0;

  virtual
  HDMapAdaptor<Key>&
  map(Key key, int add)=0;

  virtual
  bool
  hit(Key key)=0;

  virtual
  int
  get(Key key)=0;

  virtual
  Key
  fuzzy_lower(Key key)=0;

  virtual
  Key
  fuzzy_upper(Key key)=0;

  virtual
  int
  size(){ return -1; };

  virtual
  float
  ratio(){ return 0; }

  virtual
  ~HDMapAdaptor(){};
};

#endif
