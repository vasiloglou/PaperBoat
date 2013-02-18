/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
Inc, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE Ismion Inc "AS IS" AND ANY
EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COMPANY BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef FL_LITE_FASTLLIB_DATA_MIXED_POINT_MPL_DEFS_H_
#define FL_LITE_FASTLLIB_DATA_MIXED_POINT_MPL_DEFS_H_
/**
 * @brief Exporting the types
 *
 */
public:
typedef typename ParamList::DenseTypes DenseTypes_t;
typedef typename ParamList::SparseTypes SparseTypes_t;
typedef typename ParamList::CalcPrecisionType CalcPrecision_t;
typedef typename ParamList::MetaDataType MetaData_t;

typedef typename boost::mpl::not_ <
boost::is_same <
typename ParamList::DenseTypes,
typename MixedPointArgs::DenseTypes
>
> HasDense;
typedef typename boost::mpl::not_ <
boost::is_same <
typename ParamList::SparseTypes,
typename MixedPointArgs::SparseTypes
>
> HasSparse;
typedef typename boost::mpl::not_ <
boost::is_same <
typename ParamList::MetaDataType,
typename MixedPointArgs::MetaDataType
>
> HasMetaData;


/**
 * @brief class composition
 *
 */
typedef DenseTypes_t DenseMemberTypes_t;
typedef SparseTypes_t SparseMemberTypes_t;
/**
 * @brief the class that contains the the dense points
 */
template<typename T>
struct dense_wrap : public wrap<T, MonolithicPoint> {};
typedef typename boost::mpl::inherit_linearly < DenseMemberTypes_t,
boost::mpl::inherit < dense_wrap<boost::mpl::placeholders::_2>,
boost::mpl::placeholders::_1 > >::type DensePointCollection_t;

/**
 * @brief the class that contains the sparse points
 */
template<typename T>
struct sparse_wrap : public wrap<T, SparsePoint> {};
typedef typename boost::mpl::inherit_linearly < SparseMemberTypes_t,
boost::mpl::inherit < sparse_wrap<boost::mpl::placeholders::_2>,
boost::mpl::placeholders::_1 > >::type SparsePointCollection_t;

/**
 * @brief the class that contains the iterators
 */
template<typename T>
struct dense_iterator_wrap {
  public:
    typedef typename MonolithicPoint<T>::iterator type;
};
typedef typename boost::mpl::inherit_linearly < DenseMemberTypes_t,
boost::mpl::inherit < dense_iterator_wrap<boost::mpl::placeholders::_2>,
boost::mpl::placeholders::_1 > >::type DenseIteratorCollection_t;

template<typename T>
struct sparse_iterator_wrap {
  public:
    typedef typename SparsePoint<T>::iterator type;
};
typedef typename boost::mpl::inherit_linearly < SparseMemberTypes_t,
boost::mpl::inherit < sparse_iterator_wrap<boost::mpl::placeholders::_2>,
boost::mpl::placeholders::_1 > >::type SparseIteratorCollection_t;


/**
 * @brief
 */
template<typename T>
struct this_wrap {
  T value;
};
typedef typename boost::mpl::vector<> Vector0;
typedef typename boost::mpl::if_ < HasMetaData,
typename boost::mpl::push_back<Vector0, MetaData_t* >::type,
Vector0 >::type Vector1;
typedef typename boost::mpl::if_ < HasDense,
typename boost::mpl::push_back < Vector1,
DensePointCollection_t >::type, Vector1 >::type Vector2;
typedef typename boost::mpl::if_ < HasSparse,
typename boost::mpl::push_back < Vector2,
SparsePointCollection_t >::type, Vector2 >::type Vector3;
typedef typename boost::mpl::inherit_linearly < Vector3,
boost::mpl::inherit < this_wrap<boost::mpl::placeholders::_2>,
boost::mpl::placeholders::_1 > >::type PointCollection_t;

struct MetaDataSelector1 {
  static inline MetaData_t &get(MixedPoint *p) {
    return *static_cast<this_wrap<MetaData_t*> &>(p->collection_).value;
  }

  static inline const MetaData_t &get(const MixedPoint *p) {
    return *static_cast<const this_wrap<MetaData_t*> &>(p->collection_).value;
  }

  static inline void set(MixedPoint *p, MetaData_t value) {
    *static_cast<this_wrap<MetaData_t* > &>(p->collection_).value
    = value;
  }

};

struct MetaDataSelector2 {
  static inline signed char get(MixedPoint *p) {
    return 0;
  }
  static inline const signed char get(const MixedPoint *p) {
    return 0;
  }
  static void set(MixedPoint *p, int64 value) {
  }
};

struct MetaSwapSelector1 {
  template<typename MixedPointType>
  static inline void Swap(MixedPointType *p1, MixedPointType *p2) {
    MetaData_t temp = p1->meta_data();
    p1->set_meta_data(p2->meta_data());
    p2->set_meta_data(temp);
  }
};

struct MetaSwapSelector2 {
  template<typename MixedPointType>
  static inline void Swap(MixedPointType *p1, MixedPointType *p2) {
  }
};

struct MetaDataDefaultInitSelector1 {
  struct type {
    static inline void Init(MixedPoint * const p) {
      static_cast<this_wrap<MetaData_t* > &>(p->collection_).value = NULL;
    }
  };
};

struct MetaDataDefaultInitSelector2 {
  struct type {
    static inline void Init(MixedPoint * const p) {
    }
  };
};



struct MetaDataInitSelector1 {
  struct type {
    static inline void Init(MixedPoint * const p) {
      try {
        static_cast<this_wrap<MetaData_t* > &>(p->collection_).value = new MetaData_t();
      }
      catch(const std::bad_alloc &e) {
        fl::logger->Die() << "There was a problem allocating memory. "
          << "Either your dataset doesn't fit in the RAM, or"
          << "you are using a 32bit platform that limits the process "
          << "address space to 4GB";
      }
    }
  };
};

struct MetaDataInitSelector2 {
  struct type {
    static inline void Init(MixedPoint * const p) {
    }
  };
};

template<typename StreamType>
struct PrintPointDense {
public:
  PrintPointDense(const MixedPoint* point, StreamType &stream, std::string &delim) :
      point_(point), stream_(stream), delim_(delim) {
  }
  template<typename T>
  void operator()(T) {
    point_->dense_point<T>().Print(stream_, delim_);
  }
private:
  const MixedPoint* point_;
  StreamType &stream_;
  const std::string delim_;
};

template<typename StreamType>
struct PrintPointSparse {
public:
  PrintPointSparse(const MixedPoint* point, StreamType &stream, std::string &delim) :
      point_(point), stream_(stream), delim_(delim) {
  }
  template<typename T>
  void operator()(T) {
    point_->sparse_point<T>().Print(stream_, delim_);
  }
private:
  const MixedPoint* point_;
  StreamType &stream_;
  const std::string delim_;
};

struct PrintMetaDataSelector1 {
  struct Integral {
    template<typename StreamType>
    static inline void Print(const MixedPoint* p, StreamType &stream, std::string &delim)  {
      stream << p->meta_data() << delim;
    }
  };

  struct NonIntegral {
    template<typename StreamType>
    static inline void Print(const MixedPoint* p, StreamType &stream, std::string &delim) {
      //p->meta_data().Print(stream, delim);
    }
  };

  template<typename StreamType>
  static inline void Print(const MixedPoint* p, StreamType &stream, std::string &delim) {
    boost::mpl::if_ <
    boost::is_integral<MetaData_t>,
    Integral,
    NonIntegral >::type::Print(p, stream, delim);
  }
};

struct PrintMetaDataSelector2 {
  template<typename StreamType>
  static inline void Print(const MixedPoint* p, StreamType &stream, std::string &delim) {
  }
};



struct MetaDataPointerSelector1 {
  struct type {
    static inline MetaData_t* &Do(MixedPoint * const p1) {
      return static_cast<this_wrap<MetaData_t *>  &>(
               p1->collection_).value;
    }
  };
};

struct MetaDataPointerSelector2 {
  struct type {
    static inline void* Do(MixedPoint * const p1) {
      return NULL;
    }
  };
};

struct DestructSelector1 {
  struct type {
    static inline void Destruct(MixedPoint * const p1) {
      delete static_cast<this_wrap<MetaData_t* > >(
        p1->collection_).value;
    }
  };
};

struct DestructSelector2 {
  struct type {
    static inline void Destruct(MixedPoint * const p1) {
    }
  };
};


/**
 *  @brief It is possible that the mixed point contains only one type
 *  in this case we want to avoid the for_each overhead, this is why
 *  we developped these nullary metafunctions
 */
struct NullaryMetaFunctionOptimizedAccessDense {
  struct type {
    inline static void Get(const MixedPoint *p, index_t i, CalcPrecision_t * const result)  {
      *result = p->dense_point <
                typename boost::mpl::front<DenseTypes_t>::type > ().get(i);
    }
    inline static void Set(MixedPoint *p, index_t i, CalcPrecision_t * const result)  {
      p->dense_point <
                typename boost::mpl::front<DenseTypes_t>::type > ().set(i, *result);
    }

  };
};

struct NullaryMetaFunctionNonOptimizedAccessDense {
  struct type {
    inline static void Get(const MixedPoint *p, index_t i, CalcPrecision_t * const result) {
      boost::mpl::for_each<DenseTypes_t>(
        LocalGetDense(p, i, result));
    }
    inline static void Set(MixedPoint *p, index_t i, CalcPrecision_t * const result) {
      boost::mpl::for_each<DenseTypes_t>(
        LocalSetDense(p, i, result));
    }

  };
};

struct NullaryMetaFunctionOptimizedAccessSparse {
  struct type {
    inline static void Get(const MixedPoint *p, index_t i, CalcPrecision_t * const result) {
      *result = p->sparse_point <
                typename boost::mpl::front<SparseTypes_t>::type > ().get(i);
    }
    inline static void Set(MixedPoint *p, index_t i, CalcPrecision_t * const result) {
      p->sparse_point <
           typename boost::mpl::front<SparseTypes_t>::type > ().set(i, *result);
    }

  };
};

struct NullaryMetaFunctionNonOptimizedAccessSparse {
  struct type {
    inline static void Get(const MixedPoint *p, index_t i, CalcPrecision_t * const result) {
      if (*result != std::numeric_limits<CalcPrecision_t>::max() &&
          HasSparse::value == true) {

      } else {
        boost::mpl::for_each<SparseTypes_t>(LocalGetSparse(
                                              p, i, result));
      }
    }
    inline static void Set(MixedPoint *p, index_t i, CalcPrecision_t * const result) {
        boost::mpl::for_each<SparseTypes_t>(LocalSetSparse(
                                              p, i, result));
    }

  };
};

template<typename PointType, typename PrecisionType>
struct DenseSetAllOperator {
  DenseSetAllOperator(PointType *p, const PrecisionType value) :
      p_(p), v_(value) {
  }
  template<typename T>
  void operator()(T) {
    p_->template dense_point<T>().SetAll(v_);
  }
private:
  PointType *p_;
  const PrecisionType v_;
};

template<typename PointType, typename PrecisionType>
struct SparseSetAllOperator {
  SparseSetAllOperator(PointType *p, const PrecisionType value) :
      p_(p), v_(value) {
  }
  template<typename T>
  void operator()(T) {
    if (v_!=0) {
      fl::logger->Die()<<"You are trying to set all the values of a sparse matrix "
            "to a non zero value!!!!";
    }
    p_->template sparse_point<T>().SetAll(v_);
  }
private:
  PointType *p_;
  const PrecisionType v_;
};

template<typename PointType>
struct DenseSetRandomOperator {
  typedef typename PointType::CalcPrecision_t CalcPrecision_t;
  DenseSetRandomOperator(PointType *p, 
      const CalcPrecision_t low,
      const CalcPrecision_t hi) :
      p_(p), low_(low), hi_(hi) {
  }
  template<typename T>
  void operator()(T) {
    p_->template dense_point<T>().SetRandom(low_, hi_);
  }
private:
  PointType *p_;
  const CalcPrecision_t low_;
  const CalcPrecision_t hi_;
};

template<typename PointType>
struct SparseSetRandomOperator {
  typedef typename PointType::CalcPrecision_t CalcPrecision_t;
  SparseSetRandomOperator(PointType *p, 
      const CalcPrecision_t low, 
      const CalcPrecision_t hi, 
      const CalcPrecision_t sparsity) :
      p_(p), low_(low), hi_(hi), sparsity_(sparsity) {
  }
  template<typename T>
  void operator()(T) {
    p_->template sparse_point<T>().SetRandom(low_, hi_, sparsity_);
  }
private:
  PointType *p_;
  const CalcPrecision_t low_;
  const CalcPrecision_t hi_;
  const CalcPrecision_t sparsity_;
};

/**
 *  @brief Operators for serialization
 *         if mode = 0 it serializes, and for
 *            mode = 1 it desirializes
 */
template<typename PointType, typename Archive>
struct SerializeDense {
    SerializeDense(const PointType *point, Archive &ar) :
      point_(point), ar_(ar) {};
    template<typename T>
    void operator()(T) {
      ar_ << boost::serialization::make_nvp(
          std::string("dense_point_").append(fl::data::Typename<T>::Name()).c_str(), 
          point_->template dense_point<T>());
    }

  private:
    const PointType *point_;
    Archive &ar_;

};

template<typename PointType, typename Archive>
struct DeSerializeDense {
    DeSerializeDense(PointType *point, Archive &ar) :
      point_(point), ar_(ar) {};
    template<typename T>
    void operator()(T) {
      ar_ >> boost::serialization::make_nvp(
          std::string("dense_point_").append(fl::data::Typename<T>::Name()).c_str(), 
            point_->template dense_point<T>());
      
    }

  private:
    PointType *point_;
    Archive &ar_;

};

template<typename PointType, typename Archive>
struct SerializeSparse {
    SerializeSparse(const PointType *point, Archive &ar) :
      point_(point), ar_(ar) {};
    template<typename T>
    void operator()(T) {
      ar_ << boost::serialization::make_nvp(
        std::string("sparse_point_").append(fl::data::Typename<T>::Name()).c_str(), 
          point_->template sparse_point<T>());
    }

  private:
    const PointType *point_;
    Archive &ar_;
};

template<typename PointType, typename Archive>
struct DeSerializeSparse {
    DeSerializeSparse(PointType *point, Archive &ar) :
      point_(point), ar_(ar) {};
    template<typename T>
    void operator()(T) {
      ar_ >> boost::serialization::make_nvp(
          std::string("sparse_point_").append(fl::data::Typename<T>::Name()).c_str(), 
            point_->template sparse_point<T>());
    }

  private:
    PointType *point_;
    Archive &ar_;
};

struct MetaDataSerialize1 {
  struct type {
    template<typename PointType, typename Archive>
    static void Serialize(const PointType *point, Archive &ar) {
      ar << boost::serialization::make_nvp("meta_data_", point->meta_data());
    }

    template<typename PointType, typename Archive>
    static void Serialize(PointType *point, Archive &ar) {
      ar >> boost::serialization::make_nvp("meta_data_", const_cast<PointType *>(point)->meta_data());
    }
  };
};

struct MetaDataSerialize2 {
 struct type {
   template<typename PointType, typename Archive>
   static void Serialize(const PointType *point, Archive &ar) {
   }

   template<typename PointType, typename Archive>
   static void Serialize(PointType *point, Archive &ar) {
   }
 };
};

/**
 * @brief Accessing a value from a dense container
 *        when only dense containers are present
 *
 */
class GetDenseValue {
  public:
    template<typename T>
    static T get(MixedPoint *p,  index_t i) {
      index_t offset = DenseOffset::template offset<T>(p);
      return static_cast<wrap<T, MonolithicPoint> &>(p->dense()).value[i-offset];
    }
    template<typename T>
    static void set(MixedPoint *p,  index_t i, T value) {
      index_t offset = DenseOffset::template offset<T>(p);
      static_cast<wrap<T, MonolithicPoint> &>(p->dense()).value.set(i - offset, value);
    }

};

/**
 *  @brief Accessing a sparse value only when a sparse
 *         container is present
 *
 */
class GetSparseValue {
  public:
    template<typename T>
    static T get(const MixedPoint *p,  index_t i) {
      index_t offset = SparseOffset::template offset<T>(p);
      return static_cast<const sparse_wrap<T> &>(p->sparse()).value[i-offset];
    }
    template<typename T>
    static void set(MixedPoint *p,  index_t i, T value) {
      index_t offset = SparseOffset::template offset<T>(p);
      static_cast<wrap<T, SparsePoint> &>(p->sparse()).value.set(i - offset, value);
    }

};

/**
 *  @brief Accessing a value when a sparse and a densr container
 *         is present. It identifies automatically if
 *         if t comes from a dense or a sparse container
 *
 */
class GetDenseSparseValue {
  public:
    template<typename T>
    static T get(MixedPoint *p,  index_t i) {
      if (DenseOffset::template offset<T>(p) < i) {
        GetDenseValue::template get<T>(p, i);
      }
      else {
        GetSparseValue::template get<T>(p, i);
      }
    }
    template<typename T>
    static void  set(MixedPoint *p,  index_t i, T value) {
      if (DenseOffset::template offset<T>(p) <= i) {
        GetDenseValue::template set<T>(p, i, value);
      }
      else {
        GetSparseValue::template set<T>(p, i, value);
      }
    }

};

/**
 * @brief It is used to determine the offset of a specific type
 *
 */
template<typename PointCollection>
struct Size {
public:
  Size(const PointCollection *point, index_t *size) : size_(size), p_(point) {
  }
  template<typename Type>
  void operator()(Type) {
    typedef typename boost::mpl::if_ < boost::is_same<PointCollection, SparsePointCollection_t>,
    sparse_wrap<Type>, dense_wrap<Type> >::type WrapCast;
    (*size_) += static_cast<const WrapCast &>(*p_).value.size();
  }
private:
  index_t *size_;
  const PointCollection *p_;
};

/**
 * @brief Given a dense type it finds its offset in the
 *        mixed point
 *
 */
class DenseOffset {
  public:
    template<typename QType>
    static index_t offset(const MixedPoint *point) {
      typedef typename boost::mpl::find<DenseTypes_t, QType>::type EndIt;
      typedef typename boost::mpl::iterator_range <
      typename boost::mpl::begin<DenseTypes_t>::type, EndIt >
      PreviousSequence;
      index_t size = 0;
      boost::mpl::for_each<PreviousSequence>(
        Size<DensePointCollection_t>(&(point->dense()), &size));
      return size;
    }
};

/**
 * @brief Given a sparse type it finds its offset in the
 *        mixed point
 *
 */

class SparseOffset {
  public:

    struct NullaryMeta1 {
      struct type {
        static index_t offset(const MixedPoint *point) {
          return 0;
        }
      };
    };
    struct NullaryMeta2 {
      struct type {
        static index_t offset(const MixedPoint *point) {
          // This part is tricky here
          // If the point has been instanciated with Sets then we need to reverse them and
          // take the first
          // if it is with vectors as it happens with multidataset we can use back
          /***** notice the bracket operator works only if dense has one type or no type ****/
          BOOST_MPL_ASSERT((boost::mpl::equal_to <
                            boost::mpl::size<DenseTypes_t>,
                            boost::mpl::int_<1>
                            >));
          typedef typename boost::mpl::front<DenseTypes_t>::type LastDenseType;
          index_t size = DenseOffset::template offset<LastDenseType>(point) +
          point->template dense_point<LastDenseType>().size();
          return size;
        }
      };
    };

    template<typename QType>
    static index_t offset(const MixedPoint *point) {
      typedef typename boost::mpl::find<SparseTypes_t, QType>::type EndIt;
      typedef typename  boost::mpl::iterator_range <
      typename boost::mpl::begin<SparseTypes_t>::type, EndIt >
      PreviousSequence;

      index_t size =
        boost::mpl::eval_if <
        boost::mpl::empty<DenseTypes_t>,
        NullaryMeta1,
        NullaryMeta2
        >::type::offset(point);
      boost::mpl::for_each<PreviousSequence>(
        Size<SparsePointCollection_t>(&(point->sparse()), &size));
      return size;
    }
};

private:
/**
*        structs used for several for_each statements. They cannot be declared
*        inside functions, due to their template arguments
*
*/

/**
 * @brief this is used for loading point sizes in the Init function
 */
struct LoadDenseSizes {
  LoadDenseSizes(MixedPoint *p, std::vector<index_t> *sizes, index_t *count) {
    sizes_ = sizes;
    count_ = count;
    p_ = p;
  }
  template<typename T>
  void operator()(T) {
    p_->dense_point<T>().Init((*sizes_)[*count_]);
    (*count_)++;
  }
private:
  index_t *count_;
  std::vector<index_t> *sizes_;
  MixedPoint *p_;
};

struct LoadSparseSizes {
  LoadSparseSizes(MixedPoint *p, std::vector<index_t> *sizes, index_t *count) {
    p_ = p;
    sizes_ = sizes;
    count_ = count;
  }
  template<typename T>
  void operator()(T) {
    p_->sparse_point<T>().Init((*sizes_)[*count_]);
    (*count_)++;
  }
private:
  MixedPoint *p_;
  index_t *count_;
  std::vector<index_t> *sizes_;
};
/**
 * @brief this is used in the Alias
 */
struct DenseAlias {
  DenseAlias(MixedPoint *p1, const MixedPoint *p2) :
      p1_(p1), p2_(p2) {
  }
  template<typename T>
  void operator()(T) {
    p1_->dense_point<T>().Alias(p2_->dense_point<T>());
  }
  MixedPoint *p1_;
  const MixedPoint *p2_;
};
struct SparseAlias {
  SparseAlias(MixedPoint *p1, const MixedPoint *p2) :
      p1_(p1), p2_(p2) {
  }
  template<typename T>
  void operator()(T) {
    p1_->sparse_point<T>().Alias(p2_->sparse_point<T>());
  }
  MixedPoint *p1_;
  const MixedPoint *p2_;
};

struct MetaDataAliasSelector1 {
  struct type {
    static inline void Alias(MixedPoint * const p1, const MixedPoint *p2) {
      static_cast<this_wrap<MetaData_t* > &>(
        p1->collection_).value
      = static_cast<this_wrap<MetaData_t*> &>(
          const_cast<MixedPoint *>(p2)->collection_).value;
    }
  };
};

struct MetaDataAliasSelector2 {
  struct type {
    static inline void Alias(const MixedPoint *p1, const MixedPoint *p2) {
    }
  };
};

/**
 * @brief this is used in the Copy
 */
struct DenseCopy {
  DenseCopy(MixedPoint *p1, const MixedPoint *p2) :
      p1_(p1), p2_(p2) {
  }
  template<typename T>
  void operator()(T) {
    p1_->dense_point<T>().Copy(p2_->dense_point<T>());
  }
  MixedPoint *p1_;
  const MixedPoint *p2_;
};
struct SparseCopy {
  SparseCopy(MixedPoint *p1, const MixedPoint *p2) :
      p1_(p1), p2_(p2) {
  }
  template<typename T>
  void operator()(T) {
    p1_->sparse_point<T>().Copy(p2_->sparse_point<T>());
  }
  MixedPoint *p1_;
  const MixedPoint *p2_;
};

struct MetaDataCopySelector1 {
  struct type {
    static inline void Copy(MixedPoint * const p1, const MixedPoint * p2) {

      if (p1->should_free_ == true) {
        delete  static_cast<this_wrap<MetaData_t* > &>(p1->collection_).value;
      }
      try {
        static_cast<this_wrap<MetaData_t* > &>(p1->collection_).value = new MetaData_t();
      }
      catch(const std::bad_alloc &e) {
        fl::logger->Die() << "There was a problem allocating memory. "
          << "Either your dataset doesn't fit in the RAM, or"
          << "you are using a 32bit platform that limits the process "
          << "address space to 4GB";
      }


      *(static_cast<this_wrap<MetaData_t* > &>(
          p1->collection_).value)
      = *(static_cast<this_wrap<MetaData_t*> &>(
            const_cast<MixedPoint*>(p2)->collection_).value);

    }
  };
};

struct MetaDataCopySelector2 {
  struct type {
    static inline void Copy(const MixedPoint *p1, const MixedPoint *p2) {
    }
  };
};

/**
 * @brief this is used in the CopyValues
 */
struct DenseCopyValues {
  DenseCopyValues(MixedPoint *p1, const MixedPoint *p2) :
      p1_(p1), p2_(p2) {
  }
  template<typename T>
  void operator()(T) {
    p1_->dense_point<T>().CopyValues(p2_->dense_point<T>());
  }
  MixedPoint *p1_;
  const MixedPoint *p2_;
};
struct SparseCopyValues {
  SparseCopyValues(MixedPoint *p1, const MixedPoint *p2) :
      p1_(p1), p2_(p2) {
  }
  template<typename T>
  void operator()(T) {
    p1_->sparse_point<T>().CopyValues(p2_->sparse_point<T>());
  }
  MixedPoint *p1_;
  const MixedPoint *p2_;
};

struct MetaDataCopyValuesSelector1 {
  struct type {
    static inline void CopyValues(MixedPoint * const p1, const MixedPoint * p2) {
      *(static_cast<this_wrap<MetaData_t* > &>(
          p1->collection_).value)
      = *(static_cast<this_wrap<MetaData_t*> &>(
            const_cast<MixedPoint*>(p2)->collection_).value);

    }
  };
};

struct MetaDataCopyValuesSelector2 {
  struct type {
    static inline void CopyValues(const MixedPoint *p1, const MixedPoint *p2) {
    }
  };
};

/**
 * @brief this is used for swapping points
 */
struct DenseSwap {
  DenseSwap(MixedPoint *p1, MixedPoint *p2) {
    p1_ = p1;
    p2_ = p2;
  }
  template<typename T>
  void operator()(T) {
    p1_->dense_point<T>().SwapValues(&(p2_->dense_point<T>()));
  }
  MixedPoint *p1_;
  MixedPoint *p2_;
};
struct SparseSwap {
  SparseSwap(MixedPoint *p1, MixedPoint *p2) {
    p1_ = p1;
    p2_ = p2;
  }
  template<typename T>
  void operator()(T) {
    p1_->sparse_point<T>().SwapValues(&(p2_->sparse_point<T>()));
  }
  MixedPoint *p1_;
  MixedPoint *p2_;
};

/**
 *  @brief this is used inside the bracket operator
 */
struct LocalGetDense {
  LocalGetDense(const MixedPoint *p, index_t ind, CalcPrecision_t *result) :
   p_(p), ind_(ind), result_(result)
  {}
  template<typename T>
  void operator()(T) {
    size_t lo = DenseOffset::template offset<T>(p_);
    size_t hi = lo +  p_->dense_point<T>().size();
    if (lo <= static_cast<size_t>(ind_) && static_cast<size_t>(ind_) < hi) {
      *result_ = static_cast<CalcPrecision_t>(
                   p_->dense_point<T>().get(ind_ -lo));
    }
  }
 private:
  const MixedPoint *p_;
  index_t ind_;
  CalcPrecision_t *result_;
};

/**
 *  @brief this is used inside the set function
 */
struct LocalSetDense {
  LocalSetDense(MixedPoint *p, index_t ind, CalcPrecision_t *result) :
   p_(p), ind_(ind), result_(result) {
  }
  template<typename T>
  void operator()(T) {
    size_t lo = DenseOffset::template offset<T>(p_);
    size_t hi = lo +  p_->dense_point<T>().size();
    if (lo <= static_cast<size_t>(ind_) && static_cast<size_t>(ind_) < hi) {
      p_->dense_point<T>().set(ind_ -lo, *result_);
    }
  }
 private:
  MixedPoint *p_;
  index_t ind_;
  CalcPrecision_t *result_;
};

/**
 *  @brief this is used inside the bracket operator
 */

struct LocalGetSparse {
  LocalGetSparse(const MixedPoint *p, index_t ind, CalcPrecision_t *result) :
    p_(p), ind_(ind), result_(result)  {
  }
  template<typename T>
  void operator()(T) {
    size_t lo = SparseOffset::template offset<T>(p_);
    size_t hi = lo +  p_->sparse_point<T>().size();
    if (lo <= static_cast<size_t>(ind_) && static_cast<size_t>(ind_) < hi) {
      *result_ = static_cast<CalcPrecision_t>(p_->template sparse_point<T>().get(ind_-lo));
    }
  }
private:
  const MixedPoint *p_;
  index_t ind_;
  CalcPrecision_t *result_;
};

/**
 *  @brief this is used inside the set operator
 */

struct LocalSetSparse {
  LocalSetSparse(MixedPoint *p, index_t ind, CalcPrecision_t *result) :
  p_(p), ind_(ind), result_(result) {
  }
  template<typename T>
  void operator()(T) {
    size_t lo = SparseOffset::template offset<T>(p_);
    size_t hi = lo +  p_->sparse_point<T>().size();
    if (lo <= static_cast<size_t>(ind_) && static_cast<size_t>(ind_) < hi) {
      p_->template sparse_point<T>().set(ind_-lo, *result_);
    }
  }
private:
  MixedPoint *p_;
  index_t ind_;
  CalcPrecision_t *result_;
};

struct CollectionSelector1 {
  static boost::mpl::void_ get(const MixedPoint *p) {
    return boost::mpl::void_();
  }
};

template<typename T>
struct CollectionSelector2 {
  static  T &get(MixedPoint *p) {
    return static_cast<this_wrap<T> &>(p->collection_).value;
  }
  static  const T &get(const MixedPoint *p) {
    return static_cast<const this_wrap<T> &>(p->collection_).value;
  }
};


  template<typename PointType>
  struct IsNonZeroDense {
    IsNonZeroDense(const PointType &p, bool *is_non_zero) :
      p_(p), is_non_zero_(is_non_zero) {
    }
    template<typename T>
    void operator()(T) {
      if (p_.template dense_point<T>().IsZero()==false) {
        *is_non_zero_=true;
      } 
    }
    private:
      const PointType &p_;
      bool *is_non_zero_;
  };

  template<typename PointType>
  struct IsNonZeroSparse {
    IsNonZeroSparse(const PointType &p, bool *is_non_zero) :
      p_(p), is_non_zero_(is_non_zero) {
    }
    template<typename T>
    void operator()(T) {
      if (p_.template sparse_point<T>().IsZero()==false) {
        *is_non_zero_=false;
      } 
    }
    private:
      const PointType &p_;
      bool *is_non_zero_;
  };

  template<typename PointType>
  struct NonZeroDense {
    NonZeroDense(const PointType &p, size_t *non_zero) :
      p_(p), non_zero_(non_zero) {
    }
    template<typename T>
    void operator()(T) {
      *non_zero_ += p_.template dense_point<T>().nnz(); 
    }
    private:
      const PointType &p_;
      size_t *non_zero_;
  };

  template<typename PointType>
  struct NonZeroSparse {
    NonZeroSparse(const PointType &p, size_t *non_zero) :
      p_(p), non_zero_(non_zero) {
    }
    template<typename T>
    void operator()(T) {
      *non_zero_ += p_.template sparse_point<T>().nnz(); 
    }
    private:
      const PointType &p_;
      size_t *non_zero_;
  };


  template<typename FunctionType>
  class TransformOperatorDense {
    public:
      TransformOperatorDense(FunctionType &f, MixedPoint *p, index_t *offset) :
        f_(f, *offset), p_(p), offset_(offset) {
      }
      template<typename T>
      void operator()(T) {
        p_->template dense_point<T>().Transform(f_);
        *offset_+=p_->template dense_point<T>().size();
      }
    private:
      class Func {
        public:
          Func(FunctionType &f, index_t offset) :
            f_(f), offset_(offset) {}
          template<typename T>
          void operator()(const index_t &i, T *v) {
            f_(i+offset_, v);
          }
        private:
          FunctionType &f_;
          index_t offset_;
      };
      Func f_;
      MixedPoint *p_;
      index_t *offset_;
  };

  template<typename FunctionType>
  class TransformOperatorSparse {
    public:
      TransformOperatorSparse(FunctionType &f, MixedPoint *p, index_t *offset) :
        f_(f, *offset), p_(p), offset_(offset) {
      }
      template<typename T>
      void operator()(T) {
        p_->template sparse_point<T>().Transform(f_);
        *offset_+=p_->template sparse_point<T>().size();
      }
    private:
      class Func {
        public:
          Func(FunctionType &f, index_t offset) :
            f_(f), offset_(offset) {}
          template<typename T>
          void operator()(const index_t &i, T *v) {
            f_(i+offset_, v);
          }
        private:
          FunctionType &f_;
          index_t offset_;
      };
      Func f_;
      MixedPoint *p_;
      index_t *offset_;  
  };

#endif
