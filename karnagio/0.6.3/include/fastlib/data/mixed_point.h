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
#ifndef FL_LITE_FASTLIB_DATA_MIXED_POINT_H_
#define FL_LITE_FASTLIB_DATA_MIXED_POINT_H_
#include "boost/type_traits/is_same.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/mpl/push_back.hpp"
#include "boost/mpl/size.hpp"
#include "boost/mpl/contains.hpp"
#include "boost/mpl/equal_to.hpp"
#include "boost/mpl/if.hpp"
#include "boost/mpl/find.hpp"
#include "boost/mpl/insert.hpp"
#include "boost/mpl/iterator_range.hpp"
#include "boost/mpl/inherit.hpp"
#include "boost/mpl/inherit_linearly.hpp"
#include "boost/mpl/placeholders.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/mpl/size.hpp"
#include "boost/mpl/fold.hpp"
#include "boost/mpl/for_each.hpp"
#include "boost/mpl/equal_to.hpp"
#include "boost/mpl/back.hpp"
#include "boost/mpl/empty.hpp"
#include "boost/mpl/and.hpp"
#include "boost/mpl/assert.hpp"
#include "boost/static_assert.hpp"
#include "boost/serialization/split_member.hpp"
#include "boost/serialization/nvp.hpp"
#include "fastlib/base/base.h"
#include "typename.h"
#include "fastlib/data/monolithic_point.h"
#include "linear_algebra.h"

namespace fl {
namespace data {
template<typename CaclPrecision>
class SparsePoint;
template<typename T>
class MultiDataset;
}
}

namespace fl {
namespace data {
/**
 * @brief basic class to wrap the points
 *        in the composed classes
 *        that hold the points
 */
template<class T, template<typename> class Point>
struct wrap {
  Point<T> value;
};

struct MixedPointArgs {
  typedef boost::mpl::vector0<> DenseTypes;
  typedef boost::mpl::vector0<> SparseTypes;
  typedef double CalcPrecisionType;
  typedef boost::mpl::void_ SparseContainerType;
  typedef boost::mpl::void_  MetaDataType;
};


template<typename ParamList>
class MixedPoint : public fl::data::mixed_ops {
    /**
     * @brief simple constructor, does nothing
     */
#include "mixed_point_mpl_defs.h"
  public:
    friend class boost::serialization::access;
    class MixedIterator {
      public:
        MixedIterator() {
          fl::logger->Die() << "iterator for complex mixed_point" <<NOT_SUPPORTED_MESSAGE;
        }

        MixedIterator(const MixedIterator &other) {
          fl::logger->Die() << "iterator for complex mixed_point" <<NOT_SUPPORTED_MESSAGE;
        }


        const MixedIterator& operator=(const MixedIterator& other) {
          fl::logger->Die() << "iterator for complex mixed_point" <<NOT_SUPPORTED_MESSAGE;
        }

        bool operator==(const MixedIterator &other) const {
          fl::logger->Die() << "iterator for complex mixed_point" <<NOT_SUPPORTED_MESSAGE;
          return false;
        }

        bool operator!=(const MixedIterator &other) const {
          fl::logger->Die() << "iterator for complex mixed_point" <<NOT_SUPPORTED_MESSAGE;
          return false;
        }


        const CalcPrecision_t operator*() const {
          fl::logger->Die() << "iterator for complex mixed_point" <<NOT_SUPPORTED_MESSAGE;
          return 0;
        }

        const CalcPrecision_t value() const {
          fl::logger->Die() << "iterator for complex mixed_point" <<NOT_SUPPORTED_MESSAGE;
          return 0;
        }

        const index_t attribute() const {
          fl::logger->Die() << "iterator for complex mixed_point" <<NOT_SUPPORTED_MESSAGE;
          return 0;
        }

        const MixedIterator &operator++() {
          fl::logger->Die() << "iterator for complex mixed_point" <<NOT_SUPPORTED_MESSAGE;
          return *this;
        }

      private:
       DenseIteratorCollection_t dense_;
       SparseIteratorCollection_t sparse_; 
    };

    #include "mixed_iterator.h"
      typedef typename 
      boost::mpl::if_<
        boost::mpl::and_<
          boost::mpl::empty<DenseTypes_t>,
          boost::mpl::equal_to<
            boost::mpl::size<SparseTypes_t>,
            boost::mpl::int_<1>
          >
        >,
        typename SparsePoint<typename boost::mpl::front<SparseTypes_t>::type >::iterator,
        typename boost::mpl::if_<
          boost::mpl::and_<
            boost::mpl::empty<SparseTypes_t>,
            boost::mpl::equal_to<
              boost::mpl::size<DenseTypes_t>,
              boost::mpl::int_<1>
            >
          >,
          typename MonolithicPoint<typename boost::mpl::front<DenseTypes_t>::type >::iterator,
          MixedIterator
        >::type
      >::type iterator;

    template<typename T> friend class fl::data::MultiDataset;
    MixedPoint();
    virtual ~MixedPoint();
    /**
     * @brief constructor that sets the sizes
     */

    MixedPoint(std::vector<index_t> &sizes);
    /**
     * @copy constructor
     */
    MixedPoint(const MixedPoint &other);
    /**
     *  @brief Initializer
     *
     */
    inline void Init(const std::vector<index_t> &sizes);
    /**
     * @brief make this point alias of another point
     */
    inline void Alias(const MixedPoint &other);
    /**
     * @brief Copy another point
     *
     */
    inline void Copy(const MixedPoint &other);
     /**
      * @brief there are some cases that a monolithic point
      *        is wrapped around a mixed point with meta data
      *
      */
    template<typename PrecisionType>
    inline void Copy(const MonolithicPoint<PrecisionType> &other);
    /*
     *  @brief returns an iterator in the begining
     */
    inline iterator begin() const ;

    /*
     *  @brief returns an iterator in the end
     */
    inline iterator end() const ;


    /**
     * @brief Copy Values another point
     *
     */
    inline void CopyValues(const MixedPoint &other);
    /**
     * @brief Assignment operator
     *
     */
    inline MixedPoint& operator=(const MixedPoint &other);

    /**
     *  @brief swap values between points
     *
     */
    inline void SwapValues(MixedPoint *other);
    /**
     *  @brief Prints a mixed point, mainly for debuging purposes
     *
     */
    template<typename StreamType>
    void Print(StreamType &stream, std::string delim) const ;
    /**
     * @brief Destructs the point
     */
    void Destruct();
    /**
     * @this function is necessary so that the cached point can work
     *  It basically notifies the point if the content has changed
     */
    virtual void set_modified();

    inline typename boost::mpl::if_ <
    HasMetaData, MetaData_t&,
    signed char
    >::type meta_data();

    inline typename boost::mpl::if_ <
    HasMetaData, const MetaData_t&,
    signed char
    >::type meta_data() const ;

    inline void set_meta_data(const typename
                              boost::mpl::if_ <
                              HasMetaData,
                              MetaData_t,
                              int64
                              >::type &value);
    /**
      * @brief it returns the dense point of a specific type
      *
      */
    template<typename PrecisionType>
    inline MonolithicPoint<PrecisionType> &dense_point();

    template<typename PrecisionType>
    inline const MonolithicPoint<PrecisionType> &dense_point() const ;

    /**
     * @brief it returns the sparse point of a specific type
     *
     */
    template<typename PrecisionType>
    inline SparsePoint<PrecisionType> &sparse_point();

    template<typename PrecisionType>
    inline const SparsePoint<PrecisionType> &sparse_point() const;

    bool IsZero() const ;

    /**
     * @brief Transform every element of a point
     */
    template<typename FunctionType>
    void Transform(FunctionType &f);

    /**
     * @brief sets the index of a specific point to a value.
     *        automatically finds if it is sparse or
     *        dense
     *
     */
    inline void set(index_t i, CalcPrecision_t value);
    /**
     * @brief gets a reference to the index of a specific type
     *        it finds automatically from the index if it is
     *        dense or sparse in an automatic way
     *
     */
    template<typename PrecisionType>
    inline PrecisionType get(index_t i) const;
    /**
     * @brief a very inefficient way to get the value of a point
     *        at a specific index.
     *
     */
    inline CalcPrecision_t operator[](index_t i) const ;
    /**
     * @brief sets all elements to a specific value
     *  if the point contains sparse data and you try to
     *  set all to a nonzero value it will complain
     */
    template<typename PrecisionType>
    inline void SetAll(PrecisionType value);

    inline void SetRandom(const CalcPrecision_t low,
                          const CalcPrecision_t hi, 
                          const CalcPrecision_t sparsity=0);

    template<typename ResultMatrixType>
    void UpdateSefOuterProd(
        ResultMatrixType *result) const;

    template<typename ResultMatrixType,
             typename MixedPointArgsType>
    void UpdateOuterProd(
        const fl::data::MixedPoint<MixedPointArgsType> &x,
        ResultMatrixType *result) const;

    inline size_t size() const ;

    inline size_t length() const ;
    // serialization
    template<typename Archive>
    void save(Archive &ar,
              const unsigned int file_version) const ;

    template<typename Archive>
    void load(Archive &ar,
              const unsigned int file_version);

    BOOST_SERIALIZATION_SPLIT_MEMBER()


  private:

    /**
     * @brief
     */
    PointCollection_t collection_;

    /**
     * @brief the total size of the point
     */
    size_t size_;

    /**
     * @brief if it is aliased  then it doesn't free memory
     */
    bool should_free_;

    typename boost::mpl::if_ <
    HasMetaData, MetaData_t*,
    void * >::type &meta_data_ptr();

    inline typename boost::mpl::if_ <
    HasDense,
    DensePointCollection_t&,
    boost::mpl::void_
    >::type dense();

    inline  typename boost::mpl::if_ <
    HasDense,
    const DensePointCollection_t&,
    boost::mpl::void_
    >::type dense() const ;


    inline typename boost::mpl::if_ <
    HasSparse,
    SparsePointCollection_t&,
    boost::mpl::void_
    >::type sparse();

    inline  typename boost::mpl::if_ <
    HasSparse,
    const SparsePointCollection_t&,
    boost::mpl::void_
    >::type sparse() const ;

};

/**   Function Definitions **/
template<typename ParamList>
MixedPoint<ParamList>::MixedPoint() {
  should_free_ = false;
  meta_data_ptr() = NULL;
};

template<typename ParamList>
MixedPoint<ParamList>::~MixedPoint() {
  if (should_free_ == true) {
    boost::mpl::eval_if <
    HasMetaData,
    DestructSelector1,
    DestructSelector2
    >::type::Destruct(this);
  }
};

/**
 * @brief constructor that sets the sizes
 */
template<typename ParamList>
MixedPoint<ParamList>::MixedPoint(std::vector<index_t> &sizes) {
  Init(sizes);
}
/**
 * @brief Copy constructor
 */
template<typename ParamList>
MixedPoint<ParamList>::MixedPoint(const MixedPoint<ParamList>& other) {
  Copy(other);
}

/**
 *  @brief Initializer
 *
 */
template<typename ParamList>
void MixedPoint<ParamList>::Init(const std::vector<index_t> &sizes) {
  static const index_t num_of_types =
    boost::mpl::size<DenseTypes_t>::type::value
    + boost::mpl::size<SparseTypes_t>::type::value;
  DEBUG_ASSERT(sizes.size() == (size_t)num_of_types);
  size_ = 0;
  for (index_t i = 0; i < num_of_types; i++) {
    size_ += sizes[i];
  }
  index_t counter = 0;
  boost::mpl::for_each<DenseTypes_t>(
    LoadDenseSizes(this, &const_cast<std::vector<index_t> &>(sizes), &counter));
  boost::mpl::for_each<SparseTypes_t>(
    LoadSparseSizes(this, &const_cast<std::vector<index_t> &>(sizes), &counter));

  boost::mpl::eval_if <
  HasMetaData,
  MetaDataInitSelector1,
  MetaDataInitSelector2
  >::type::Init(this);
  should_free_ = true;
}

/**
 * @brief make this point alias of another point
 */
template<typename ParamList>
void MixedPoint<ParamList>::Alias(const MixedPoint &other) {
  should_free_ = false;
  boost::mpl::for_each<DenseTypes_t>(
    DenseAlias(this, &other));
  boost::mpl::for_each<SparseTypes_t>(
    SparseAlias(this, &other));
  boost::mpl::eval_if <
  HasMetaData,
  MetaDataAliasSelector1,
  MetaDataAliasSelector2
  >::type::Alias(this, &other);
  size_ = other.size();
}

/**
 * @brief copy another mixed point
 */
template<typename ParamList>
void MixedPoint<ParamList>::Copy(const MixedPoint &other) {
  boost::mpl::for_each<DenseTypes_t>(
    DenseCopy(this, &other));
  boost::mpl::for_each<SparseTypes_t>(
    SparseCopy(this, &other));
  boost::mpl::eval_if <
  HasMetaData,
  MetaDataCopySelector1,
  MetaDataCopySelector2
  >::type::Copy(this, &other);
  // don't move this from here it affects the for_each_statements
  should_free_ = true;
  size_ = other.size();
}
/*
 * @brief copies a monolithic point that has the same precision
 */
template<typename ParamList>
template<typename PrecisionType>
void MixedPoint<ParamList>::Copy(const MonolithicPoint<PrecisionType> &other) {
  // if this assertion fails it means that you cannot copy this ponolithic
  // point to the mixed point because of precision mismatch
  BOOST_MPL_ASSERT((boost::mpl::contains<DenseTypes_t, PrecisionType>));
  // It must not contains sparse
  BOOST_MPL_ASSERT((boost::mpl::empty<SparseTypes_t>));
  // We must also make sure that dense doesn't contain other types
  BOOST_MPL_ASSERT((boost::mpl::equal_to<
        boost::mpl::size<DenseTypes_t>,
        boost::mpl::int_<1>
        >));
  this->dense_point<PrecisionType>().Copy(other);
  should_free_=false;
}

template<typename ParamList>
typename MixedPoint<ParamList>::iterator MixedPoint<ParamList>::begin() const{
  return
    boost::mpl::if_<
      boost::mpl::and_<
        boost::mpl::empty<DenseTypes_t>,
        boost::mpl::equal_to<
          boost::mpl::size<SparseTypes_t>,
          boost::mpl::int_<1>
        >
      >,
      NullaryIterator1,
      typename boost::mpl::if_<
        boost::mpl::and_<
          boost::mpl::empty<SparseTypes_t>,
          boost::mpl::equal_to<
            boost::mpl::size<DenseTypes_t>,
            boost::mpl::int_<1>
          >
        >,
        NullaryIterator2,
        NullaryIterator3
      >::type
    >::type::begin(this);
}

template<typename ParamList>
typename MixedPoint<ParamList>::iterator MixedPoint<ParamList>::end() const {
   return 
    boost::mpl::if_<
      boost::mpl::and_<
        boost::mpl::empty<DenseTypes_t>,
        boost::mpl::equal_to<
          boost::mpl::size<SparseTypes_t>,
          boost::mpl::int_<1>
        >
      >,
      NullaryIterator1,
      typename boost::mpl::if_<
        boost::mpl::and_<
          boost::mpl::empty<SparseTypes_t>,
          boost::mpl::equal_to<
            boost::mpl::size<DenseTypes_t>,
            boost::mpl::int_<1>
          >
        >,
        NullaryIterator2,
        NullaryIterator3
      >::type
    >::type::end(this);

}

/**
 * @brief copy of values from another mixed point
 */
template<typename ParamList>
void MixedPoint<ParamList>::CopyValues(const MixedPoint &other) {
  boost::mpl::for_each<DenseTypes_t>(
    DenseCopyValues(this, &other));
  boost::mpl::for_each<SparseTypes_t>(
    SparseCopyValues(this, &other));
  boost::mpl::eval_if <
  HasMetaData,
  MetaDataCopyValuesSelector1,
  MetaDataCopyValuesSelector2
  >::type::CopyValues(this, &other);
}

/**
 * @brief assignment operator
 */
template<typename ParamList>
MixedPoint<ParamList> &MixedPoint<ParamList>::operator=(const MixedPoint &other) {
  CopyValues(other);
  return *this;
}

/**
 *  @brief swap values between points
 *
 */
template<typename ParamList>
void MixedPoint<ParamList>::SwapValues(MixedPoint *other) {
  boost::mpl::for_each<DenseTypes_t>(
    DenseSwap(this, other));
  boost::mpl::for_each<SparseTypes_t>(
    SparseSwap(this, other));
  boost::mpl::if_ <
  HasMetaData,
  MetaSwapSelector1,
  MetaSwapSelector2 >::type::Swap(this, other);
}

template<typename ParamList>
template<typename StreamType>
void MixedPoint<ParamList>::Print(StreamType &stream, std::string delim) const {
  boost::mpl::if_ <
  HasMetaData,
  PrintMetaDataSelector1,
  PrintMetaDataSelector2 >::type::Print(this, stream, delim);
  boost::mpl::for_each<DenseTypes_t>(
    PrintPointDense<StreamType>(this, stream, delim));
  boost::mpl::for_each<SparseTypes_t>(
    PrintPointSparse<StreamType>(this, stream, delim));
}

template<typename ParamList>
void MixedPoint<ParamList>::set_modified() {
};

template<typename ParamList>
typename boost::mpl::if_ <
typename MixedPoint<ParamList>::HasMetaData,
typename MixedPoint<ParamList>::MetaData_t&,
signed char >::type
MixedPoint<ParamList>::meta_data() {
  this->set_modified();
  return boost::mpl::if_<HasMetaData, MetaDataSelector1, MetaDataSelector2>::type::get(this);
}

template<typename ParamList>
typename boost::mpl::if_ <
typename MixedPoint<ParamList>::HasMetaData,
const typename MixedPoint<ParamList>::MetaData_t&,
signed char >::type
MixedPoint<ParamList>::meta_data() const  {
  return boost::mpl::if_<HasMetaData, MetaDataSelector1, MetaDataSelector2>::type::get(this);
}

template<typename ParamList>
void MixedPoint<ParamList>::set_meta_data(const typename
    boost::mpl::if_ <
    typename MixedPoint<ParamList>::HasMetaData,
    typename MixedPoint<ParamList>::MetaData_t,
    int64
    >::type &value) {
  set_modified();
  boost::mpl::if_ <
  HasMetaData,
  MetaDataSelector1,
  MetaDataSelector2
  >::type::set(this, value);
}

/**
 * @brief it returns the dense point of a specific type
 *
 */
template<typename ParamList>
template<typename PrecisionType>
MonolithicPoint<PrecisionType> & MixedPoint<ParamList>::dense_point() {
  set_modified();
  return static_cast<wrap<PrecisionType, MonolithicPoint> &>(this->dense()).value;
}

template<typename ParamList>
template<typename PrecisionType>
const MonolithicPoint<PrecisionType> & MixedPoint<ParamList>::dense_point() const {
  return static_cast<const wrap<PrecisionType,  MonolithicPoint> &>(this->dense()).value;
}

/**
 * @brief it returns the sparse point of a specific type
 *
 */
template<typename ParamList>
template<typename PrecisionType>
SparsePoint<PrecisionType> &MixedPoint<ParamList>::sparse_point() {
  set_modified();
  return static_cast<wrap<PrecisionType, SparsePoint> &>(this->sparse()).value;
}

template<typename ParamList>
template<typename PrecisionType>
const SparsePoint<PrecisionType> &MixedPoint<ParamList>::sparse_point() const {
  return static_cast<const wrap<PrecisionType, SparsePoint> &>(this->sparse()).value;
}

template<typename ParamList>
bool MixedPoint<ParamList>::IsZero() const {
  bool is_zero=true;
  boost::mpl::for_each<DenseTypes_t>(
      IsZeroDense<MixedPoint<ParamList> >(*this, &is_zero));
  if (is_zero==true) {
    boost::mpl::for_each<SparseTypes_t>(
        IsZeroSparse<MixedPoint<ParamList> >(*this, &is_zero));
  }
  return is_zero;
}

template<typename ParamList>
template<typename FunctionType>
void MixedPoint<ParamList>::Transform(FunctionType &f) {
  boost::mpl::for_each<DenseTypes_t>(TransformOperatorDense<FunctionType>(f, this));
  boost::mpl::for_each<SparseTypes_t>(TransformOperatorSparse<FunctionType>(f, this));
}


/**
 * @brief sets the index of a specific point to a value.
 *        automatically finds if it is sparse or
 *        dense. WARNING!!! never pass value by reference. The copy
 *        inside the function changes value
 *
 */
template<typename ParamList>
void MixedPoint<ParamList>::set(index_t i,
                                CalcPrecision_t value) {
  // it is possible that a mixed point has only one dense and
  // probably metadata. The first part of if statement optimizes access
  // so that the for_each is not called which will have more overhead
  boost::mpl::eval_if <
  boost::mpl::and_ <
  boost::mpl::equal_to <
  boost::mpl::size<DenseTypes_t>,
  boost::mpl::int_<1>
  > ,
  boost::mpl::not_<HasSparse>
  > ,
  NullaryMetaFunctionOptimizedAccessDense,
  NullaryMetaFunctionNonOptimizedAccessDense
  >::type::Set(this, i, &value);

  boost::mpl::eval_if <
  boost::mpl::and_ <
  boost::mpl::equal_to <
  boost::mpl::size<SparseTypes_t>,
  boost::mpl::int_<1>
  > ,
  boost::mpl::not_<HasDense>
  > ,
  NullaryMetaFunctionOptimizedAccessSparse,
  NullaryMetaFunctionNonOptimizedAccessSparse
  >::type::Set(this, i, &value);
  set_modified();
}

/**
 * @brief gets a reference to the index of a specific type
 *        it finds automatically from the index if it is
 *        dense or sparse in an automatic way
 *
 */
template<typename ParamList>
template<typename PrecisionType>
PrecisionType MixedPoint<ParamList>::get(index_t i) const {
  BOOST_STATIC_ASSERT(!(boost::mpl::contains < DenseTypes_t,
                        PrecisionType >::type::value == false &&
                        boost::mpl::contains<SparseTypes_t, PrecisionType>::type::value
                        == false));

  typename boost::mpl::if_c < boost::mpl::contains < DenseTypes_t,
  PrecisionType >::value == true
  &&
  boost::mpl::contains < SparseTypes_t,
  PrecisionType >::value == false,
  GetDenseValue,
  typename boost::mpl::if_c < boost::mpl::contains < DenseTypes_t,
  PrecisionType >::value == false
  &&
  boost::mpl::contains < SparseTypes_t,
  PrecisionType >::value == true,
  GetSparseValue, GetDenseSparseValue >::type
  >::type accessor;
  return accessor.template get<PrecisionType>(this, i);
}

/**
 * @brief a very inefficient way to get the value of a point
 *        at a specific index.
 *
 */
template<typename ParamList>
typename MixedPoint<ParamList>::CalcPrecision_t MixedPoint<ParamList>::operator[](index_t i) const {
  CalcPrecision_t result=std::numeric_limits<CalcPrecision_t>::max();
  // it is possible that a mixed point has only one dense and
  // probably metadata. The first part of if statement optimizes access
  // so that the for_each is not called which will have more overhead
  boost::mpl::eval_if <
  boost::mpl::and_ <
  boost::mpl::equal_to <
  boost::mpl::size<DenseTypes_t>,
  boost::mpl::int_<1>
  > ,
  boost::mpl::not_<HasSparse>
  > ,
  NullaryMetaFunctionOptimizedAccessDense,
  NullaryMetaFunctionNonOptimizedAccessDense
  >::type::Get(this, i, &result);

  boost::mpl::eval_if <
  boost::mpl::and_ <
  boost::mpl::equal_to <
  boost::mpl::size<SparseTypes_t>,
  boost::mpl::int_<1>
  > ,
  boost::mpl::not_<HasDense>
  > ,
  NullaryMetaFunctionOptimizedAccessSparse,
  NullaryMetaFunctionNonOptimizedAccessSparse
  >::type::Get(this, i, &result);
  return result;
}

template<typename ParamList>
template<typename PrecisionType>
void MixedPoint<ParamList>::SetAll(PrecisionType value) {
  boost::mpl::for_each<DenseTypes_t>(DenseSetAllOperator<MixedPoint, PrecisionType>
                                     (this, value));
  boost::mpl::for_each<SparseTypes_t>(SparseSetAllOperator<MixedPoint, PrecisionType>
                                      (this, value));
  set_modified();
}


template<typename ParamList>
void MixedPoint<ParamList>::SetRandom(const CalcPrecision_t low,
    const CalcPrecision_t hi, 
    const CalcPrecision_t sparsity) {

  boost::mpl::for_each<DenseTypes_t>(DenseSetRandomOperator<MixedPoint>
                                     (this, low, hi));
  boost::mpl::for_each<SparseTypes_t>(SparseSetRandomOperator<MixedPoint>
                                      (this, low, hi, sparsity));
  set_modified();

}


template<typename PointType1,
         typename ResultType,
         typename DenseList,
         typename SparseList>
struct UpdateSelfOuterDense1 {
  public:
    UpdateSelfOuterDense1(
        const PointType1 &point,
        ResultType *result,
        index_t *ind) :
      point_(point),  
      result_(result), ind_(ind) {}

    template<typename T>
    void operator()(T) {
      point_.template dense_point<T>().
          UpdateSelfOuterProd(*ind_, result_);
      typedef typename boost::mpl::find<DenseList, T>::type It_t; 
      typedef typename boost::mpl::insert<
        boost::mpl::vector0<>,
        typename boost::mpl::advance<
          It_t, 
          boost::mpl::int_<1>
        >::type,
        typename boost::mpl::end<DenseList>::type
      >::type TheRestDense_t;

      boost::mpl::for_each<TheRestDense_t>(
          UpdateSelfOuter2(point_, ind_, result_));  
      boost::mpl::for_each<SparseList>(
          UpdateSelfOuter3(point_, ind_, result_));

    }

  private:
    struct UpdateSelfOuter2 {
      public:
        UpdateSelfOuter2(const PointType1 &point,
            ResultType *result,
            index_t *ind) :
          point_(point), result_(result), ind_(ind) {} 
        template<typename T>
        void operator()(T) {
          point_.template dense_point<T>().UpdateSelfOuter(
              result_, *ind_);
          *ind_+=point_.template dense_point<T>.size();
        }
      private:
        const PointType1 &point_;
        ResultType *result_;
        index_t *ind_;
    };
    
    struct UpdateSelfOuter3 {
      public:
        UpdateSelfOuter3(const PointType1 &point,
            ResultType *result,
            index_t *ind) :
          point_(point), result_(result), ind_(ind) {} 
        template<typename T>
        void operator()(T) {
          point_.template sparse_point<T>().UpdateSelfOuter(
              result_, *ind_);
          *ind_+=point_.template dense_point<T>.size();
        }
      private:
        const PointType1 &point_;
        ResultType *result_;
        index_t *ind_;
    };

    const PointType1 &point_;
    ResultType *result_;
    index_t *ind_; 
  
};

template<typename PointType1,
         typename ResultType,
         typename SparseList>
struct UpdateSelfOuterSparse1 {
  public:
    UpdateSelfOuterSparse1(
        const PointType1 &point,
        ResultType *result,
        index_t *ind) :
      point_(point),  
      result_(result), ind_(ind) {}

    template<typename T>
    void operator()(T) {
      point_.template sparse_point<T>().
          UpdateSelfOuterProd(*ind_, result_);
      typedef typename boost::mpl::find<SparseList, T>::type It_t; 
      typedef typename boost::mpl::insert<
        boost::mpl::vector0<>,
        typename boost::mpl::advance<
          It_t, 
          boost::mpl::int_<1>
        >::type,
        typename boost::mpl::end<SparseList>::type
      >::type TheRestDense_t;

      boost::mpl::for_each<SparseList>(
          UpdateSelfOuter2(point_, ind_, result_));

    }

  private:
    struct UpdateSelfOuter2 {
      public:
        UpdateSelfOuter2(const PointType1 &point,
            ResultType *result,
            index_t *ind) :
          point_(point), result_(result), ind_(ind) {} 
        template<typename T>
        void operator()(T) {
          point_.template sparse_point<T>().UpdateSelfOuter(
              result_, *ind_);
          *ind_+=point_.template sparse_point<T>.size();
        }
      private:
        const PointType1 &point_;
        ResultType *result_;
        index_t *ind_;
    };
    
    const PointType1 &point_;
    ResultType *result_;
    index_t *ind_; 
  
};

template<typename ParamList>
template<typename ResultMatrixType>
void MixedPoint<ParamList>::UpdateSefOuterProd(
     ResultMatrixType *result) const {
  
  index_t ind=0;
  boost::mpl::for_each<DenseTypes_t>(
      UpdateSelfOuterDense1<MixedPoint<ParamList>,
                       ResultMatrixType,
                       DenseTypes_t,
                       SparseTypes_t>(*this, result, &ind));
  boost::mpl::for_each<SparseTypes_t>(
      UpdateSelfOuterSparse1<
        MixedPoint<ParamList>,
                   ResultMatrixType,
                   SparseTypes_t>(*this, result, &ind));
}



template<typename ParamList>
size_t MixedPoint<ParamList>::size() const {
  return size_;
}

template<typename ParamList>
size_t MixedPoint<ParamList>::length() const {
  return size_;
}


template<typename ParamList>
template<typename Archive>
void MixedPoint<ParamList>::save(Archive &ar,
          const unsigned int file_version) const {
  
  boost::mpl::for_each<DenseTypes_t>(SerializeDense<MixedPoint, Archive>(this, ar));
  boost::mpl::for_each<SparseTypes_t>(SerializeSparse<MixedPoint, Archive>(this, ar));
  boost::mpl::eval_if <
    HasMetaData,
    MetaDataSerialize1,
    MetaDataSerialize2
  >::type::Serialize(this, ar);
  ar << boost::serialization::make_nvp("size", size_);

}

template<typename ParamList>
template<typename Archive>
void MixedPoint<ParamList>::load(Archive &ar,
    const unsigned int file_version) {

  if (should_free_ == true) {
    boost::mpl::eval_if <
    HasMetaData,
    DestructSelector1,
    DestructSelector2
    >::type::Destruct(this);
  }

  boost::mpl::eval_if <
    HasMetaData,
    MetaDataInitSelector1,
    MetaDataInitSelector2
  >::type::Init(this);

  boost::mpl::for_each<DenseTypes_t>(DeSerializeDense<MixedPoint, Archive>(this, ar));
  boost::mpl::for_each<SparseTypes_t>(DeSerializeSparse<MixedPoint, Archive>(this, ar));
  boost::mpl::eval_if <
    HasMetaData,
    MetaDataSerialize1,
    MetaDataSerialize2
  >::type::Serialize(this, ar);

  ar >> boost::serialization::make_nvp("size", size_);
  should_free_=true;
  set_modified();
}


template<typename ParamList>
typename boost::mpl::if_ <
typename MixedPoint<ParamList>::HasMetaData,
typename MixedPoint<ParamList>::MetaData_t*,
void *
>::type &MixedPoint<ParamList>::meta_data_ptr() {
  return  boost::mpl::eval_if <
          HasMetaData,
          MetaDataPointerSelector1,
          MetaDataPointerSelector2
          >::type::Do(this);
   set_modified();
}

template<typename ParamList>
typename boost::mpl::if_ <
typename MixedPoint<ParamList>::HasDense,
typename MixedPoint<ParamList>::DensePointCollection_t&,
boost::mpl::void_
>::type MixedPoint<ParamList>::dense() {
  return boost::mpl::if_ < HasDense, CollectionSelector2<DensePointCollection_t>,
         CollectionSelector1 >::type::get(this);
  set_modified();
}

template<typename ParamList>
typename boost::mpl::if_ <
typename MixedPoint<ParamList>::HasDense,
const typename MixedPoint<ParamList>::DensePointCollection_t&,
boost::mpl::void_
>::type MixedPoint<ParamList>::dense() const {
  return boost::mpl::if_ < HasDense, CollectionSelector2<DensePointCollection_t>,
         CollectionSelector1 >::type::get(this);
}

template<typename ParamList>
typename boost::mpl::if_ <
typename MixedPoint<ParamList>::HasSparse,
typename MixedPoint<ParamList>::SparsePointCollection_t&,
boost::mpl::void_
>::type MixedPoint<ParamList>::sparse() {
  return boost::mpl::if_ < HasSparse, CollectionSelector2<SparsePointCollection_t>,
         CollectionSelector1 >::type::get(this);
  set_modified();
}

template<typename ParamList>
typename boost::mpl::if_ <
typename MixedPoint<ParamList>::HasSparse,
const typename MixedPoint<ParamList>::SparsePointCollection_t&,
boost::mpl::void_
>::type MixedPoint<ParamList>::sparse() const {
  return boost::mpl::if_ < HasSparse, CollectionSelector2<SparsePointCollection_t>,
         CollectionSelector1 >::type::get(this);
}

} // namespace data
}  // namespace fl

#endif
