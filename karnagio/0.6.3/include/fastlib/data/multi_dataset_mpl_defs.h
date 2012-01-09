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
#ifndef FL_LITE_FASTLIB_DATA_MULTIDATASET_MPL_DEFS_H_
#define FL_LITE_FASTLIB_DATA_MULTIDATASET_MPL_DEFS_H_

// typedef typename ParameterList::DenseTypes DenseTypeList1;
typedef typename ParameterList::DenseTypes DenseTypeList_t;
// we moved to vectors so the following is uneccassary

// for some reason the types in the set are put in reverse order so we have to fix that
//typedef typename boost::mpl::reverse_fold <
//DenseTypeList1,
//boost::mpl::vector<>,
//boost::mpl::push_back <
//boost::mpl::placeholders::_1,
//boost::mpl::placeholders::_2
//>
//>::type DenseTypeList_t;


typedef typename ParameterList::SparseTypes SparseTypeList_t;
// same for the sparse
//typedef typename ParameterList::SparseTypes SparseTypeList1;

//typedef typename boost::mpl::reverse_fold <
//SparseTypeList1,
//boost::mpl::vector<>,
//boost::mpl::push_back <
//boost::mpl::placeholders::_1,
//boost::mpl::placeholders::_2
//>
//>::type SparseTypeList_t;

typedef typename ParameterList::MetaDataType MetaDataType_t;

typedef typename boost::mpl::if_ <
boost::is_same <
typename ParameterList::MetaDataType,
typename DatasetArgs::MetaDataType
> ,
boost::mpl::bool_<false>,
boost::mpl::bool_<true>
>::type HasMetaData_t;

// get the CaclPrecision
typedef typename ParameterList::CalcPrecisionType CalcPrecision_t;
/*
    typedef typename boost::mpl::if_<
    typename boost::mpl::has_key<ParameterList, DatasetArgs::CalcPrecision>::type,
    typename boost::mpl::at<ParameterList,
    typename DatasetArgs::CalcPrecision>::type,
    double >::type CalcPrecision_t;
*/

typedef typename ParameterList::StorageType Storage_t;
/*
    typedef typename boost::mpl::if_<
    typename boost::mpl::has_key<ParameterList, DatasetArgs::StorageType>::type,
    typename boost::mpl::at<ParameterList, DatasetArgs::StorageType>::type,
    typename DatasetArgs::StorageType::Extendable>::type Storage_t;
*/
// This is the case where the DenseTypeList is just an integral type
// the SparseTypeList is empty so the point is a MonolithicPoint
typedef typename
boost::mpl::if_ <
boost::mpl::and_ <
boost::mpl::and_ <
boost::mpl::empty<SparseTypeList_t>,
boost::mpl::equal_to <
boost::mpl::size<DenseTypeList_t>,
boost::mpl::int_<1>
>
> ,
boost::mpl::not_<HasMetaData_t>
> ,
boost::mpl::bool_<true>,
boost::mpl::bool_<false>
>::type IsMonolithicOnly_t;

typedef typename boost::mpl::if_<
  boost::mpl::empty<SparseTypeList_t>,
  boost::mpl::bool_<true>,
  boost::mpl::bool_<false>
>::type IsDenseOnly_t;

typedef typename boost::mpl::if_<
  boost::mpl::and_<
    boost::mpl::empty<SparseTypeList_t>,
    boost::mpl::and_<
      boost::mpl::equal_to<
        boost::mpl::size<DenseTypeList_t>,
        boost::mpl::int_<1>
      >,
      boost::is_same<
        Storage_t,
        typename DatasetArgs::Compact
      >
    >
  >,
  boost::mpl::bool_<true>,
  boost::mpl::bool_<false>
>::type IsNativeMatrix_t;

// This flag is important because some algorithms have a special version when dealing
// with matrix
typedef typename
boost::mpl::if_ <
  boost::mpl::and_ <
    boost::mpl::and_ <
      boost::mpl::empty<SparseTypeList_t>,
      boost::mpl::equal_to <
        boost::mpl::size<DenseTypeList_t>,
        boost::mpl::int_<1>
      >
    > ,
    boost::is_same<
      Storage_t,
      typename DatasetArgs::Compact
    >
  > ,
  // this is a hack
  boost::mpl::bool_<true>,
  boost::mpl::bool_<false>
>::type IsMatrixOnly_t;


typedef typename boost::mpl::if_ <
boost::mpl::empty<DenseTypeList_t>,
boost::mpl::bool_<false>,
boost::mpl::bool_<true>
>::type HasDense_t;

// we use the same trick if the all we need is a sparse point
typedef typename
boost::mpl::if_ <
boost::mpl::and_ <
boost::mpl::and_ <
boost::mpl::empty<DenseTypeList_t>,
boost::mpl::equal_to <
boost::mpl::size<SparseTypeList_t>,
boost::mpl::int_<1>
>
> ,
boost::mpl::not_<HasMetaData_t>
> ,
boost::mpl::bool_<true>,
boost::mpl::bool_<false>
>::type IsSparseOnly_t;


// this is the case we need both sparse and dense. This distinction is not necessary
// any more since MixedPoint is super optimized now to hold only the necessary points.
// In the older version it would waste one byte if sparse xor dense was missing
typedef typename boost::mpl::if_ <
boost::mpl::and_ <
boost::mpl::not_<IsSparseOnly_t>,
boost::mpl::not_<IsMonolithicOnly_t>
> ,
boost::mpl::bool_<true>,
boost::mpl::bool_<false>
>::type IsMixed_t;

// this is necessary, when it is dense compact then we can optimize and use a dense matrix
// we should do the same optimization in the future for sparse points too by using trilinos
// sparse matrices
typedef typename
boost::mpl::if_ <
  boost::mpl::and_ <
    boost::mpl::not_<IsSparseOnly_t>,
    boost::is_same <
      Storage_t,
      DatasetArgs::Compact
    >
  > ,
  boost::mpl::bool_<true>,
  boost::mpl::bool_<false>
>::type IsDenseCompact_t;

// In this section we are constructing the mpl::map that will have the parameters
// of a point
/*
    typedef typename boost::mpl::map<> PointMap0;

    typedef typename boost::mpl::if_<
      boost::mpl::empty<DenseTypeList_t>,
      PointMap0,
      typename boost::mpl::insert<
        PointMap0,
        boost::mpl::pair<
          MixedPointArgs::DenseTypeList,
          DenseTypeList_t
        >
      >::type>::type PointMap1;

    typedef typename boost::mpl::if_<boost::mpl::empty<SparseTypeList_t>,
    PointMap1,
    typename boost::mpl::insert<
    PointMap1,
    boost::mpl::pair<
    MixedPointArgs::SparseTypeList,
    SparseTypeList_t
    >
    >::type
    >::type PointMap2;

    typedef typename boost::mpl::if_<
      boost::mpl::not_<
        boost::is_same<
          typename ParameterList::MetaDataType,
          typename DatasetArgs::MetaDataType
        >
      >,
      typename boost::mpl::insert<
        PointMap2,
        boost::mpl::pair<
          typename MixedPointArgs::MetaDataType,
          typename ParameterList::MetaDataType
        >
      >::type,
      PointMap2
    >::type PointMap3;

    typedef typename boost::mpl::insert<
    PointMap3,
    boost::mpl::pair<
    MixedPointArgs::CalcPrecisionType,
    CalcPrecision_t
    >
    >::type PointMap4;
*/

struct MixedNullaryMetafunction {
  typedef MixedPoint<ParameterList> type;
};
struct DenseOnlyNullaryMetafunction {
  typedef MonolithicPoint<typename boost::mpl::front<DenseTypeList_t>::type > type;
};

struct SparseNullaryMetafunction {
  typedef SparsePoint<typename boost::mpl::front<SparseTypeList_t>::type> type;
};


// Here we generate the type of the Point that this dataset is going to use
typedef typename
boost::mpl::eval_if <
IsMonolithicOnly_t,
DenseOnlyNullaryMetafunction,
typename boost::mpl::eval_if <
IsSparseOnly_t,
SparseNullaryMetafunction,
MixedNullaryMetafunction
>
>::type Point_t;

// we do this trick to optimize for GenMatrix<GenMatrix<> >
// into GenMatrix<>, at some point we will need to do the same
// for the sparse matrices to use a trilinos one.
template<typename T>
struct IsMonolithic {
  static const bool value =
    boost::mpl::if_ < boost::is_same < MonolithicPoint <
    typename T::CalcPrecision_t > , T > ,
    boost::mpl::true_, boost::mpl::false_ >::type::value;
};

template<typename T>
class CompactContainer : public
      boost::mpl::if_ < IsMonolithic<T>,
      fl::dense::Matrix<typename T::CalcPrecision_t, false>,
      fl::dense::Matrix<T, false> >::type {
  public:
    typedef typename boost::mpl::if_ < IsMonolithic<T>,
    fl::dense::Matrix<typename T::CalcPrecision_t, false>,
    fl::dense::Matrix<T, false> >::type type;

};

// the other containers are simple STL ones
template<typename T>
class ExtendableContainer : public std::vector<T> {
  public:
    typedef  std::vector<T>  type;
    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & boost::serialization::make_nvp("vector", 
          boost::serialization::base_object<std::vector<T> >(*this));
    }
};

template<typename T>
class DeletableContainer : public std::list<T> {
  public:
    typedef std::list<T> type;
};


/**
 * @brief this typedef generates a list of Matrices
 *        of different types. We need that when we store
 *        points in matrices
 */
template < typename TypeList,
template <typename> class PointType,
template <typename> class ContainerType >
class PointCollection {
  public:
    template<typename PointType_1, template<typename> class Container>
    struct wrap {
      typedef typename Container<PointType_1>::type StorageBox_t;
      StorageBox_t value;
    };

    template<typename PrecisionType>
    class this_wrap : public wrap < PointType<PrecisionType>,
          ContainerType > {};
    class  Generated :  public
          boost::mpl::inherit_linearly < TypeList,
          boost::mpl::inherit < this_wrap<boost::mpl::placeholders::_2>,
          boost::mpl::placeholders::_1 > >::type {
      public:
        template<typename T>
        static typename this_wrap<T>::StorageBox_t &get(Generated &box) {
          return static_cast<this_wrap<T> &>(box).value;
        }
        template<typename T>
        typename this_wrap<T>::StorageBox_t &get() {
          return get<T>(*this);
        }

        template<typename Archive>
        struct SerializeOperator {
          SerializeOperator(Generated *p, Archive &ar) :
            p_(p), ar_(ar) {}

          template<typename T>
          void operator()(T) {
            ar_ & boost::serialization::make_nvp(
                fl::data::Typename<T>::Name().c_str(), 
                p_->template get<T>());
          }

          private:
          Generated *p_;
          Archive &ar_;
        };

        template<typename Archive>
        inline void serialize(Archive &ar, const unsigned int version) {
          boost::mpl::for_each<TypeList>(
            SerializeOperator<Archive>(this, ar));
        }
    };
};

/**
 *  @brief This struct helps so that we can use meta_data with nonmixed points
 */
struct SetMetaDataTrait1 {
  struct type {
    template<typename PointType, typename MetaDataType>
    static void set_meta_data(PointType &p, MetaDataType m);
  };
};

struct SetMetaDataTrait2 {
  struct type {
    template<typename PointType, typename MetaDataType>
    static void set_meta_data(PointType &p, MetaDataType m);  };
};

#endif
