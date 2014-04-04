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

#ifndef FL_LITE_FASTLIB_DATA_LB_CACHED_POINT_MIXED_H_
#define FL_LITE_FASTLIB_DATA_LB_CACHED_POINT_MIXED_H_

#include "cached_point.h"
#include "fastlib/data/mixed_point.h"
namespace fl{namespace data{
  template<typename MixedPointParams, typename MultiDatasetType>
  class CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType> 
    : public MixedPoint<MixedPointParams> {
    public:
      typedef typename MixedPoint<MixedPointParams>::CalcPrecision_t CalcPrecision_t;
      typedef index_t Key_t;
      typedef typename boost::mpl::if_<
        boost::mpl::empty<typename MixedPoint<MixedPointParams>::DenseTypes_t>,
        fl::data::SparsePoint<
           typename boost::mpl::front<typename MixedPoint<MixedPointParams>::SparseTypes_t>::type
        >,
        fl::data::MonolithicPoint<
           typename boost::mpl::front<typename MixedPoint<MixedPointParams>::DenseTypes_t>::type
        >
      >::type::iterator iterator;

      CachedPoint();
      CachedPoint(const CachedPoint& other);
      CachedPoint(const MixedPoint<MixedPointParams>& other);
      iterator begin();
      iterator end();
      virtual ~CachedPoint();
      void Alias(const Key_t &row, MultiDatasetType * const data);
      void Alias(const CachedPoint &other);
      CachedPoint &operator=(const CachedPoint &other); 
      void Copy(const MixedPoint<MixedPointParams> &other); 
      void CopyValues(const MixedPoint<MixedPointParams> &other); 
      void set(index_t i, CalcPrecision_t value);
      const Key_t row() const;
      bool &has_been_modified();
      virtual void set_modified();
      
    private:
      MultiDatasetType *data_;
      Key_t row_;
      bool has_been_modified_;

  }; 

  template<typename MixedPointParams, typename MultiDatasetType>
  CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::CachedPoint() : data_(NULL),
  has_been_modified_(false) {  
  }

  template<typename MixedPointParams, typename MultiDatasetType>
  CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::CachedPoint(const CachedPoint &other) : 
      MixedPoint<MixedPointParams>::template MixedPoint<MixedPointParams>(other) {
    has_been_modified()=true;
  }

  template<typename MixedPointParams, typename MultiDatasetType>
  CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::CachedPoint(
      const MixedPoint<MixedPointParams> &other) :
      MixedPoint<MixedPointParams>::template MixedPoint<MixedPointParams>(other) {
    has_been_modified()=true;
  }
  
  struct return_iterator1 {
    template<typename PointType, typename IteratorType>
    static void DoBegin(PointType *point, IteratorType *it) {
      point->template dense_point<
        typename boost::mpl::front<
          typename PointType::DenseTypes_t
        >::type
      >().begin();
    }
 
    template<typename PointType, typename IteratorType>
    static void DoEnd(PointType *point, IteratorType *it) {
      point->template dense_point<
        typename boost::mpl::front<
          typename PointType::DenseTypes_t
        >::type
      >().end();
    }
 
  };

   struct return_iterator2 {
    template<typename PointType, typename IteratorType>
    static void DoBegin(PointType *point, IteratorType *it) {
      point->template sparse_point<
        typename boost::mpl::front<
          typename PointType::SparseTypes_t
        >::type
      >().begin();
    }
 
    template<typename PointType, typename IteratorType>
    static void DoEnd(PointType *point, IteratorType *it) {
      point->template sparse_point<
        typename boost::mpl::front<
          typename PointType::SparseTypes_t
        >::type
      >().end();
    }
 
  };


  template<typename MixedPointParams, typename MultiDatasetType>
  typename CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::iterator 
      CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::begin() {
    iterator it;
    boost::mpl::if_<
        boost::mpl::empty<typename MixedPoint<MixedPointParams>::DenseTypes_t>,
        return_iterator2,
        return_iterator1      
    >::type::DoBegin(this, &it);
    return it;  
  }

  template<typename MixedPointParams, typename MultiDatasetType>
  typename CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::iterator
      CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::end() {
    iterator it;
    boost::mpl::if_<
        boost::mpl::empty<typename MixedPoint<MixedPointParams>::DenseTypes_t>,
        return_iterator2,
        return_iterator1      
    >::type::DoEnd(this, &it);  
    return it;
  }

  template<typename MixedPointParams, typename MultiDatasetType>
  CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::~CachedPoint() {
    if (data_!=NULL && has_been_modified_==true) {       
       data_->CopyBack(row_, this);
    }
  }

  template<typename MixedPointParams, typename MultiDatasetType>
  void CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::Alias(const Key_t &row, 
      MultiDatasetType *const data) {
    data_=data;
    row_=row;
    data_->CopyTo(this);
  }

  template<typename MixedPointParams, typename MultiDatasetType>
  void CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::Alias(
      const CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType> &other) {
    MixedPoint<MixedPointParams>::Alias(other);
    has_been_modified_=true;
  }

  template<typename MixedPointParams, typename MultiDatasetType>
  void CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::Copy(
      const MixedPoint<MixedPointParams> &other) {
      MixedPoint<MixedPointParams>::Copy(other);
      has_been_modified_=true;
  }

  template<typename MixedPointParams, typename MultiDatasetType>
  void CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::CopyValues(
      const MixedPoint<MixedPointParams> &other) {
      MixedPoint<MixedPointParams>::CopyValues(other);
      has_been_modified_=true;
  }
  template<typename MixedPointParams, typename MultiDatasetType>
  CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType> &CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::operator=(
    const CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType> &other) {
    MixedPoint<MixedPointParams>::CopyValues(other);
    has_been_modified()=true;
    return *this;
  }


  template<typename MixedPointParams, typename MultiDatasetType>
  void CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::set(index_t i, 
      CalcPrecision_t value) {
    MixedPoint<MixedPointParams>::set(i, value);
    has_been_modified()=true;
  }
 

  template<typename MixedPointParams, typename MultiDatasetType>
  const typename CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::Key_t 
    CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::row() const {
      return row_;
  }


  template<typename MixedPointParams, typename MultiDatasetType>
  bool &CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::has_been_modified() {
    return has_been_modified_;
  }

  template<typename MixedPointParams, typename MultiDatasetType>
  void CachedPoint<MixedPoint<MixedPointParams>, MultiDatasetType>::set_modified() {
    has_been_modified()=true;
  }

}};
#endif
