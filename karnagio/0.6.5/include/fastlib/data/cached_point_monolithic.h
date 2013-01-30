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
#ifndef FL_LITE_FASTLIB_DATA_LB_CACHED_POINT_MONOLITHIC_H_
#define FL_LITE_FASTLIB_DATA_LB_CACHED_POINT_MONOLITHIC_H_

#include "boost/scoped_ptr.hpp"
#include "cached_point.h"
#include "fastlib/data/monolithic_point.h"
#include "fastlib/data/sparse_point.h"
#include "fastlib/data/mixed_point.h"

namespace fl{namespace data{

  template<typename CalcPrecisionType, typename MultiDatasetType>
  class CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType> 
    : public MonolithicPoint<CalcPrecisionType>  {
    public:
      typedef CalcPrecisionType CalcPrecision_t;
      typedef index_t Key_t;
      typedef typename MonolithicPoint<CalcPrecisionType>::iterator iterator;

      CachedPoint();
      CachedPoint(const CachedPoint& other);
      CachedPoint(const MonolithicPoint<CalcPrecisionType>& other);
      virtual ~CachedPoint();
      void Alias(const Key_t &row, MultiDatasetType *const data);
      void Alias(const CachedPoint &other);
      CachedPoint &operator=(const CachedPoint &other); 
      template<typename PrecisionType>
      void Copy(const fl::data::MonolithicPoint<PrecisionType> &other); 
      template<typename PrecisionType>
      void CopyValues(const fl::data::MonolithicPoint<PrecisionType> &other); 
      template<typename PrecisionType>
      void Copy(const fl::data::SparsePoint<PrecisionType> &other); 
      template<typename PrecisionType>
      void CopyValues(const fl::data::SparsePoint<PrecisionType> &other); 
      template<typename ParamList>
      void Copy(const fl::data::MixedPoint<ParamList> &other); 
      template<typename ParamList>
      void CopyValues(const fl::data::MixedPoint<ParamList> &other);
      CalcPrecision_t &operator[](index_t i);
      const CalcPrecision_t &operator[](index_t i) const;
      void set(index_t i, CalcPrecision_t value);
      virtual CalcPrecision_t *ptr();
      const Key_t row() const;
      bool &has_been_modified();

    private:
      MultiDatasetType *data_;
      Key_t row_;
      bool has_been_modified_;

  }; 

  template<typename CalcPrecisionType, typename MultiDatasetType>
  CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::CachedPoint() : data_(NULL),
  has_been_modified_(false) {  
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::CachedPoint(const CachedPoint &other) : 
      MonolithicPoint<CalcPrecisionType>(other) {
    has_been_modified()=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::~CachedPoint() {
    if (data_!=NULL && has_been_modified_==true) {       
      data_->CopyBack(row_, this);
    }
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  void CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::Alias(const Key_t &row, 
      MultiDatasetType * const data) {
    data_=data;
    row_=row;
    data_->CopyTo(this);
    has_been_modified()=false;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  void CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::Alias( 
      const CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType> &other) {
    MonolithicPoint<CalcPrecisionType>::Alias(other);
    has_been_modified_=true;
  }


  template<typename CalcPrecisionType, typename MultiDatasetType>
  CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType> & CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::operator=(
    const CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType> &other) {
    MonolithicPoint<CalcPrecisionType>::CopyValues(other);
    has_been_modified()=true;
    return *this;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  template<typename PrecisionType>
  void CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::Copy(
      const fl::data::MonolithicPoint<PrecisionType> &other) {
    MonolithicPoint<CalcPrecisionType>::Copy(other);
    has_been_modified()=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  template<typename PrecisionType>
  void CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::CopyValues(
      const fl::data::MonolithicPoint<PrecisionType> &other) {
    MonolithicPoint<CalcPrecisionType>::CopyValues(other);
    has_been_modified()=true;
  }


  template<typename CalcPrecisionType, typename MultiDatasetType>
  template<typename PrecisionType>
  void CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::Copy(
      const fl::data::SparsePoint<PrecisionType> &other) {
    MonolithicPoint<CalcPrecisionType>::Copy(other);
    has_been_modified()=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  template<typename PrecisionType>
  void CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::CopyValues(
      const fl::data::SparsePoint<PrecisionType> &other) {
    MonolithicPoint<CalcPrecisionType>::CopyValues(other);
    has_been_modified()=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  template<typename ParamList>
  void CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::Copy(const fl::data::MixedPoint<ParamList> &other) {
    MonolithicPoint<CalcPrecisionType>::Copy(other);
    has_been_modified()=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  template<typename ParamList>
  void CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::CopyValues(
      const fl::data::MixedPoint<ParamList> &other) {
    MonolithicPoint<CalcPrecisionType>::CopyValues(other);
    has_been_modified()=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  CalcPrecisionType &CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::operator[](index_t i) {
    has_been_modified()=true;
    return MonolithicPoint<CalcPrecisionType>::operator[](i);
  }
 
  template<typename CalcPrecisionType, typename MultiDatasetType>
  const CalcPrecisionType &CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::operator[](index_t i) const {
    return MonolithicPoint<CalcPrecisionType>::operator[](i);
  }
 

  template<typename CalcPrecisionType, typename MultiDatasetType> 
  void CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::set(index_t i, CalcPrecisionType value) {
    MonolithicPoint<CalcPrecisionType>::set(i, value);
    has_been_modified()=true;
  }
  
  template<typename CalcPrecisionType, typename MultiDatasetType>  
  CalcPrecisionType *CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::ptr() {
    has_been_modified()=true;
    return MonolithicPoint<CalcPrecisionType>::ptr();
  }
 
  template<typename CalcPrecisionType, typename MultiDatasetType>  
  const typename CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::Key_t 
    CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::row() const {
      return row_;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  bool &CachedPoint<MonolithicPoint<CalcPrecisionType>, MultiDatasetType>::has_been_modified() {
    return has_been_modified_;
  }



}} // namespaces
#endif
