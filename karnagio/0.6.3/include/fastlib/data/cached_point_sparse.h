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

#ifndef FL_LITE_FASTLIB_DATA_LB_CACHED_POINT_SPARSE_H_
#define FL_LITE_FASTLIB_DATA_LB_CACHED_POINT_SPARSE_H_

#include "cached_point.h"
#include "fastlib/data/monolithic_point.h"
#include "fastlib/data/sparse_point.h"
#include "fastlib/data/mixed_point.h"

namespace fl{namespace data{
  template<typename CalcPrecisionType, typename MultiDatasetType>
  class CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType> 
    : public SparsePoint<CalcPrecisionType> {
    public:
	  typedef typename SparsePoint<CalcPrecisionType>::Container_t SparsePointContainer_t;
      typedef CalcPrecisionType CalcPrecision_t;
      typedef index_t Key_t;
      typedef typename SparsePoint<CalcPrecisionType>::iterator iterator;

      CachedPoint();
      CachedPoint(const CachedPoint& other);
      CachedPoint(const SparsePoint<CalcPrecisionType>& other);
      virtual ~CachedPoint();
      void Alias(const Key_t &row, MultiDatasetType * const data);
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
      void set(index_t i, CalcPrecision_t value);
      virtual SparsePointContainer_t *&elem();
      const Key_t row() const;
      bool &has_been_modified();
      
    private:
      MultiDatasetType *data_;
      Key_t row_;
      bool has_been_modified_;

  }; 

  template<typename CalcPrecisionType, typename MultiDatasetType>
  CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::CachedPoint() : data_(NULL),
  has_been_modified_(false) {  
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::CachedPoint(const CachedPoint &other) : 
      SparsePoint<CalcPrecisionType>::template SparsePoint<CalcPrecisionType>(other) {
    has_been_modified()=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::CachedPoint(
      const SparsePoint<CalcPrecisionType> &other) :
      SparsePoint<CalcPrecisionType>::template SparsePoint<CalcPrecisionType>(other) {
    has_been_modified()=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::~CachedPoint() {
    if (data_!=NULL && has_been_modified_==true) {       
       data_->CopyBack(row_, this);
    }
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  void CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::Alias(const Key_t &row, 
      MultiDatasetType *const data) {
    data_=data;
    row_=row;
    data_->CopyTo(this);
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  void CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::Alias(
      const CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType> &other) {
    SparsePoint<CalcPrecisionType>::Alias(other);
    has_been_modified_=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType> &CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::operator=(
    const CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType> &other) {
    MonolithicPoint<CalcPrecisionType>::CopyValues(other);
    has_been_modified()=true;
    return *this;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  template<typename PrecisionType>
  void CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::Copy(
      const fl::data::MonolithicPoint<PrecisionType> &other) {
    SparsePoint<CalcPrecisionType>::Copy(other);
    has_been_modified()=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  template<typename PrecisionType>
  void CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::CopyValues(
      const fl::data::MonolithicPoint<PrecisionType> &other) {
    SparsePoint<CalcPrecisionType>::CopyValues(other);
    has_been_modified()=true;
  }


  template<typename CalcPrecisionType, typename MultiDatasetType>
  template<typename PrecisionType>
  void CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::Copy(
      const fl::data::SparsePoint<PrecisionType> &other) {
    SparsePoint<CalcPrecisionType>::Copy(other);
    has_been_modified()=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>
  template<typename PrecisionType>
  void CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::CopyValues(
      const fl::data::SparsePoint<PrecisionType> &other) {
    SparsePoint<CalcPrecisionType>::CopyValues(other);
    has_been_modified()=true;
  }

  template<typename CalcPrecisionType, typename MultiDatasetType> 
  void CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::set(index_t i, CalcPrecisionType value) {
    SparsePoint<CalcPrecisionType>::set(i, value);
    has_been_modified()=true;
  }
  
  template<typename CalcPrecisionType, typename MultiDatasetType>
  typename CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::SparsePointContainer_t *&
      CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::elem() {

    has_been_modified()=true;
    return this->elem();
  }

  template<typename CalcPrecisionType, typename MultiDatasetType>  
  const typename CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::Key_t 
    CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::row() const {
      return row_;
  }


  template<typename CalcPrecisionType, typename MultiDatasetType>
  bool &CachedPoint<SparsePoint<CalcPrecisionType>, MultiDatasetType>::has_been_modified() {
    return has_been_modified_;
  }

}} // namespaces

#endif
