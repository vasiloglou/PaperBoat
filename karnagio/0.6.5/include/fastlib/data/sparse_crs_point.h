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
#ifndef FASTLIB_DATA_SPARSE_CRS_POINT_H_
#define FASTLIB_DATA_SPARSE_CSR_POINT_H_

#include <string.h>
#include "fastlib/base/base.h"

namespace fl {
namespace data {
template<typename IndexPrecisionType, typename ValuePrecisionType>
class SparseCrsPoint {
  public:
    typedef ValuePrecisionType CalcPrecision_t;
    SparseCrsPoint() : indices_(NULL), values_(NULL),
        size_(0), n_entries_(0), should_free_(false) {};

    SparseCrsPoint(index_t size) : indices_(NULL),
        values_(NULL), n_entries_(0),
        size_(size), should_free_(true) {};

    SparseCrsPoint(const SparseCrsPoint &other) {
      size_ = other.size_ ;
      n_entries_ = other.n_entries_;
      should_free_ = should_free_;
      if (n_entries_ != 0) {
        indices_ = new IndexPrecisionType[n_entries_];
        values_ = new ValuePrecisionType[n_entries_];
      }
      memcpy(indices_, other.indices_, n_entries_*sizeof(IndexPrecisionType));
      memcpy(values_, other.values_, n_entries_*sizeof(ValuePrecisionType));
    }

    ~SparseCrsPoint() {
      if (should_free_ == true) {
        delete []indices_;
        delete []values_;
      }
    }

    void Init(index_t size, index_t n_entries) {
      should_free_ = true;
      n_entries_ = n_entries;
      size_ = size;
      indices_ = new IndexPrecisionType[n_entries_];
      values_ = new ValuePrecisionType[n_entries_];
    }

    void Init(index_t size, index_t n_entries,
              IndexPrecisionType *indices,
              ValuePrecisionType *values) {
      size_ = size;
      n_entries_ = n_entries;
      should_free_ = true;
      indices_ = new IndexPrecisionType[n_entries_];
      values_ = new ValuePrecisionType[n_entries_];
      memcpy(indices_, indices, n_entries_*sizeof(IndexPrecisionType));
      memcpy(values_, values, n_entries_*sizeof(ValuePrecisionType));
    }

    template<typename ContainerType>
    void Init(index_t size, ContainerType &indices,
              ContainerType &values) {

    }

    void Alias(index_t size, index_t n_entries,
               IndexPrecisionType *indices,
               ValuePrecisionType *values) {
      size_ = size;
      n_entries_ = n_entries;
      should_free_ = false;
      indices_ = indices;
      values_ = values;
    }

    void Alias(const SparseCrsPoint &other) {
      size_ = other.size_;
      n_entries_ = other.n_entries_;
      should_free_ = false;
      indices_ = other.indices_;
      values_ = other.values_;
    }

    void Copy(const SparseCrsPoint& other) {
      if (should_free_ == true) {
        delete []indices_;
        delete []values_;
        indices_ = new IndexPrecisionType[other.n_entries_];
        values_ = new ValuePrecisionType[other.n_entries_];
      }
      size_ = other.size_;
      n_entries_ = other.n_entries_;
      should_free_ = other.should_free_;
      memcpy(indices_, other.indices_, n_entries_*sizeof(IndexPrecisionType));
      memcpy(values_, other.values_, n_entries_*sizeof(ValuePrecisionType));
    }

    void set(index_t i, CalcPrecision_t value) {
      DEBUG_BOUNDS(i, size_);
      IndexPrecisionType *loc = std::find(indices_, indices_ + n_entries_);
      if (loc == indices_ + n_entries_) {
        n_entries_++;
        IndexPrecisionType *temp1 = new IndexPrecisionType[n_entries_];
        ValuePrecisionType *temp2 = new ValuePrecisionType[n_entries_];
        memcpy(temp1, indices_, (n_entries_ -1)*sizeof(IndexPrecisionType));
        memcpy(temp2, values_, (n_entries_ -1)*sizeof(ValuePrecisionType));
        indices_[n_entries_-1] = i;
        values_[n_entries_-1] = value;
      }
      else {
        values_[static_cast<ptrdiff_t>(loc-indices_)] = value;
      }
    }

    CalcPrecision_t get(index_t i) const {
      DEBUG_BOUNDS(i, size_);
      IndexPrecisionType *loc = std::find(indices_, indices_ + n_entries_);
      if (loc == indices_ + n_entries_) {
        return 0;
      }
      else {
        return values_[static_cast<ptrdiff_t>(loc-indices_)];
      }
    }

    CalcPrecision_t operator[](const index_t i) const {
      return get(i);
    }


    void Swap(SparseCrsPoint *other) {
      std::swap(size_, other->size_);
      std::swap(n_entries_, other->n_entries_);
      std::swap(indices_, other->indices_);
      std::swap(values_, other->values_);
      std::swap(should_free_, other->should_free_);
    }

    const index_t size() const {
      return size_;
    }

    const index_t length() const {
      return size_;
    }

    ValuePrecisionType * values() {
      return values_;
    }


    IndexPrecisionType *indices()  {
      return indices_;
    }

    index_t nnz() const  {
      return n_entries_;
    }

  private:
    IndexPrecisionType* indices_;
    ValuePrecisionType* values_;
    index_t size_;
    index_t n_entries_;
    bool should_free_;
};
}
}

#endif
