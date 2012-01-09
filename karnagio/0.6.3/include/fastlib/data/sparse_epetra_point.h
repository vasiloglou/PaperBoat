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
#ifndef FASTLIB_DATA_SPARSE_EPETRA_POINT_H_
#define FASTLIB_DATA_SPARSE_EPETRA_POINT_H_
#include "fastlib/base/base.h"
#include "ops.h"

namespace fl {
namespace sparse {
template<typename IndexPrecisionType, typename ValuePrecisionType>
class Matrix;
}
}

namespace fl {
namespace data {
template<typename IndexPrecisionType, typename ValuePrecisionType>
class SparseEpetraPoint : public ops  {
  public:
    typedef fl::sparse::Matrix<IndexPrecisionType, ValuePrecisionType> Matrix_t;
    typedef ValuePrecisionType CalcPrecision_t;
    SparseEpetraPoint() : row_(0), mat_(NULL) {
    }
    SparseEpetraPoint(IndexPrecisionType row, Matrix_t *mat): row_(row), mat_(mat) {
    }
    IndexPrecisionType &row() {
      return row_;
    }
    Matrix_t* &matrix() {
      return mat_;
    }

    CalcPrecision_t get(IndexPrecisionType col) const {
      return mat_->get(row_, col);
    }

    void set(IndexPrecisionType col, ValuePrecisionType value) {
      mat_->set(row_, col, value);
    }

    IndexPrecisionType *indices() {
      IndexPrecisionType *nnz;
      IndexPrecisionType **indices;
      ValuePrecisionType **values;
      elements(nnz, indices, values);
      return *indices;
    }
    ValuePrecisionType *values() {
      IndexPrecisionType *nnz;
      IndexPrecisionType **indices;
      ValuePrecisionType **values;
      elements(nnz, indices, values);
      return *values;
    }
    IndexPrecisionType nnz() {
      IndexPrecisionType *nnz;
      IndexPrecisionType **indices;
      ValuePrecisionType **values;
      elements(nnz, indices, values);
      return *nnz;
    }
    void elements(IndexPrecisionType *nnz,
                  IndexPrecisionType **indices,
                  ValuePrecisionType **values) {
      mat_->get_row_view(row_, *nnz, *indices, *values);
    }
    void SwapValues(SparseEpetraPoint *other) {
      mat_->SwapRows(row_, other->row_, other->mat_);
    }
  private:
    fl::sparse::Matrix<IndexPrecisionType, ValuePrecisionType> *mat_;
    IndexPrecisionType row_;
};
} // namespace data
}  // namespace fl

#endif

