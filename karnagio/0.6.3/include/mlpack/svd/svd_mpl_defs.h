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
#ifndef FL_LITE_MLPACK_SVD_SVD_MPL_DEFS_H
#define FL_LITE_MLPACK_SVD_SVD_MPL_DEFS_H

struct  NullaryMetaInitMatrix1 {
  template<typename TableType, typename MatrixType>
  static void Init(TableType &table, MatrixType * const matrix) {
    typedef typename TableType::CalcPrecision_t CalcPrecision_t;
    matrix->Alias(table.get_point_collection().dense->
                  template get<CalcPrecision_t>());
  }
  template<typename MatrixType>
  static void Init(index_t rows, index_t columns, MatrixType * const matrix) {
    matrix->Init(columns, rows);
  }
  template<typename MatrixType, typename TableType>
  static void CopyBack(MatrixType &matrix, TableType * const table) {
    typedef typename TableType::CalcPrecision_t CalcPrecision_t;
    table->Destruct();
    table->Init(matrix);
  }
};


struct  NullaryMetaInitMatrix2 {
  template<typename TableType, typename MatrixType>
  static void Init(TableType &table, MatrixType * const matrix) {
    matrix->Init(table.n_attributes(), table.n_entries());
    for (index_t i = 0; i < matrix->n_cols(); ++i) {
      typename TableType::Point_t point;
      table.get(i, &point);
      for (index_t j = 0; j < point.size(); ++j) {
        matrix->set(j, i, point[j]);
      }
    }
  }
  template<typename MatrixType>
  static void Init(index_t rows, index_t columns, MatrixType * const matrix) {
    matrix->Init(columns, rows);
  }
  template<typename MatrixType, typename TableType>
  static void CopyBack(MatrixType &matrix, TableType * const table) {
    for (index_t i = 0; i < matrix.n_cols(); ++i) {
      typename TableType::Point_t point;
      table->get(i, &point);
      for (index_t j = 0; j < point.size(); ++j) {
        point.set(j, matrix.get(j, i));
      }
    }
  }
};





#endif
