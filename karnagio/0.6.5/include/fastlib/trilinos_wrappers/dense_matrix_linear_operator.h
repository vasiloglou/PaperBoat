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

#ifndef FL_LITE_FASTLIB_TRILINOS_WRAPPERS_DENSE_MATRIX_LINEAR_OPERATOR_H
#define FL_LITE_FASTLIB_TRILINOS_WRAPPERS_DENSE_MATRIX_LINEAR_OPERATOR_H


#include "boost/utility.hpp"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/data/monolithic_point.h"
#include "fastlib/trilinos_wrappers/linear_operator.h"
#include <vector>

namespace Anasazi {

class DenseMatrixLinearOperator: public LinearOperator {

  private:

    const fl::dense::Matrix<double, false> *dense_matrix_;

  public:

#ifdef EPETRA_MPI
    DenseMatrixLinearOperator(const Epetra_MpiComm &comm_in,
                              const Epetra_Map &map_in) {
      comm_ = &comm_in;
      map_ = &map_in;
    }
#else
    DenseMatrixLinearOperator(const Epetra_SerialComm &comm_in,
                              const Epetra_Map &map_in) {
      comm_ = &comm_in;
      map_ = &map_in;
    }
#endif

    void Init(const fl::dense::Matrix<double, false> &dense_matrix_in) {
      dense_matrix_ = &dense_matrix_in;
    }

    int Apply(const Epetra_MultiVector &vec,
              Epetra_MultiVector &prod) const {

      prod.PutScalar(0);
      for (int j = 0; j < dense_matrix_->n_cols(); j++) {
        for (int i = 0; i < dense_matrix_->n_rows(); i++) {
          for (int k = 0; k < vec.NumVectors(); k++) {
            prod.Pointers()[k][i] += dense_matrix_->get(i, j) *
                                     vec.Pointers()[k][j];
          }
        }
      }
      return 0;
    }

    int n_rows() const {
      return dense_matrix_->n_rows();
    }

    int n_cols() const {
      return dense_matrix_->n_cols();
    }

    double get(int row, int col) const {
      return dense_matrix_->get(row, col);
    }
};

template<>
class OperatorTraits<double, Epetra_MultiVector, DenseMatrixLinearOperator> {
  public:

    static void Apply(const Epetra_Operator& Op,
                      const Epetra_MultiVector& x,
                      Epetra_MultiVector& y) {
      Op.Apply(x, y);
    }
};
};

#endif
