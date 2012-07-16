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
#ifndef FL_LITE_MLPACK_KERNEL_PCA_DENSE_KERNEL_MATRIX_INVERSE_H
#define FL_LITE_MLPACK_KERNEL_PCA_DENSE_KERNEL_MATRIX_INVERSE_H

#include "fastlib/dense/matrix.h"
#include "boost/utility.hpp"

namespace fl {
namespace ml {
class DenseKernelMatrixInverse: boost::noncopyable {
  public:

    static fl::dense::Matrix<double, false> *Update(
      const fl::dense::Matrix<double, false> &previous_inverse,
      const fl::dense::Matrix<double, true> &inverse_times_kernel_vector,
      const double &projection_error) {

      fl::dense::Matrix<double, false> *new_kernel_matrix_inverse =
        new fl::dense::Matrix<double, false>();
      new_kernel_matrix_inverse->Init(previous_inverse.n_rows() + 1,
                                      previous_inverse.n_cols() + 1);

      for (int j = 0; j < previous_inverse.n_cols(); j++) {
        for (int i = 0; i < previous_inverse.n_rows(); i++) {
          new_kernel_matrix_inverse->set(
            i, j, previous_inverse.get(i, j) +
            inverse_times_kernel_vector[i] *
            inverse_times_kernel_vector[j] / projection_error);
        }
      }

      for (int j = 0; j < previous_inverse.n_cols(); j++) {
        new_kernel_matrix_inverse->set(j,
                                       previous_inverse.n_cols(), - inverse_times_kernel_vector[j] /
                                       projection_error);
        new_kernel_matrix_inverse->set(
          previous_inverse.n_rows(), j,
          - inverse_times_kernel_vector[j] / projection_error);
      }
      new_kernel_matrix_inverse->set(
        new_kernel_matrix_inverse->n_rows() - 1,
        new_kernel_matrix_inverse->n_cols() - 1,
        1.0 / projection_error);

      // Return the computed inverse.
      return new_kernel_matrix_inverse;
    }
};
};
};

#endif
