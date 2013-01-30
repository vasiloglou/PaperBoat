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

#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_RESULT_DEV_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_RESULT_DEV_H

#include "mlpack/regression/linear_regression_result.h"

namespace fl {
namespace ml {

template<typename TableType>
void LinearRegressionResult<TableType>::Init(const TableType &query_table_in) {
  query_table_ = &query_table_in;
  predictions_.Init(query_table_->n_entries());
  residual_sum_of_squares_ = 0;
}

template<typename TableType>
fl::data::MonolithicPoint<double> &LinearRegressionResult<TableType>::
predictions() {
  return predictions_;
}

template<typename TableType>
const fl::data::MonolithicPoint<double> &LinearRegressionResult<TableType>::
predictions() const {
  return predictions_;
}

template<typename TableType>
double LinearRegressionResult<TableType>::residual_sum_of_squares() const {
  return residual_sum_of_squares_;
}

template<typename TableType>
double &LinearRegressionResult<TableType>::residual_sum_of_squares() {
  return residual_sum_of_squares_;
}
};
};

#endif
