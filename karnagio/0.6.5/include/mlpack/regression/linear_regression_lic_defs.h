/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
LLC, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE ISMION INC "AS IS" AND ANY
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

#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_LIC_DEFS_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_LIC_DEFS_H
#include "linear_regression_lic.h"
namespace fl {namespace ml {
  template<typename TableType>
  void LinearRegressionLIC::Predict(const Vector_t &coefficients,
               const TableType &query_table,
               std::vector<double> *result) const {
    result->resize(query_table.n_entries());
    typename TableType::Point_t point;
    for(index_t i=0; i<query_table.n_entries(); ++i) {
      query_table.get(i ,&point);
      (*result)[i]=fl::la::Dot(coefficients, point);
    }
  };

  template<typename PointType>
  void LinearRegressionLIC::Predict(const Vector_t &coefficients, 
               const PointType &point, 
               double *result) const {
    *result=fl::la::Dot(coefficients, point);
  }

}}
#endif

