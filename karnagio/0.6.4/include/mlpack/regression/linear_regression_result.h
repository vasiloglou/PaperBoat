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

#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_RESULT_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_RESULT_H

#include "fastlib/data/monolithic_point.h"

namespace fl {
namespace ml {
template<typename TableType>
class LinearRegressionResult {
  private:

    const TableType *query_table_;

    fl::data::MonolithicPoint<double> predictions_;

    double residual_sum_of_squares_;

  public:

    void Init(const TableType &query_table_in);

    const fl::data::MonolithicPoint<double> &predictions() const;

    fl::data::MonolithicPoint<double> &predictions();

    double residual_sum_of_squares() const;

    double &residual_sum_of_squares();
};
};
};

#endif
