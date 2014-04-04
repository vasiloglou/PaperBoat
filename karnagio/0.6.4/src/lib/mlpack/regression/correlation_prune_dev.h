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

#ifndef FL_LITE_MLPACK_REGRESSION_CORRELATION_PRUNE_DEV_H
#define FL_LITE_MLPACK_REGRESSION_CORRELATION_PRUNE_DEV_H

#include "mlpack/regression/correlation_prune.h"
#include <deque>

namespace fl {
namespace ml {
template<typename TableType, bool do_naive_least_squares>
void CorrelationPrune::Compute(
  double correlation_threshold,
  std::deque<int> &prune_column_indices,
  LinearRegressionModel<TableType, do_naive_least_squares> *model) {

  std::vector<int> removed_column_indices;

  // For each predictor that is considered for pruning,
  for (int j = 0; j < prune_column_indices.size(); j++) {

    // The outer attribute index.
    int prune_predictor_index = prune_column_indices[j];

    // The maximum absolute correlation found so far.
    double max_abs_correlation = 0;

    int max_correlation_column_index = -1;

    // For each predictor in the list,
    std::deque<int> active_column_indices = model->active_column_indices();
    for (int i = 0; i < active_column_indices.size(); i++) {

      // The inner attribute index.
      int predictor_index = active_column_indices[i];

      // Skip if it is the bias term.
      if (predictor_index >= model->table()->n_attributes()) {
        continue;
      }

      // Skip if the inner and the outer index are equal.
      if (prune_predictor_index == predictor_index) {
        continue;
      }

      // The correlation coefficient.
      double abs_correlation =
        fabs(model->CorrelationCoefficient(prune_predictor_index,
                                           predictor_index));
      if (boost::math::isnan(abs_correlation) || boost::math::isinf(abs_correlation)) {
        abs_correlation = std::numeric_limits<double>::max();
      }

      fl::logger->Debug() << "Absolute value of the correlation between " <<
      model->column_name(prune_predictor_index) << " and " <<
      model->column_name(predictor_index) << " : " << abs_correlation;
      if (abs_correlation > max_abs_correlation) {
        max_correlation_column_index = predictor_index;
        max_abs_correlation = abs_correlation;
      }

    } // end of the inner loop.

    // If the maximum absolute correlation is above the threshold,
    // then remove it from the predictor list and the pruned
    // predictor list.
    if (max_abs_correlation > correlation_threshold) {
      fl::logger->Debug() << "Removing: " <<
      model->column_name(max_correlation_column_index);
      removed_column_indices.push_back(max_correlation_column_index);
      model->MakeColumnInactive(max_correlation_column_index);
    }

    // There should be at least one predictor, so if not break.
    if (model->active_column_indices().size() < 2) {
      break;
    }

  } // end of the outer loop.

  // Remove the attributes from the prune list as well after the loop.
  for (int i = 0; i < removed_column_indices.size(); i++) {
    for (std::deque<int>::iterator it = prune_column_indices.begin();
         it != prune_column_indices.end(); it++) {
      if (*it == removed_column_indices[i]) {
        prune_column_indices.erase(it);
        break;
      }
    }
  }

  // Clear the inactive indices.
  model->ClearInactiveColumns();
}
};
};

#endif
