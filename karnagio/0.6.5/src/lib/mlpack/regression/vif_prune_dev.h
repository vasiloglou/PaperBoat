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

#ifndef FL_LITE_MLPACK_REGRESSION_VIF_PRUNE_DEV_H
#define FL_LITE_MLPACK_REGRESSION_VIF_PRUNE_DEV_H

#include "mlpack/regression/vif_prune.h"
#include "mlpack/regression/linear_regression_result_dev.h"

namespace fl {
namespace ml {
template<typename TableType, bool do_naive_least_squares>
void VifPrune::Compute(
  double vif_threshold,
  std::deque<int> &prune_column_indices,
  LinearRegressionModel<TableType, do_naive_least_squares> *model) {

  // Save the initial right hand side index before VIF selection.
  int original_active_right_hand_side_column_index =
    model->active_right_hand_side_column_index();

  // Boolean flag for outer loop termination.
  bool done_flag = false;

  // The outer loop.
  while (!done_flag && prune_column_indices.size() > 1) {

    double max_vif = 0.0;
    std::deque<int>::iterator max_vif_prune_column_indices_it =
      prune_column_indices.end();

    done_flag = true;

    // The inner loop that loops over the current list of prune column
    // indices that have survived.
    for (std::deque<int>::iterator it = prune_column_indices.begin();
         it != prune_column_indices.end() && max_vif < 100; it++) {

      // Set the current dimension to be the one that will be
      // regressed again.
      LinearRegressionResult<TableType> temp_result;
      int current_prune_column_index = *it;
      model->set_active_right_hand_side_column_index(
        current_prune_column_index);

      // Solve for the linear coefficients.
      model->Solve();

      // Do prediction.
      model->Predict(*(model->table()), &temp_result);

      // Compute the variance inflation factor of the
      // dimension being regressed again.
      double vif = model->VarianceInflationFactor(temp_result);

      fl::logger->Debug() << "Variance inflation factor for " <<
      model->column_name(current_prune_column_index) << ": " << vif;

      if (vif > max_vif) {
        max_vif = vif;
        max_vif_prune_column_indices_it = it;
      }

      // Need to put back the feature back into the active set here.
      model->set_active_right_hand_side_column_index(
        original_active_right_hand_side_column_index);
      model->MakeColumnActive(current_prune_column_index);

    } // end of the inner loop.

    // If nothing was found, then break.
    if (max_vif_prune_column_indices_it == prune_column_indices.end()) {
      break;
    }

    // If the maximum variance inflation factor exceeded the
    // threshold, then eliminate it from the current list of
    // features, and the do-while loop repeats.
    fl::logger->Debug() << "Max variance inflation factor in the current "
    "iteration was achieved by: " <<
    model->column_name(*max_vif_prune_column_indices_it) <<
    " and had vif of " << max_vif;
    if (max_vif > vif_threshold) {

      fl::logger->Debug() << "Removing " <<
      model->column_name(*max_vif_prune_column_indices_it);

      // Remove it from the active indices from the factor.
      model->MakeColumnInactive(*max_vif_prune_column_indices_it);

      // Remove it from the prune list.
      prune_column_indices.erase(max_vif_prune_column_indices_it);

      // Done flag is re-set to false.
      done_flag = false;
    }
  } // end of outer loop.

  // Set the active prediction index to the original one and compute
  // the coefficients.
  model->set_active_right_hand_side_column_index(
    original_active_right_hand_side_column_index);
  model->Solve();

  // Take out the inactive columns (pruned ones) - we do not need them
  // any more.
  model->ClearInactiveColumns();

  fl::logger->Debug() << model->active_column_indices().size() <<
  " features have survived the VIF selection.";
}
};
};

#endif
