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

#ifndef FL_LITE_MLPACK_REGRESSION_STEPWISE_REGRESSION_DEV_H
#define FL_LITE_MLPACK_REGRESSION_STEPWISE_REGRESSION_DEV_H

#include "mlpack/regression/stepwise_regression.h"
#include "mlpack/regression/linear_regression_model_dev.h"
#include "mlpack/regression/linear_regression_result_dev.h"

namespace fl {
namespace ml {

template <bool do_forward_selection, bool do_backward_selection >
template<typename TableType, bool do_naive_least_squares>
void StepwiseRegression <do_forward_selection, do_backward_selection >::
Compute(double min_improvement_threshold,
        LinearRegressionModel<TableType, do_naive_least_squares> *model) {

  double current_best_score = model->aic_score();
  bool done_flag = false;

  std::deque<int> inactive_column_indices_copy;
  if (do_forward_selection) {
    inactive_column_indices_copy = model->active_column_indices();
    for(size_t i=0; i<inactive_column_indices_copy.size(); ++i) {
      model->MakeColumnInactive(inactive_column_indices_copy[i]);
    }
  }

  while ((!done_flag)) {
      //&& model->active_column_indices().size() >= 2) {

    double best_score_in_this_iteration = std::numeric_limits<double>::max();
    int action_column_index = -1;
    bool forward_select = true;
    fl::logger->Debug() << "Current best score: " << current_best_score;

    if (do_forward_selection) {
      inactive_column_indices_copy = model->inactive_column_indices();
      // Make a copy of the active column indices that become inactive for the forward stepwise
      for (std::deque<int>::iterator inactive_column_indices_it =
             inactive_column_indices_copy.begin();
           inactive_column_indices_it != inactive_column_indices_copy.end();
           inactive_column_indices_it++) {

        // Add a feature from the inactive set, and compute its AIC
        // score.
        LinearRegressionResult<TableType> temp_result;
        model->MakeColumnActive(*inactive_column_indices_it);
        model->Solve();
        model->Predict(*(model->table()), &temp_result);
        model->ComputeModelStatistics(temp_result);
        double new_aic_score = model->aic_score();

        fl::logger->Debug() << "Adding the feature: " <<
        model->column_name(*inactive_column_indices_it) << " would result "
        "in the AIC score of " << new_aic_score;

        if (new_aic_score < best_score_in_this_iteration) {
          best_score_in_this_iteration = new_aic_score;
          action_column_index = *inactive_column_indices_it;
          forward_select = true;
        }

        // Remove the feature again for the next iteration.
        model->MakeColumnInactive(*inactive_column_indices_it);
      }
    }

    if (do_backward_selection) {

      // Make a copy of the active column indices.
      std::deque<int> active_column_indices_copy =
        model->active_column_indices();

      for (std::deque<int>::iterator active_column_indices_it =
             active_column_indices_copy.begin();
           active_column_indices_it != active_column_indices_copy.end();
           active_column_indices_it++) {

        // Skip the intercept term.
        if (*active_column_indices_it >= model->table()->n_attributes()) {
          continue;
        }

        // Remove a feature from the active set, and compute its AIC
        // score.
        LinearRegressionResult<TableType> temp_result;
        model->MakeColumnInactive(*active_column_indices_it);
        model->Solve();
        model->Predict(*(model->table()), &temp_result);
        model->ComputeModelStatistics(temp_result);
        double new_aic_score = model->aic_score();

        fl::logger->Debug() << "Removing the feature: " <<
        model->column_name(*active_column_indices_it) << " would result "
        "in the AIC score of " << new_aic_score;

        if (new_aic_score < best_score_in_this_iteration) {
          best_score_in_this_iteration = new_aic_score;
          action_column_index = *active_column_indices_it;
          forward_select = false;
        }

        // Add the feature again for the next iteration.
        model->MakeColumnActive(*active_column_indices_it);
      }
    }

    // Only change the model if the improvement is above the
    // user-specified level.
    if (fabs(best_score_in_this_iteration - current_best_score) > min_improvement_threshold) {

      // Replace the best score.
      current_best_score = best_score_in_this_iteration;
      if (forward_select) {
        if (action_column_index<0) {
          fl::logger->Warning()<<"Failed to add a column ";
          done_flag=true;
          break;
        }
        fl::logger->Debug() << "The final decision for this iteration is "
        "to add the feature: " <<
        model->column_name(action_column_index);
        model->MakeColumnActive(action_column_index);
      }
      else {
        if (action_column_index<0) {
          fl::logger->Die()<<"Failed to remove a column";
          done_flag=true;
          break;
        }
        fl::logger->Debug() << "The final decision for this iteration is "
        "to remove the feature: " <<
        model->column_name(action_column_index);
        model->MakeColumnInactive(action_column_index);
      }
    }

    // Otherwise, we are done.
    else {
      fl::logger->Debug() << "Could not find improvement, terminating.";
      done_flag = true;
    }
  } // end of the main loop.

  // Compute the final model.
  model->Solve();

  fl::logger->Debug() << "After stepwise regression: " <<
  model->active_column_indices().size() << " attributes which may include "
  "the bias term survived.";
}
};
};

#endif
