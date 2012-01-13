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
#ifndef MLPACK_LASSO_GAUSS_SEIDEL_LASSO_DEV_H
#define MLPACK_LASSO_GAUSS_SEIDEL_LASSO_DEV_H

#include "mlpack/lasso/gauss_seidel_lasso.h"
#include "fastlib/la/linear_algebra.h"

namespace fl {
namespace ml {

template<typename TableType>
void LassoModel<TableType>::PrintCoefficients(
  const char *file_name) {

  const typename TableType::Dataset_t &dataset = *(table->data());
  const std::vector<std::string> &features = dataset.labels();
  FILE *model_output = fopen(file_name, "w+");

  // Assume that the first coefficient is the bias term.
  for (index_t i = 0; i < (index_t) coefficients.size(); i++) {
    if (i == 0) {
      fprintf(model_output, "Bias term: %g\n", coefficients[i]);
    }
    else {
      if (features.size() > 0) {
        fprintf(model_output, "%s: %g\n",
                features[predictor_indices[i - 1]].c_str(),
                coefficients[i]);
      }
      else {
        fprintf(model_output, "%d: %g\n", predictor_indices[i - 1],
                coefficients[i]);
      }
    }
  }
  fclose(model_output);
}

template<typename TableType>
template<typename TableType1>
void LassoModel<TableType>::Export(TableType1 *coeff_table) {

  // Put the intercept at the very end.
  for (int j = 0; j < static_cast<int>(coefficients.size()); j++) {
    coeff_table->set(j, 0);
  }
  coeff_table->set(coeff_table->n_entries() - 1, coefficients[0]);

  for (index_t j = 1; j < static_cast<index_t>(coefficients.size()); j++) {
    coeff_table->set(predictor_indices[j - 1], coefficients[j]);
  }
}

template<typename TableType>
template<typename Precision_t>
void GaussSeidelLasso<TableType>::ExtractNormalEquations_(
  TableType &table,
  const std::vector<index_t> &predictor_indices,
  const index_t &prediction_index,
  fl::dense::Matrix<Precision_t, false> *normal_left_hand_side,
  fl::dense::Matrix<Precision_t, true> *normal_right_hand_side) {

  normal_left_hand_side->Init(predictor_indices.size() + 1,
                              predictor_indices.size() + 1);
  normal_left_hand_side->SetZero();
  normal_right_hand_side->Init(predictor_indices.size() + 1);
  normal_right_hand_side->SetZero();

  // Loop through each point and compute the outerproduct for X^T X and
  // dot product with the prediction dimension for X^T y.
  for (index_t i = 0; i < table.n_entries(); i++) {

    typename TableType::Dataset_t::Point_t point;
    table.get(i, &point);
    for (index_t j = 0; j <= predictor_indices.size(); j++) {
      Precision_t outer_factor = (j == 0) ? 1.0 : point[predictor_indices[j - 1]];
      for (index_t k = 0; k <= predictor_indices.size(); k++) {
        Precision_t inner_factor = (k == 0) ?
                                   1.0 : point[predictor_indices[k - 1]];
        normal_left_hand_side->set(
          k, j, normal_left_hand_side->get(k, j) + inner_factor *
          outer_factor);
      }
    }
    ((*normal_right_hand_side)[0]) += point[prediction_index];
    for (index_t j = 0; j < predictor_indices.size(); j++) {
      ((*normal_right_hand_side)[j + 1]) +=
        point[predictor_indices[j]] * point[prediction_index];
    }
  }
}

template<typename TableType>
template<typename Precision_t>
void GaussSeidelLasso<TableType>::ExtractNormalEquations_(
  TableType &table,
  const index_t &prediction_index,
  fl::dense::Matrix<Precision_t, false> *normal_left_hand_side,
  fl::dense::Matrix<Precision_t, true> *normal_right_hand_side) {

  normal_left_hand_side->Init(table.n_attributes() ,
                              table.n_attributes() );
  normal_left_hand_side->SetZero();
  normal_right_hand_side->Init(table.n_attributes());
  normal_right_hand_side->SetZero();
  normal_left_hand_side->set(0,0, table.n_entries());
  // Loop through each point and compute the outerproduct for X^T X and
  // dot product with the prediction dimension for X^T y.
  for (index_t i = 0; i < table.n_entries(); ++i) {
    typename TableType::Point_t point;
    table.get(i, &point);
    for (typename TableType::Point_t::iterator it=point.begin();
        it!=point.end(); ++it) {
      if (it.attribute()!=prediction_index) {
        normal_left_hand_side->set(0,it.attribute(), 
            normal_left_hand_side->get(0, it.attribute())+it.value());
      }
    }
    for(typename TableType::Point_t::iterator it1=point.begin();
        it1!=point.end(); ++it1) {
      if (it1.attribute()!=prediction_index) {
        for(typename TableType::Point_t::iterator it2=point.begin();
          it2!=point.end(); ++it2){
          if (it2.attribute()!=prediction_index) {
            normal_left_hand_side->set(
              it2.attribute(), it1.attribute(), 
              normal_left_hand_side->get(it2.attribute(), it1.attribute()) 
              + it1.value() * it2.value());
          }
        }
      }
    }
    ((*normal_right_hand_side)[0]) += point[prediction_index];
    for (typename TableType::Point_t::iterator it=point.begin();
        it!=point.end(); ++it){
      if (it.attribute()!=prediction_index) { 
        if (it.attribute()<prediction_index) {
          ((*normal_right_hand_side)[it.attribute() + 1]) +=
          it.value() * point[prediction_index];
        } else {
           ((*normal_right_hand_side)[it.attribute()]) +=
           it.value() * point[prediction_index];     
        }
      }
    }
  }
}

template<typename TableType>
template<typename Precision_t>
void GaussSeidelLasso<TableType>::ComputeViolations_(
  const fl::dense::Matrix<Precision_t, true> &coefficients,
  const double &threshold, const double &lambda,
  const fl::dense::Matrix<Precision_t, true> &gradient,
  fl::dense::Matrix<Precision_t, true> *violations) {

  // The intercept term is treated in a special way.
  (*violations)[0] = fabs(gradient[0]);
  for (index_t i = 1; i < coefficients.length(); i++) {
    if (coefficients[i] > 0) {
      (*violations)[i] = fabs(lambda + gradient[i]);
    }
    else
      if (coefficients[i] < 0) {
        (*violations)[i] = fabs(lambda - gradient[i]);
      }
      else {
        (*violations)[i] = std::max(std::max(-gradient[i] - lambda,
                                             gradient[i] - lambda),
                                    0.0);
      }
  }
}

template<typename TableType>
template<typename Precision_t>
Precision_t GaussSeidelLasso<TableType>::MaxSlope_(
  const fl::dense::Matrix<Precision_t, true> &coefficients,
  const double &zero_threshold, const double &lambda,
  const fl::dense::Matrix<Precision_t, true> &gradient,
  const index_t &max_position) {

  // If it is the intercept term, then just return the
  // gradient along it.
  if (max_position == 0) {
    return gradient[max_position];
  }
  else {
    if (coefficients[max_position] > 0 ||
        (fabs(coefficients[max_position]) < zero_threshold &&
         lambda + gradient[max_position] < 0)) {
      return lambda + gradient[max_position];
    }
    else
      if (coefficients[max_position] < 0 ||
          (fabs(coefficients[max_position]) < zero_threshold &&
           -lambda + gradient[max_position] > 0)) {
        return -lambda + gradient[max_position];
      }
      else {
        return 0.0;
      }
  }
}

template<typename TableType>
template<typename Precision_t>
void GaussSeidelLasso<TableType>::MaxElement_(
  const fl::dense::Matrix<Precision_t, true> &violations,
  Precision_t *max_value,
  index_t *max_position) {

  *max_value = fabs(violations[0]);
  *max_position = 0;
  for (index_t i = 1; i < violations.length(); i++) {
    if (fabs(violations[i]) > *max_value) {
      *max_value = fabs(violations[i]);
      *max_position = i;
    }
  }
}

template<typename TableType>
template<bool inner_loop, typename Precision_t>
void GaussSeidelLasso<TableType>::MaxViolation_(
  const fl::dense::Matrix<Precision_t, true> &coefficients,
  const fl::dense::Matrix<Precision_t, true> &violations,
  const Precision_t &zero_threshold,
  Precision_t *max_value,
  index_t *max_position) {

  *max_value = 0;
  *max_position = 0;
  for (index_t i = 0; i < violations.length(); i++) {
    if (inner_loop) {
      if (fabs(coefficients[i]) >= zero_threshold &&
          violations[i] > *max_value) {
        *max_value = fabs(violations[i]);
        *max_position = i;
      }
    } else {
      if (fabs(coefficients[i]) < zero_threshold &&
          violations[i] > *max_value) {
        *max_value = fabs(violations[i]);
        *max_position = i;
      }
    }
  }
}

template<typename TableType>
template<typename Precision_t>
bool GaussSeidelLasso<TableType>::DifferentSigns_(
  const Precision_t &first, const Precision_t &second) {

  return !((first < 0 && second < 0) || (first > 0 && second > 0) ||
           (first == 0 && second == 0));
}

template<typename TableType>
template<typename Precision_t>
void GaussSeidelLasso<TableType>::ComputeGradient_(
  const fl::dense::Matrix<Precision_t, false> &normal_left_hand_side,
  const fl::dense::Matrix<Precision_t, true> &normal_right_hand_side,
  const fl::dense::Matrix<Precision_t, true> &coefficients,
  fl::dense::Matrix<Precision_t, true> &gradient) {

  fl::dense::ops::Mul<fl::la::Overwrite>(normal_left_hand_side,
                                         coefficients, &gradient);
  fl::dense::ops::SubFrom(normal_right_hand_side.length(),
                          normal_right_hand_side.ptr(), gradient.ptr());
}

template<typename TableType>
template<typename Precision_t>
void GaussSeidelLasso<TableType>::UpdateGradient_(
  const fl::dense::Matrix<Precision_t, false> &normal_left_hand_side,
  const index_t &column_index,
  const Precision_t &coeff_original,
  const Precision_t &coeff_new,
  fl::dense::Matrix<Precision_t, true> &gradient) {

  Precision_t change = coeff_new - coeff_original;
  for (index_t i = 0; i < gradient.length(); i++) {
    gradient[i] += normal_left_hand_side.get(i, column_index) * change;
  }
}

template<typename TableType>
template<typename Precision_t>
bool GaussSeidelLasso<TableType>::SolutionFound_(
  const fl::dense::Matrix<Precision_t, true> &violations,
  const double &optimal_tolerance) {

  bool flag = true;
  for (index_t i = 0; i < violations.length(); i++) {
    if (violations[i] > optimal_tolerance) {
      flag = false;
      break;
    }
  }
  return flag;
}

template<typename TableType>
template<typename Precision_t, bool initial_strategy_is_zero>
void GaussSeidelLasso<TableType>::Compute(
  TableType &table,
  Precision_t violation_tolerance,
  Precision_t gradient_tolerance,
  index_t iterations,
  const std::vector<index_t> &predictor_indices,
  const index_t &prediction_index,
  const double &lambda,
  fl::ml::LassoModel<TableType> *model_in) {

  // Necessary constants.
  const double zero_threshold = 0.00001;

  // const index_t max_iteration = 10000;

  // Depending on the top-down (L2) or the bottom-up (zero
  // vector) strategy, initialize the coefficients.
  fl::dense::Matrix<Precision_t, true> &model = model_in->coefficients;
  //if (initial_strategy_is_zero) {
  model.Init(predictor_indices.size() + 1);
  model.SetZero();
  //} else {
  //fl::ml::RidgeRegression<TableType>::Compute(table, predictor_indices,
  //  prediction_index, lambda, &model);
  //}

  // Precompute X^T y and X^T X with the augmented intercept 1 column.
  fl::dense::Matrix<Precision_t, false> normal_left_hand_side;
  fl::dense::Matrix<Precision_t, true> normal_right_hand_side;
  fl::logger->Message()<<"Extracting normal equations"<<std::endl;
  if (predictor_indices.size()==table.n_attributes()-1) {
     fl::logger->Message()<<"All attributes selected, using optimized version"
       <<std::endl;
     if (prediction_index!=0) {
       fl::logger->Die()<< "For the optimized version prediction index must be 0";
     }
     ExtractNormalEquations_(table, prediction_index,
                            &normal_left_hand_side,
                            &normal_right_hand_side);
  } else {
    fl::logger->Message()<<"Omitting selected attributes, using the non-optimized version"
      <<std::endl;
    ExtractNormalEquations_(table, predictor_indices, prediction_index,
                            &normal_left_hand_side,
                            &normal_right_hand_side);
  }
  fl::logger->Message()<<"Finished extracting normal equations"<<std::endl;

  // Compute the gradient.
  fl::dense::Matrix<Precision_t, true> gradient;
  gradient.Init(normal_right_hand_side.length());
  ComputeGradient_(normal_left_hand_side, normal_right_hand_side,
                   model, gradient);

  // Compute the initial violation set and the initial max
  // violating variable.
  fl::dense::Matrix<Precision_t, true> violations;
  violations.Init(model.length());
  ComputeViolations_(model, zero_threshold, lambda, gradient,
                     &violations);
  Precision_t max_violation;
  index_t max_violation_position;
  MaxElement_(violations, &max_violation, &max_violation_position);

  // The main iteration: run until convergence.
  index_t iteration = 0;
  index_t line_mins = 0;
  fl::logger->Message()<<"Starting iterations"<<std::endl;
  Precision_t old_gradient_norm=std::numeric_limits<Precision_t>::max();
  Precision_t new_gradient_norm=fl::la::LengthEuclidean(gradient);
  while (fabs(old_gradient_norm-new_gradient_norm)/old_gradient_norm > gradient_tolerance
      && iteration<iterations) {
    do {
      line_mins++;

      Precision_t max_violation_slope =
        MaxSlope_(model, zero_threshold, lambda / 2.0,
                  gradient, max_violation_position);
      Precision_t coeff_original = model[max_violation_position];
      model[max_violation_position] =
        model[max_violation_position] - max_violation_slope /
        normal_left_hand_side.get(max_violation_position,
                                  max_violation_position);
      if (coeff_original != 0 &&
          DifferentSigns_(coeff_original,
                          model[max_violation_position])) {
        model[max_violation_position] = 0.0;
      }

      // Update the gradient due to a change in one weight component and
      // the compute the maximum violator.
      UpdateGradient_(normal_left_hand_side, max_violation_position,
                      coeff_original, model[max_violation_position],
                      gradient);
      ComputeViolations_(model, zero_threshold, lambda / 2.0,
                         gradient, &violations);
      MaxViolation_<true>(
        model, violations, zero_threshold, &max_violation,
        &max_violation_position);

      if (max_violation < violation_tolerance) {
        break;
      }
    } while (true); // end of inner iteration...

    iteration++;

    MaxViolation_<false>(
      model, violations, zero_threshold, &max_violation,
      &max_violation_position);

    if (iteration % 5 == 0) {
      fl::logger->Message()<<"Iteration: "
        <<iteration 
        <<", Maximum violation: " <<max_violation
        <<std::endl;
    }
    old_gradient_norm=new_gradient_norm;
    new_gradient_norm=fl::la::LengthEuclidean(gradient);
    
  } // end of outer iteration...

  // Return the final model.
  model_in->table = &table;
  model_in->predictor_indices = predictor_indices;
}
};
};

#endif
