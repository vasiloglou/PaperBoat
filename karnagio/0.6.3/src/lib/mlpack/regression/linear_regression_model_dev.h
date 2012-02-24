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

#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MODEL_DEV_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MODEL_DEV_H

#include "boost/math/distributions/students_t.hpp"
#include "mlpack/regression/linear_regression_model.h"

namespace fl {
namespace ml {

template<typename TableType, bool do_naive_least_squares>
std::deque<int> &LinearRegressionModel<TableType, do_naive_least_squares>::active_column_indices() {
  return factorization_.active_column_indices();
}

template<typename TableType, bool do_naive_least_squares>
std::deque<int> &LinearRegressionModel<TableType, do_naive_least_squares>::inactive_column_indices() {
  return factorization_.inactive_column_indices();
}

template<typename TableType, bool do_naive_least_squares>
double LinearRegressionModel<TableType, do_naive_least_squares>::aic_score() const {
  return aic_score_;
}

template<typename TableType, bool do_naive_least_squares>
const TableType *LinearRegressionModel<TableType, do_naive_least_squares>::table() const {
  return table_;
}

template<typename TableType, bool do_naive_least_squares>
std::string LinearRegressionModel<TableType, do_naive_least_squares>::column_name(
  int column_index) const {

  const std::vector<std::string> &features = table_->data()->labels();
  if(features.size() == 0) {
    std::string s;
    std::stringstream out;
    out << column_index;
    return std::string(out.str());
  }
  else {
    return features[ column_index ];
  }
}

template<typename TableType, bool do_naive_least_squares>
void LinearRegressionModel<TableType, do_naive_least_squares>::Init(
  const TableType &table_in,
  const std::deque<int> &initial_active_column_indices,
  int initial_right_hand_side_index,
  bool include_bias_term_in,
  double conf_prob_in) {

  // Set the confidence level.
  conf_prob_ = conf_prob_in;

  // Set the table pointer.
  table_ = &table_in;

  // Initialize the QR factor.
  factorization_.Init(table_in, initial_active_column_indices,
                      initial_right_hand_side_index,
                      include_bias_term_in);

  // Initialize the coefficients.
  int num_coefficients = table_in.n_attributes();
  if (include_bias_term_in) {
    num_coefficients++;
  }

  // Allocate space and initialize bunches of stuffs.
  coefficients_.Init(index_t(num_coefficients));
  coefficients_.SetZero();
  standard_errors_.Init(index_t(num_coefficients));
  standard_errors_.SetZero();
  confidence_interval_los_.Init(index_t(num_coefficients));
  confidence_interval_los_.SetZero();
  confidence_interval_his_.Init(index_t(num_coefficients));
  confidence_interval_his_.SetZero();
  t_statistics_.Init(index_t(num_coefficients));
  t_statistics_.SetZero();
  p_values_.Init(index_t(num_coefficients));
  p_values_.SetZero();

  adjusted_r_squared_ = 0;
  f_statistic_ = 0;
  r_squared_ = 0;
  sigma_ = 0;
  aic_score_ = 0;
}

template<typename TableType, bool do_naive_least_squares>
int LinearRegressionModel<TableType, do_naive_least_squares>::active_right_hand_side_column_index()
const {
  return factorization_.active_right_hand_side_column_index();
}

template<typename TableType, bool do_naive_least_squares>
void LinearRegressionModel<TableType, do_naive_least_squares>::set_active_right_hand_side_column_index(
  int right_hand_side_column_index_in) {

  // Set the right hand side column index, which will trigger the
  // QR factor update automatically.
  factorization_.set_active_right_hand_side_column_index(
    table_, right_hand_side_column_index_in);
}

template<typename TableType, bool do_naive_least_squares>
template<typename ResultType>
void LinearRegressionModel<TableType, do_naive_least_squares>::Predict(
  const TableType &query_table,
  ResultType *result_out) const {

  // Allocate space for the result.
  int prediction_index = factorization_.active_right_hand_side_column_index();
  const std::deque<int> &active_column_indices =
    factorization_.active_column_indices();
  result_out->Init(query_table);

  // Initialize the residual sum of squares.
  result_out->residual_sum_of_squares() = 0.0;

  for (int i = 0; i < query_table.n_entries(); i++) {

    typename TableType::Point_t point;
    query_table.get(i, &point);
    result_out->predictions()[i] = 0.0;
    int j = 0;
    for (std::deque<int>::const_iterator it = active_column_indices.begin();
         it != active_column_indices.end(); it++, j++) {
      int column_index = *it;
      if (column_index < point.length()) {
        result_out->predictions()[i] += coefficients_[j] *
                                        point[column_index];
      }
      else {
        result_out->predictions()[i] += coefficients_[j];
      }
    }

    // If it is leave-one-out, then we can compute the residual sum of
    // squares.
    if ((&query_table) == table_) {
      result_out->residual_sum_of_squares() +=
        fl::math::Sqr(result_out->predictions()[i] - point[prediction_index]);
    }
  }
}

template<typename TableType, bool do_naive_least_squares>
void LinearRegressionModel<TableType, do_naive_least_squares>::MakeColumnActive(int column_index) {
  factorization_.MakeColumnActive(table_, column_index);
}

template<typename TableType, bool do_naive_least_squares>
void LinearRegressionModel<TableType, do_naive_least_squares>::MakeColumnInactive(int column_index) {
  factorization_.MakeColumnInactive(table_, column_index, false);
}

template<typename TableType, bool do_naive_least_squares>
void LinearRegressionModel<TableType, do_naive_least_squares>::Solve() {
  factorization_.Solve((fl::data::MonolithicPoint<double> *) NULL,
                       &coefficients_);
}

template<typename TableType, bool do_naive_least_squares>
double LinearRegressionModel<TableType, do_naive_least_squares>::CorrelationCoefficient(
  int first_attribute_index,
  int second_attribute_index) const {

  // Compute the average of each attribute index.
  double first_attribute_average = 0;
  double second_attribute_average = 0;
  for (int i = 0; i < table_->n_entries(); i++) {
    typename TableType::Point_t point;
    table_->get(i, &point);
    first_attribute_average += point[first_attribute_index];
    second_attribute_average += point[second_attribute_index];
  }
  first_attribute_average /= ((double) table_->n_entries());
  second_attribute_average /= ((double) table_->n_entries());

  // Compute the variance of each attribute index.
  double first_attribute_variance = 0;
  double second_attribute_variance = 0;
  double covariance = 0;
  for (int i = 0; i < table_->n_entries(); i++) {
    typename TableType::Point_t point;
    table_->get(i, &point);
    covariance += (point[first_attribute_index] -
                   first_attribute_average) *
                  (point[second_attribute_index] -
                   second_attribute_average);
    first_attribute_variance +=
      fl::math::Sqr(point[first_attribute_index] -
                    first_attribute_average);
    second_attribute_variance +=
      fl::math::Sqr(point[second_attribute_index] -
                    second_attribute_average);
  }
  covariance /= ((double) table_->n_entries());
  first_attribute_variance /= ((double) table_->n_entries() - 1);
  second_attribute_variance /= ((double) table_->n_entries() - 1);

  return covariance / sqrt(first_attribute_variance *
                           second_attribute_variance);
}

template<typename TableType, bool do_naive_least_squares>
double LinearRegressionModel<TableType, do_naive_least_squares>::SquaredCorrelationCoefficient(
  const LinearRegressionResult<TableType> &result) const {

  int dimension = active_right_hand_side_column_index();

  // Compute the average of the observed values.
  double avg_observed_value = 0;

  for (int i = 0; i < table_->n_entries(); i++) {
    typename TableType::Point_t point;
    table_->get(i, &point);
    avg_observed_value += point[dimension];
  }
  avg_observed_value /= ((double) table_->n_entries());

  // Compute something proportional to the variance of the observed
  // values, and the sum of squared residuals of the predictions
  // against the observations.
  double variance = 0;
  for (int i = 0; i < table_->n_entries(); i++) {
    typename TableType::Point_t point;
    table_->get(i, &point);
    variance += math::Sqr(point[dimension] - avg_observed_value);
  }
  return (variance - result.residual_sum_of_squares()) / variance;
}

template<typename TableType, bool do_naive_least_squares>
double LinearRegressionModel<TableType, do_naive_least_squares>::VarianceInflationFactor(
  const LinearRegressionResult<TableType> &result) const {

  double denominator = 1.0 - SquaredCorrelationCoefficient(result);

  if (!boost::math::isnan(denominator)) {
    if (fabs(denominator) >= 1e-2) {
      return 1.0 / denominator;
    }
    else {
      return 100;
    }
  }
  else {
    return 0.0;
  }
}

template<typename TableType, bool do_naive_least_squares>
double LinearRegressionModel<TableType, do_naive_least_squares>::FStatistic() {
  double numerator = r_squared_ /
                     ((double) factorization_.active_column_indices().size() - 1);
  double denominator = (1.0 - r_squared_) /
                       ((double) table_->n_entries() -
                        factorization_.active_column_indices().size());
  return numerator / denominator;
}

template<typename TableType, bool do_naive_least_squares>
double LinearRegressionModel<TableType, do_naive_least_squares>::
AdjustedSquaredCorrelationCoefficient() {
  int num_points = table_->n_entries();
  int num_coefficients = factorization_.active_column_indices().size();
  double factor = (((double) num_points - 1)) /
                  ((double)(num_points - num_coefficients));
  return 1.0 - (1.0 - r_squared_) * factor;
}

template<typename TableType, bool do_naive_least_squares>
void LinearRegressionModel<TableType, do_naive_least_squares>::ComputeModelStatistics(
  const LinearRegressionResult<TableType> &result) {

  // Declare the student t-distribution and find out the
  // appropriate quantile for the confidence interval
  // (currently hardcoded to 90 % centered confidence).
  boost::math::students_t_distribution<double> distribution(
    factorization_.active_column_indices().size());

  double t_score = quantile(distribution, 0.5 + 0.5 * conf_prob_);

  double variance = result.residual_sum_of_squares() /
                    (table_->n_entries() -
                     factorization_.active_column_indices().size());

  // Store the computed standard deviation of the predictions.
  sigma_ = sqrt(variance);

  fl::data::MonolithicPoint<double> dummy_vector;
  dummy_vector.Init(index_t(factorization_.active_column_indices().size()));
  dummy_vector.SetZero();
  fl::data::MonolithicPoint<double> first_vector;
  first_vector.Init(index_t(factorization_.n_rows()));
  first_vector.SetZero();
  fl::data::MonolithicPoint<double> second_vector;
  second_vector.Init(index_t(factorization_.active_column_indices().size()));
  second_vector.SetZero();

  for (int i = 0; i < factorization_.active_column_indices().size(); i++) {
    dummy_vector[i] = 1.0;
    if (i > 0) {
      dummy_vector[i - 1] = 0.0;
    }
    factorization_.TransposeSolve(dummy_vector, &first_vector);
    factorization_.Solve(&first_vector, &second_vector);
    standard_errors_[i] = sqrt(variance * second_vector[i]);
    confidence_interval_los_[i] =
      coefficients_[i] - t_score * standard_errors_[i];
    confidence_interval_his_[i] =
      coefficients_[i] + t_score * standard_errors_[i];

    // Compute t-statistics.
    t_statistics_[i] = coefficients_[i] / standard_errors_[i];

    // Compute p-values.
    // Here we take the absolute value of the t-statistics since
    // we want to push all p-values toward the right end.
    double min_t_statistic = std::min(t_statistics_[i],
                                      -(t_statistics_[i]));
    double max_t_statistic = std::max(t_statistics_[i],
                                      -(t_statistics_[i]));
    p_values_[i] =
      1.0 - (cdf(distribution, max_t_statistic) -
             cdf(distribution, min_t_statistic));
  }

  // Compute the r-squared coefficients (normal and adjusted).
  r_squared_ = SquaredCorrelationCoefficient(result);
  adjusted_r_squared_ = AdjustedSquaredCorrelationCoefficient();

  // Compute the f-statistic between the final refined model and
  // the null model, i.e. the model with all zero coefficients.
  f_statistic_ = FStatistic();

  // Compute the AIC score.
  aic_score_ = ComputeAICScore(result);
}

template<typename TableType, bool do_naive_least_squares>
double LinearRegressionModel<TableType, do_naive_least_squares>::ComputeAICScore(
  const LinearRegressionResult<TableType> &result) {

  // Compute the squared errors from the predictions.
  double aic_score = result.residual_sum_of_squares();
  aic_score /= ((double) table_->n_entries());
  aic_score = log(aic_score);
  aic_score *= ((double) table_->n_entries());
  aic_score += (2 * factorization_.active_column_indices().size());
  return aic_score;
}

template<typename TableType, bool do_naive_least_squares>
template<typename PointType>
void LinearRegressionModel<TableType, do_naive_least_squares>::ExportHelper_(
  const fl::data::MonolithicPoint<double> &source,
  PointType &destination) const {

  for (int i = 0; i < destination.size(); i++) {
    destination.set(i, 0.0);
  }

  // Put the bias term at the position of the prediction index, if
  // available. Remember that the last coefficient is for the bias
  // term, if available.
  if (table_->n_attributes() + 1 == source.length()) {
    destination.set(
      factorization_.active_right_hand_side_column_index(),
      source[ factorization_.active_column_indices().size() - 1]);
  }

  // Copy the rest.
  int j = 0;
  for (std::deque<int>::const_iterator active_it =
         factorization_.active_column_indices().begin();
       active_it != factorization_.active_column_indices().end();
       active_it++, j++) {

    // Skip the bias term.
    if (*active_it < table_->n_attributes()) {
      destination.set(*active_it, source[j]);
    }
  }
}

template<typename TableType, bool do_naive_least_squares>
void LinearRegressionModel<TableType, do_naive_least_squares>::ClearInactiveColumns() {
  factorization_.ClearInactiveColumns();
}

template<typename TableType, bool do_naive_least_squares>
template<typename TableType2>
void LinearRegressionModel<TableType, do_naive_least_squares>::Export(
  TableType2 *coefficients_table,
  TableType2 *standard_errors_table,
  TableType2 *confidence_interval_los_table,
  TableType2 *confidence_interval_his_table,
  TableType2 *t_statistics_table,
  TableType2 *p_values_table,
  TableType2 *adjusted_r_squared_table,
  TableType2 *f_statistic_table,
  TableType2 *r_squared_table,
  TableType2 *sigma_table) const {

  // Create the table for dumping the coefficients.
  if (coefficients_table!=NULL) {
    ExportHelper_(coefficients_, *coefficients_table);
  }
  // Create the table for dumping the standard errors.
  if (standard_errors_table!=NULL) {
    ExportHelper_(standard_errors_, *standard_errors_table);
  }

  // Create the tables for dumping the confidence intervals.
  if (confidence_interval_los_table!=NULL) {
    ExportHelper_(confidence_interval_los_, *confidence_interval_los_table);
  }

  if (confidence_interval_his_table!=NULL) {
    ExportHelper_(confidence_interval_his_, *confidence_interval_his_table);
  }
  // Create the table for dumping the t-statistic values.
  if (t_statistics_table!=NULL) {
    ExportHelper_(t_statistics_, *t_statistics_table);
  }
  // Create the table for p-values.
  if (p_values_table!=NULL) {
    ExportHelper_(p_values_, *p_values_table);
  }
  // Create the table for adjusted r-square statistic.
  if (adjusted_r_squared_table!=NULL) {
    adjusted_r_squared_table->set(0, adjusted_r_squared_);
  }
  if (f_statistic_table!=NULL) {
    f_statistic_table->set(0, f_statistic_);
  }
  if (r_squared_table!=NULL) {
    r_squared_table->set(0, r_squared_);
  }
  if (sigma_table!=NULL) {
    sigma_table->set(0, sigma_);
  }
}

template<typename TableType, bool do_naive_least_squares>
const fl::data::MonolithicPoint<double> &LinearRegressionModel<TableType, do_naive_least_squares>::coefficients() const {
  return coefficients_;
}

template<typename TableType, bool do_naive_least_squares>
fl::data::MonolithicPoint<double> &LinearRegressionModel<TableType, do_naive_least_squares>::coefficients() {
  return coefficients_;
}

};
};

#endif
