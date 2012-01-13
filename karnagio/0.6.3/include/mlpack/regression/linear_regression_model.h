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

#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MODEL_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MODEL_H

#include "mlpack/regression/qr_least_squares_dev.h"
#include "mlpack/regression/linear_regression_result_dev.h"

namespace fl {
namespace ml {
template<typename TableType, bool do_naive_least_squares>
class LinearRegressionModel {

  private:

    double conf_prob_;

    const TableType *table_;

    QRLeastSquares<do_naive_least_squares> factorization_;

    fl::data::MonolithicPoint<double> coefficients_;

    fl::data::MonolithicPoint<double> standard_errors_;

    fl::data::MonolithicPoint<double> confidence_interval_los_;

    fl::data::MonolithicPoint<double> confidence_interval_his_;

    fl::data::MonolithicPoint<double> t_statistics_;

    fl::data::MonolithicPoint<double> p_values_;

    double adjusted_r_squared_;

    double f_statistic_;

    double r_squared_;

    double sigma_;

    double aic_score_;

  private:
    template<typename PointType>
    void ExportHelper_(
      const fl::data::MonolithicPoint<double> &source,
      PointType &destination) const;

  public:

    double aic_score() const;

    const TableType *table() const;

    std::string column_name(int column_index) const;

    void Init(const TableType &table_in,
              const std::deque<int> &initial_active_column_indices,
              int initial_right_hand_side_index,
              bool include_bias_term_in, double conf_prob_in);

    std::deque<int> &active_column_indices();

    std::deque<int> &inactive_column_indices();

    int active_right_hand_side_column_index() const;

    void set_active_right_hand_side_column_index(int right_hand_side_index_in);

    template<typename ResultType>
    void Predict(const TableType &query_table,
                 ResultType *result_out) const;

    void MakeColumnActive(int column_index);

    void MakeColumnInactive(int column_index);

    void Solve();

    double FStatistic();

    double AdjustedSquaredCorrelationCoefficient();

    void ComputeModelStatistics(
      const LinearRegressionResult<TableType> &result);

    double SquaredCorrelationCoefficient(
      const LinearRegressionResult<TableType> &result) const;

    double VarianceInflationFactor(
      const LinearRegressionResult<TableType> &result) const;

    template<typename TableType2>
    void Export(TableType2 *coefficients_table,
                TableType2 *standard_errors_table,
                TableType2 *confidence_interval_los_table,
                TableType2 *confidence_interval_his_table,
                TableType2 *t_statistics_table,
                TableType2 *p_values_table,
                TableType2 *adjusted_r_squared_table,
                TableType2 *f_statistic_table,
                TableType2 *r_squared_table,
                TableType2 *sigma_table) const;

    double ComputeAICScore(const LinearRegressionResult<TableType> &result);

    double CorrelationCoefficient(int first_attribute_index,
                                  int second_attribute_index) const;

    void ClearInactiveColumns();

    const fl::data::MonolithicPoint<double> &coefficients() const ;

    fl::data::MonolithicPoint<double> &coefficients();


};
};
};

#endif
