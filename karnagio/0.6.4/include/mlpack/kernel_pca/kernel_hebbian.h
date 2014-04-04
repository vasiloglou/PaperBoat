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
/** @file kernel_hebbian.h
 *
 *  @brief Implements the KHA-SMD (kernel hebbian algorithm with
 *         stochastic meta descent) described in "Fast Iterative
 *         Kernel PCA" by Schraudolph, Gunter, and Vishwanathan (JMLR
 *         version).
 */

#ifndef MLPACK_KERNEL_PCA_KERNEL_HEBBIAN_H
#define MLPACK_KERNEL_PCA_KERNEL_HEBBIAN_H

#include "boost/random/normal_distribution.hpp"
#include "boost/random/lagged_fibonacci.hpp"
#include "boost/utility.hpp"
#include "fastlib/dense/matrix.h"

namespace fl {
namespace ml {

class KernelHebbian: boost::noncopyable {

  private:

    template<typename Precision_t>
    static void Symmetricize_(fl::dense::Matrix<Precision_t, false>
                              &matrix) {

      for (index_t j = 0; j < matrix.n_cols(); j++) {
        for (index_t i = j; i < matrix.n_rows(); i++) {
          matrix.set(i, j, matrix.get(i, j) + matrix.get(j, i));
        }
      }
    }

    template<typename Precision_t>
    static void LowerTriangularize_(fl::dense::Matrix<Precision_t, false>
                                    &matrix) {

      for (index_t j = 0; j < matrix.n_cols(); j++) {
        for (index_t i = 0; i < matrix.n_rows(); i++) {
          if (j > i) {
            matrix.set(i, j, 0.0);
          }
        }
      }
    }

    template<typename Precision_t>
    static void LowerTriangularOuterProduct_
    (const fl::dense::Matrix<Precision_t, true>
     &exp_coeffs_times_current_kernel_column,
     fl::dense::Matrix<Precision_t, false> &lt_outer_product) {

      lt_outer_product.Init
      (exp_coeffs_times_current_kernel_column.length(),
       exp_coeffs_times_current_kernel_column.length());
      for (index_t j = 0; j < lt_outer_product.n_cols(); j++) {
        for (index_t i = 0; i < lt_outer_product.n_rows(); i++) {
          if (i >= j) {
            lt_outer_product.set
            (i, j, exp_coeffs_times_current_kernel_column[i] *
             exp_coeffs_times_current_kernel_column[j]);
          }
          else {
            lt_outer_product.set(i, j, 0.0);
          }
        }
      }
    }

    /** @brief Computes the centered kernel column vector for the
     *         given column index.
     *
     *  @param table The table.
     *  @param kernel The kernel.
     *  @param constant_row_vector The average in each column of the
     *                             kernel matrix.
     *  @param average_kernel_value The average kernel value in the kernel
     *                              matrix.
     *  @param column_number The column index to compute.
     *  @param column_vector The computed column.
     */
    template<typename Table_t, typename Kernel_t, typename Precision_t>
    static void ComputeKernelColumn_
    (const Table_t &table, const Kernel_t &kernel,
     const fl::dense::Matrix<Precision_t, true> &constant_row_vector,
     const Precision_t &average_kernel_value,
     index_t column_number,
     fl::dense::Matrix<Precision_t, true>
     &column_vector) {

      for (index_t i = 0; i < column_vector.length(); i++) {
        column_vector[i] =
          CenteredKernelValue_(table, kernel, constant_row_vector,
                               average_kernel_value, column_number, i);
      }
    }

    static void RandomPermute_(std::vector<index_t> &permutation,
                               index_t num_points,
                               index_t iteration_number) {

      if (iteration_number == 0) {
        permutation.Init(num_points);
        for (index_t i = 0; i < permutation.size(); i++) {
          permutation[i] = i;
        }
      }
      else {
        for (index_t i = 0; i < permutation.size(); i++) {
          index_t random_index = fl::math::Random(i + 1);
          std::swap(permutation[i], permutation[random_index]);
        }
      }
    }

    template<typename Precision_t>
    static void ComputeGainVector_
    (const fl::dense::Matrix<Precision_t, true> &leading_eigenvalues,
     index_t num_points, index_t iteration_number,
     double initial_gain_rate,
     fl::dense::Matrix<Precision_t, true> &gain_vector) {

      // Compute the length of the eigenvalue vector.
      Precision_t length_eigenvalue_vector =
        fl::dense::ops::LengthEuclidean(leading_eigenvalues);

      for (index_t i = 0; i < gain_vector.length(); i++) {
        gain_vector[i] =
          (length_eigenvalue_vector / leading_eigenvalues[i]) *
          (((Precision_t) num_points) /
           ((Precision_t) iteration_number + num_points)) *
          initial_gain_rate;
      }
    }

    /** @brief Compute the current estimate of the leading eigenvalues.
     *
     *  @param decomposition_factor The current factor of the kernel matrix.
     *  @param expansion_coefficients The expansion coefficients.
     *  @param leading_eigenvalues The computed leading eigenvalues.
     */
    template<typename Precision_t>
    static void ComputeLeadingEigenvalues_
    (const fl::dense::Matrix<Precision_t, false> &decomposition_factor,
     const fl::dense::Matrix<Precision_t, false> &expansion_coefficients,
     fl::dense::Matrix<Precision_t, true> &leading_eigenvalues) {

      for (index_t i = 0; i < leading_eigenvalues.length(); i++) {

        Precision_t numerator = 0;
        Precision_t denominator = 0;

        for (index_t j = 0; j < decomposition_factor.n_cols(); j++) {
          numerator += fl::math::Sqr(decomposition_factor.get(i, j));
          denominator += fl::math::Sqr(expansion_coefficients.get(i, j));
        }

        leading_eigenvalues[i] = sqrt(numerator / denominator);
      }
    }

    /** @brief Returns the kernel value of the pairs of points.
     *
     *  @param table The table.
     *  @param kernel The kernel.
     *  @param point1_index The index of the first point in the table.
     *  @param point2_index The index of the second point in the table.
     *
     *  @return The kernel value of the pairs of points.
     */
    template<typename Table_t, typename Kernel_t>
    static typename Table_t::Dataset_t::Point_t::CalcPrecision_t
    KernelValue_(const Table_t &table, const Kernel_t &kernel,
                 const index_t &point1_index,
                 const index_t &point2_index) {

      typename Table_t::Dataset_t::Point_t point1, point2;
      table.get(point1_index, &point1);
      table.get(point2_index, &point2);

      typename Table_t::Dataset_t::Point_t::CalcPrecision_t
      squared_distance = fl::dense::ops::DistanceSqEuclidean(point1,
                         point2);
      return kernel.EvalUnnormOnSq(squared_distance);
    }

    /** @brief Returns the kernel value of the pairs of points
     *         with the centering.
     *
     *  @param table The table.
     *  @param kernel The kernel.
     *  @param constant_row_vector The average value in each column of
     *                             the kernel matrix.
     *  @param average_kernel_value The average value in the kernel matrix.
     *  @param point1_index The index of the first point in the table.
     *  @param point2_index The index of the second point in the table.
     *
     *  @return The centered kernel value of the pairs of points.
     */
    template<typename Table_t, typename Kernel_t, typename Precision_t>
    static typename Table_t::Dataset_t::Point_t::CalcPrecision_t
    CenteredKernelValue_
    (const Table_t &table,
     const Kernel_t &kernel,
     const fl::dense::Matrix<Precision_t, true> &constant_row_vector,
     const Precision_t &average_kernel_value, const index_t &point1_index,
     const index_t &point2_index) {

      typename Table_t::Dataset_t::Point_t point1, point2;
      table.get(point1_index, &point1);
      table.get(point2_index, &point2);

      Precision_t squared_distance =
        fl::dense::ops::DistanceSqEuclidean(point1, point2);
      return kernel.EvalUnnormOnSq(squared_distance) -
             constant_row_vector[point2_index] -
             constant_row_vector[point1_index] + average_kernel_value;
    }

    /** @brief Takes a given matrix and right-multiplies the
     *         centered kernel matrix.
     *
     *  @param table The table.
     *  @param kernel The kernel.
     *  @param constant_row_vector The column averages of the kernel matrix.
     *  @param average_kernel_value The average kernel value in the kernel
     *                              matrix.
     *  @param left_mult The matrix to be on the left-side of the centered
     *                   kernel matrix.
     *  @param result The computed product.
     */
    template<typename Table_t, typename Precision_t, typename Kernel_t>
    static void ApplyKernelMatrix_
    (const Table_t &table, const Kernel_t &kernel,
     const fl::dense::Matrix<Precision_t, true> &constant_row_vector,
     const Precision_t &average_kernel_value,
     const fl::dense::Matrix<Precision_t, false> &left_mult,
     fl::dense::Matrix<Precision_t, false> &result) {

      for (index_t j = 0; j < result.n_cols(); j++) {

        for (index_t i = 0; i < result.n_rows(); i++) {

          // Take the i-th row of the matrix to be multiplied to the left
          // and the j-th column of the centered kernel matrix and
          // compute the dot-product.
          Precision_t dot_product = 0;
          for (index_t k = 0; k < left_mult.n_cols(); k++) {

            // Get the k-th point, and compute the kernel value
            // and hence the dot-product.
            dot_product += left_mult.get(i, k) *
                           CenteredKernelValue_(table, kernel, constant_row_vector,
                                                average_kernel_value, k, j);
          }
          result.set(i, j, dot_product);
        }
      }
    }

    template<typename Table_t, typename Precision_t, typename Kernel_t>
    static void Precompute_
    (const Table_t &table,
     const Kernel_t &kernel,
     fl::dense::Matrix<Precision_t, true> *constant_row_vector,
     Precision_t *average_kernel_value) {

      constant_row_vector->Init(table.n_entries());
      constant_row_vector->SetZero();
      *average_kernel_value = 0;

      for (index_t i = 0; i < table.n_entries(); i++) {
        for (index_t j = 0; j < table.n_entries(); j++) {

          // Squared distance between inner and outer point, and
          // kernel value.
          Precision_t kernel_value = KernelValue_(table, kernel, i, j);

          (*constant_row_vector)[i] += kernel_value;
          (*average_kernel_value) += kernel_value;
        }
        (*constant_row_vector)[i] /= ((Precision_t) table.n_entries());
      }
      (*average_kernel_value) /=
        ((Precision_t) fl::math::Sqr(table.n_entries()));
    }

    /** @brief Compute the update coefficient matrix.
     *
     *  @param exp_coeffs_times_current_kernel_column y_t in the paper.
     *  @param column_index The currently selected column index in the
     *                      kernel matrix.
     *  @param expansion_coefficients The expansion coefficients.
     *  @param update_coeff_matrix The computed update coefficient matrix.
     */
    template<typename Precision_t>
    static void ComputeUpdateCoeffMatrix_
    (const fl::dense::Matrix<Precision_t, true>
     &exp_coeffs_times_current_kernel_column,
     const fl::dense::Matrix<Precision_t, true> &current_kernel_column,
     const index_t &column_index,
     const fl::dense::Matrix<Precision_t, false> &expansion_coefficients,
     const fl::dense::Matrix<Precision_t, false> &decomposition_factor,
     fl::dense::Matrix<Precision_t, false> &update_coeff_matrix,
     fl::dense::Matrix<Precision_t, false>
     &prod_update_coeff_matrix_and_kernel) {

      // Compute lt(y_t y_t^T) and lt(y_t, y_t^T) A_t.
      fl::dense::Matrix<Precision_t, false> lt_outer_product;
      fl::dense::Matrix<Precision_t, false>
      lt_outer_product_times_exp_coeffs;
      LowerTriangularOuterProduct_(exp_coeffs_times_current_kernel_column,
                                   lt_outer_product);
      fl::dense::ops::Mul < fl::la::Overwrite, fl::la::NoTrans,
      fl::la::NoTrans > (lt_outer_product, expansion_coefficients,
                         &update_coeff_matrix);

      // Compute lt(y_t, y_t^T) A_t K'
      fl::dense::ops::Mul < fl::la::Overwrite, fl::la::NoTrans,
      fl::la::NoTrans > (lt_outer_product, decomposition_factor,
                         &prod_update_coeff_matrix_and_kernel);

      for (index_t j = 0; j < update_coeff_matrix.n_cols(); j++) {
        for (index_t i = 0; i < update_coeff_matrix.n_rows(); i++) {
          update_coeff_matrix.set(i, j, - update_coeff_matrix.get(i, j));
          prod_update_coeff_matrix_and_kernel.set
          (i, j, - prod_update_coeff_matrix_and_kernel.get(i, j));

          if (j == column_index) {
            update_coeff_matrix.set
            (i, j, update_coeff_matrix.get(i, j) +
             exp_coeffs_times_current_kernel_column[i]);
          }
          prod_update_coeff_matrix_and_kernel.set
          (i, j, prod_update_coeff_matrix_and_kernel.get(i, j) +
           exp_coeffs_times_current_kernel_column[i] *
           current_kernel_column[j]);
        }
      }
    }

    /** @brief Update the expansion coefficients.
     *
     *  @param log_gain_vector The log-gain vector.
     *  @param gain_decay_vector The gain-decay vector.
     *  @param update_coeff_matrix The update coefficient matrix.
     *  @param expansion_coefficients The computed updated expansion
     *         coefficients.
     */
    template<typename Precision_t>
    static void UpdateExpansionCoefficient_
    (const fl::dense::Matrix<Precision_t, true> &log_gain_vector,
     const fl::dense::Matrix<Precision_t, true> &gain_decay_vector,
     const fl::dense::Matrix<Precision_t, false> &update_coeff_matrix,
     fl::dense::Matrix<Precision_t, false> &expansion_coefficients) {

      for (index_t j = 0; j < expansion_coefficients.n_cols(); j++) {
        for (index_t i = 0; i < expansion_coefficients.n_rows(); i++) {
          expansion_coefficients.set
          (i, j, expansion_coefficients.get(i, j) +
           exp(log_gain_vector[i]) * gain_decay_vector[i] *
           update_coeff_matrix.get(i, j));
        }
      }
    }

    /** @brief Update the decomposition factor (Equation 34).
     *
     *  @param log_gain_vector The log-gain vector.
     *  @param gain_decay_vector The gain-decay vector.
     *  @param prod_update_coeff_matrix_and_kernel The product between the
     *         \gamma and kernel matrix.
     *  @param decomposition_factor The decomposition factor of the kernel
     *                              matrix (which is updated).
     */
    template<typename Precision_t>
    static void UpdateDecompositionFactor_
    (const fl::dense::Matrix<Precision_t, true> &log_gain_vector,
     const fl::dense::Matrix<Precision_t, true> &gain_decay_vector,
     const fl::dense::Matrix<Precision_t, false>
     &prod_update_coeff_matrix_and_kernel,
     fl::dense::Matrix<Precision_t, false> &decomposition_factor) {

      for (index_t j = 0; j < decomposition_factor.n_cols(); j++) {
        for (index_t i = 0; i < decomposition_factor.n_rows(); i++) {
          decomposition_factor.set
          (i, j, decomposition_factor.get(i, j) +
           exp(log_gain_vector[i]) * gain_decay_vector[i] *
           prod_update_coeff_matrix_and_kernel.get(i, j));
        }
      }
    }

    /** @brief Update the log-gain vector.
     *
     *  @param prod_update_coeff_matrix_and_kernel \gamma_t K' in the paper.
     *  @param differential_parameters B_t in the paper.
     *  @param meta_gain \mu in the paper.
     *  @param log_gain_vector log-gain vector to be updated.
     */
    template<typename Precision_t>
    static void UpdateLogGain_
    (const fl::dense::Matrix<Precision_t, false>
     &prod_update_coeff_matrix_and_kernel,
     const fl::dense::Matrix<Precision_t, false>
     &differential_parameters, const double &meta_gain,
     fl::dense::Matrix<Precision_t, true> &log_gain_vector) {

      for (index_t i = 0; i < log_gain_vector.length(); i++) {
        Precision_t dot_product = 0;

        for (index_t j = 0; j < prod_update_coeff_matrix_and_kernel.n_cols();
             j++) {
          dot_product += prod_update_coeff_matrix_and_kernel.get(i, j) *
                         differential_parameters.get(i, j);
        }
        log_gain_vector[i] += meta_gain * dot_product;
      }
    }

    /** @brief Update the differential parameters.
     *
     *  @param log_gain_vector The log-gain vector.
     *  @param gain_decay_vector The gain-decay
     *  @param expansion_coefficients The expansion coefficients.
     *  @param current_kernel_column The column of the centered kernel
     *                               matrix.
     *  @param exp_coeffs_times_current_kernel_column The expansion
     *         coefficients times the current kernel column vector.
     *  @param decay_factor The decay factor.
     *  @param update_index The currently selected point index.
     *  @param differential_parameters The computed differential parameters.
     */
    template<typename Precision_t>
    static void UpdateDiffParameters_
    (const fl::dense::Matrix<Precision_t, true> &log_gain_vector,
     const fl::dense::Matrix<Precision_t, true> &gain_decay_vector,
     const fl::dense::Matrix<Precision_t, false> &expansion_coefficients,
     fl::dense::Matrix<Precision_t, true> &current_kernel_column,
     fl::dense::Matrix<Precision_t, true>
     &exp_coeffs_times_current_kernel_column,
     const double &decay_factor, const index_t &update_index,
     fl::dense::Matrix<Precision_t, false> &differential_parameters) {

      // This function implements Equation 37 in the paper.

      // Compute A_t + \xi * B_t.
      fl::dense::Matrix<Precision_t, false> shifted_matrix;
      shifted_matrix.Copy(expansion_coefficients);
      fl::dense::ops::AddExpert(decay_factor, differential_parameters,
                                &shifted_matrix);

      // Compute (A_t + \xi * B_t) * k'_{p(t)}.
      fl::dense::Matrix<Precision_t, true> shifted_matrix_times_column;
      fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>
      (shifted_matrix, current_kernel_column,
       &shifted_matrix_times_column);

      // Compute lt(y_t y_t^T) and the product lt(y_t y_t^T)(A_t + \xi B_t).
      fl::dense::Matrix<Precision_t, false> lt_outer_product;
      fl::dense::Matrix<Precision_t, false> lt_outer_product_times_shifted;
      LowerTriangularOuterProduct_(exp_coeffs_times_current_kernel_column,
                                   lt_outer_product);
      fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>
      (lt_outer_product, shifted_matrix, &lt_outer_product_times_shifted);

      // Compute lt(B_t k'_{p(t)} y_t^T + y_t k'_{p(t)}^T B_t^T).
      fl::dense::Matrix<Precision_t, false> diff_param_first_prod;
      fl::dense::Matrix<Precision_t, false> diff_param_second_prod;
      fl::dense::Matrix<Precision_t, false> current_kernel_column_alias;
      current_kernel_column_alias.Alias
      (current_kernel_column.ptr(), current_kernel_column.length(), 1);

      fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>
      (differential_parameters, current_kernel_column_alias,
       &diff_param_first_prod);
      fl::dense::Matrix<Precision_t, false>
      exp_coeffs_times_current_kernel_column_alias;
      exp_coeffs_times_current_kernel_column_alias.Alias
      (exp_coeffs_times_current_kernel_column.ptr(),
       exp_coeffs_times_current_kernel_column.length(), 1);

      fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::Trans>
      (diff_param_first_prod,
       exp_coeffs_times_current_kernel_column_alias,
       &diff_param_second_prod);
      Symmetricize_(diff_param_second_prod);
      LowerTriangularize_(diff_param_second_prod);

      // Compute \xi lt(B_t k'_{p(t)} y_t^T + y_t k'_{p(t)}^T B_t^T) A_t.
      fl::dense::Matrix<Precision_t, false> third_product;
      fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>
      (diff_param_second_prod, expansion_coefficients, &third_product);
      fl::dense::ops::ScaleExpert((Precision_t) decay_factor,
                                  &third_product);

      // Now update the differential parameters.
      for (index_t j = 0; j < differential_parameters.n_cols(); j++) {
        for (index_t i = 0; i < differential_parameters.n_rows(); i++) {
          Precision_t new_value = decay_factor *
                                  differential_parameters.get(i, j);
          Precision_t scaling = exp(log_gain_vector[i]) *
                                gain_decay_vector[i];

          Precision_t first_part = (j == update_index) ?
                                   shifted_matrix_times_column[i] : 0;
          new_value += scaling *
                       (first_part - lt_outer_product_times_shifted.get(i, j) -
                        third_product.get(i, j));

          differential_parameters.set(i, j, new_value);
        }
      }
    }

  public:

    template<typename Precision_t>
    class KernelHebbianResult: boost::noncopyable {
      public:

        /** @brief Extracted eigenvectors: each row is a set of
        *         expansion coefficients for an eigenvector in terms of
        *         points in the feature space. This is denoted as A.
         */
        fl::dense::Matrix<Precision_t, false> expansion_coefficients;

        /** @brief The centered kernel matrix is factored as:
         *         K' = (AK')^T (AK')
         */
        fl::dense::Matrix<Precision_t, false> decomposition_factor;

        void Init(index_t num_components, index_t num_points) {

          Precision_t variance = 1.0 / ((Precision_t) num_components *
                                        num_points);
          boost::lagged_fibonacci19937 engine;
          boost::normal_distribution<double> norm_dist(0.0, sqrt(variance));

          expansion_coefficients.Init(num_components, num_points);
          decomposition_factor.Init(num_components, num_points);

          // Initialize the expansion coefficients to Gaussian noise.
          for (index_t j = 0; j < num_points; j++) {
            for (index_t i = 0; i < num_components; i++) {
              expansion_coefficients.set
              (i, j, norm_dist.operator()<boost::lagged_fibonacci19937>
               (engine));
            }
          }
          decomposition_factor.SetZero();
        }
    };

    /** @brief Computes the kernel principal component analysis
     *         using the stochastic meta descent kernel hebbian
     *         algorithm.
     *
     *  @param table The table of dataset.
     *  @param kernel The kernel used for the computation.
     *  @param num_components The number of kernel principal components to
     *         extract.
     *  @param initial_gain_rate The initial gain rate, \eta_0
     *  @param meta_gain The mega gain rate, \mu
     *  @param decay_factor The decay rate, \xi
     *  @param result The computed result.
     */
    template<typename Table_t, typename Kernel_t, typename Precision_t>
    static void Compute(const Table_t &table,
                        const Kernel_t &kernel,
                        index_t num_components,
                        double initial_gain_rate,
                        double meta_gain, double decay_factor,
                        index_t max_num_passes,
                        KernelHebbianResult<Precision_t> *result) {

      // Typedef the precision.
      typedef typename Table_t::Dataset_t::Point_t::CalcPrecision_t
      Precision_t;

      // Initialize the result.
      result->Init(num_components, table.n_entries());

      // Precompute MK (one row vector that is,
      // [ \frac{1}{N} \sum K(x_i, x_1), \frac{1}{N} \sum K(x_i, x_2) ...
      // \frac{1}{N} \sum K(x_i, x_N) ]
      //
      // and MKM needed for doing centering in the
      // feature space, that is, \frac{1}{N^2} \sum_i \sum_j K(x_i, x_j)
      fl::dense::Matrix<Precision_t, true> constant_row_vector;
      Precision_t average_kernel_value;
      Precompute_(table, kernel, &constant_row_vector,
                  &average_kernel_value);

      // The gain decay vector.
      fl::dense::Matrix<Precision_t, true> gain_decay_vector;
      gain_decay_vector.Init(num_components);

      // The log-gain vector.
      fl::dense::Matrix<Precision_t, true> log_gain_vector;
      log_gain_vector.Init(num_components);
      log_gain_vector.SetAll(1.0);

      // Differential parameters.
      fl::dense::Matrix<Precision_t, false> differential_parameters;
      differential_parameters.Init(num_components, table.n_entries());
      differential_parameters.SetZero();

      // Apply the kernel matrix to the current expansion
      // coefficients to get the decomposition factor.
      ApplyKernelMatrix_(table, kernel, constant_row_vector,
                         average_kernel_value,
                         result->expansion_coefficients,
                         result->decomposition_factor);

      // Current estimate of the top leading eigenvalues.
      fl::dense::Matrix<Precision_t, true> leading_eigenvalues;
      leading_eigenvalues.Init(num_components);

      // The permutation for which is used to visit the dataset.
      std::vector<index_t> permutation;

      // The currently generated kernel column vector, the product
      // between the generated kernel column vector and the
      // expansion coefficients.
      fl::dense::Matrix<Precision_t, true> current_kernel_column;
      current_kernel_column.Init(table.n_entries());
      fl::dense::Matrix<Precision_t, true>
      exp_coeffs_times_current_kernel_column;
      exp_coeffs_times_current_kernel_column.Init(num_components);

      // Update coefficient matrix and the product between the
      // update coefficient matrix and the kernel matrix.
      fl::dense::Matrix<Precision_t, false> update_coeff_matrix;
      update_coeff_matrix.Init(num_components, table.n_entries());
      fl::dense::Matrix<Precision_t, false>
      prod_update_coeff_matrix_and_kernel;
      prod_update_coeff_matrix_and_kernel.Init(num_components,
          table.n_entries());

      // Apply the update rule until convergence.
      index_t iteration_number = table.n_entries();
      index_t num_passes = -1;
      do {

        // If we have passed through the dataset entirely, then
        // generate the next permutation.
        if (iteration_number == table.n_entries()) {
          iteration_number = 0;
          RandomPermute_(permutation, table.n_entries(), iteration_number);
          num_passes++;
        }

        // Compute the leading eigenvalues (Equation 20).
        ComputeLeadingEigenvalues_(result->decomposition_factor,
                                   result->expansion_coefficients,
                                   leading_eigenvalues);

        // Compute the gain decay vector (Equation 22).
        ComputeGainVector_(leading_eigenvalues, table.n_entries(),
                           iteration_number, initial_gain_rate,
                           gain_decay_vector);

        // Compute the current column of the kernel matrix.
        ComputeKernelColumn_(table, kernel,
                             constant_row_vector, average_kernel_value,
                             permutation[iteration_number],
                             current_kernel_column);

        // Compute the product between the expansion coefficients
        // and the current column (Equation 14).
        fl::dense::ops::Mul < fl::la::Overwrite, fl::la::NoTrans,
        fl::la::NoTrans >
        (result->expansion_coefficients, current_kernel_column,
         &exp_coeffs_times_current_kernel_column);

        // Compute the update coefficient matrix (Equation 16 and
        // Equation 33).
        ComputeUpdateCoeffMatrix_
        (exp_coeffs_times_current_kernel_column, current_kernel_column,
         permutation[iteration_number],
         result->expansion_coefficients, result->decomposition_factor,
         update_coeff_matrix, prod_update_coeff_matrix_and_kernel);

        // Update the log-gain vector (Equation 32).
        UpdateLogGain_(prod_update_coeff_matrix_and_kernel,
                       differential_parameters, meta_gain, log_gain_vector);

        // Update the differential parameters (Equation 37).
        UpdateDiffParameters_(log_gain_vector, gain_decay_vector,
                              result->expansion_coefficients,
                              current_kernel_column,
                              exp_coeffs_times_current_kernel_column,
                              decay_factor, permutation[iteration_number],
                              differential_parameters);

        // Update the decomposition factor (Equation 34).
        UpdateDecompositionFactor_
        (log_gain_vector, gain_decay_vector,
         prod_update_coeff_matrix_and_kernel,
         result->decomposition_factor);

        // Update the expansion coefficients (Equation 31).
        UpdateExpansionCoefficient_
        (log_gain_vector, gain_decay_vector, update_coeff_matrix,
         result->expansion_coefficients);

        // Increment the iteration number.
        iteration_number++;

      }
      while (num_passes < max_num_passes);
    }
};
};
};

#endif
