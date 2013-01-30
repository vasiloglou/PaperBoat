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
/** @file bpp_nnls_nmf_dev.h
 *
 *  Implements the following algorithm:
 *    @conference{kim2008tfn,
 *      title={{Toward Faster Nonnegative Matrix Factorization: A New Algorithm
*               and Comparisons}},
 *      author={Kim, J. and Park, H.},
 *      booktitle={Proceedings of the 2008 Eighth IEEE International Conference
 *                 on Data Mining},
 *      pages={353--362},
 *      year={2008},
 *      organization={IEEE Computer Society Washington, DC, USA}
 *    }
 */

#ifndef FL_LITE_MLPACK_NMF_BPP_NNLS_NMF_DEV_H
#define FL_LITE_MLPACK_NMF_BPP_NNLS_NMF_DEV_H

#include "mlpack/nmf/bpp_nnls_nmf.h"

namespace fl {
namespace ml {

template<typename Table_t>
template<typename Precision_t>
void BppNnlsNmf<Table_t>::ExtractSubVectorHelper_(
  const fl::dense::Matrix<Precision_t, false> &matrix,
  const fl::dense::Matrix<index_t, true> &num_active_primal_variables,
  const fl::dense::Matrix<bool, false> &primal_active_set,
  const index_t &column_to_solve,
  const bool index_row_by_active_vars,
  fl::dense::Matrix<Precision_t, true> *submatrix) {

  index_t num_rows = (index_row_by_active_vars) ?
                     num_active_primal_variables[column_to_solve] :
                     (primal_active_set.n_rows() -
                      num_active_primal_variables[column_to_solve]);

  // This happens if all the variables in the current column are active
  // or inactive.
  if (num_rows == 0) {
    return;
  }

  submatrix->Init(num_rows);

  index_t row_index = 0;

  // Loop each variable and determine which variable indexes the row.
  for (index_t i = 0; i < primal_active_set.n_rows() &&
       row_index < num_rows; i++) {

    // Skip the row if it is not the index we want.
    if (primal_active_set.get(i, column_to_solve) !=
        index_row_by_active_vars) {
      continue;
    }
    (*submatrix)[row_index] = matrix.get(i, column_to_solve);
    row_index++;
  }
}

template<typename Table_t>
template<typename Precision_t>
void BppNnlsNmf<Table_t>::ExtractSubMatrixHelper_(
  const fl::dense::Matrix<Precision_t, false> &matrix,
  const fl::dense::Matrix<index_t, true> &num_active_primal_variables,
  const fl::dense::Matrix<bool, false> &primal_active_set,
  const index_t &column_to_solve,
  const bool index_row_by_active_vars,
  const bool index_col_by_active_vars,
  fl::dense::Matrix<Precision_t, false> *submatrix) {

  index_t num_rows = (index_row_by_active_vars) ?
                     num_active_primal_variables[column_to_solve] :
                     (primal_active_set.n_rows() -
                      num_active_primal_variables[column_to_solve]);
  index_t num_cols = (index_col_by_active_vars) ?
                     num_active_primal_variables[column_to_solve] :
                     (primal_active_set.n_rows() -
                      num_active_primal_variables[column_to_solve]);

  // This happens if all the variables in the current column are active
  // or inactive.
  if (num_rows == 0 || num_cols == 0) {
    return;
  }

  submatrix->Init(num_rows, num_cols);

  index_t row_index = 0;

  // Loop each variable and determine which variable indexes the row.
  for (index_t i = 0; i < primal_active_set.n_rows() &&
       row_index < num_rows; i++) {

    // Skip the row if it is not the index we want.
    if (primal_active_set.get(i, column_to_solve) !=
        index_row_by_active_vars) {
      continue;
    }

    index_t col_index = 0;

    // Loop each variable and determine which variable indexes the
    // column.
    for (index_t j = 0; j < primal_active_set.n_rows() &&
         col_index < num_cols; j++) {

      if (primal_active_set.get(j, column_to_solve) !=
          index_col_by_active_vars) {
        continue;
      }
      submatrix->set(row_index, col_index, matrix.get(i, j));
      col_index++;
    }
    row_index++;
  }
}

template<typename Table_t>
template<typename Precision_t>
void BppNnlsNmf<Table_t>::ExtractSubMatrices_(
  const fl::dense::Matrix<Precision_t, false> &normal_left_hand_side,
  const fl::dense::Matrix<Precision_t, false> &normal_right_hand_side,
  const fl::dense::Matrix<index_t, true> &num_active_primal_variables,
  const fl::dense::Matrix<bool, false> &primal_active_set,
  const index_t &column_to_solve,
  fl::dense::Matrix<Precision_t, false> *primal_left_submatrix,
  fl::dense::Matrix<Precision_t, true> *primal_right_submatrix,
  fl::dense::Matrix<Precision_t, false> *dual_left_submatrix,
  fl::dense::Matrix<Precision_t, true> *dual_right_submatrix) {

  // Extract C_F^T C_F.
  ExtractSubMatrixHelper_(normal_left_hand_side,
                          num_active_primal_variables,
                          primal_active_set,
                          column_to_solve, true, true,
                          primal_left_submatrix);

  // Extract C_F^T b.
  ExtractSubVectorHelper_(normal_right_hand_side,
                          num_active_primal_variables,
                          primal_active_set,
                          column_to_solve, true,
                          primal_right_submatrix);

  // Extract C_G^T C_F.
  ExtractSubMatrixHelper_(normal_left_hand_side,
                          num_active_primal_variables,
                          primal_active_set,
                          column_to_solve, false, true,
                          dual_left_submatrix);

  // Extract C_G^T b.
  ExtractSubVectorHelper_(normal_right_hand_side,
                          num_active_primal_variables,
                          primal_active_set,
                          column_to_solve, false,
                          dual_right_submatrix);
}

template<typename Table_t>
template<typename Precision_t>
bool BppNnlsNmf<Table_t>::Feasible_(
  const fl::dense::Matrix<bool, false> &primal_active_set,
  const fl::dense::Matrix<Precision_t, false> &primal_variables,
  const fl::dense::Matrix<Precision_t, false> &dual_variables,
  std::vector< std::vector<index_t> > &primal_violations,
  std::vector< std::vector<index_t> > &dual_violations) {

  // Clear the set of violation vectors.
  for (index_t i = 0; i < (index_t) primal_violations.size(); i++) {
    primal_violations[i].clear();
    dual_violations[i].clear();
  }

  // Check each column whether it is feasible or not.
  bool is_feasible = true;

  // For each column,
  for (index_t i = 0; i < primal_active_set.n_cols(); i++) {

    // For each variable,
    for (index_t j = 0; j < primal_variables.n_rows(); j++) {

      // If the variable is in the primary active set and is
      // nonnegative, then add the blocking variable to the primary
      // violation set.
      // If the variable is in the dual active set, and is nonnegative,
      // then add the blocking variable to the dual violation set.
      if (primal_active_set.get(j, i)) {
        if (primal_variables.get(j, i) < 0) {
          is_feasible = false;
          primal_violations[i].push_back(j);
        }
      }
      else {
        if (dual_variables.get(j, i) < 0) {
          is_feasible = false;
          dual_violations[i].push_back(j);
        }
      }
    } // end of looping over each variable.
  } // end of looping over each column.

  return is_feasible;
}

template<typename Table_t>
template<typename Precision_t>
void BppNnlsNmf<Table_t>::UpdateDualVariables_(
  const fl::dense::Matrix<Precision_t, false> &dual_left_submatrix,
  const fl::dense::Matrix<Precision_t, true> &dual_right_submatrix,
  const fl::dense::Matrix<bool, false> &primal_active_set,
  const index_t &column_to_solve,
  const fl::dense::Matrix<Precision_t, false> &primal_variables,
  fl::dense::Matrix<Precision_t, false> &dual_variables) {

  index_t row_index = 0;
  for (index_t i = 0; i < dual_variables.n_rows(); i++) {
    if (! primal_active_set.get(i, column_to_solve)) {
      dual_variables.set(i, column_to_solve,
                         -dual_right_submatrix[row_index]);

      // Compute the dot product between the current i-th row of
      // C_G^T C_F and the primal variables x_F.
      Precision_t dot_product = 0;
      index_t column_index = 0;
      for (index_t j = 0; j < primal_variables.n_rows(); j++) {
        if (primal_active_set.get(j, column_to_solve)) {
          dot_product += dual_left_submatrix.get(row_index,
                                                 column_index) *
                         primal_variables.get(j, column_to_solve);
          column_index++;
        }
      }
      dual_variables.set(i, column_to_solve,
                         dual_variables.get(i, column_to_solve) +
                         dot_product);
      row_index++;
    }
  }
}

template<typename Table_t>
template<typename Precision_t>
void BppNnlsNmf<Table_t>::SolveNormalEquation_(
  const fl::dense::Matrix<Precision_t, false> &left_hand_side,
  const fl::dense::Matrix<Precision_t, true> &right_hand_side,
  const fl::dense::Matrix<bool, false> &primal_active_set,
  const index_t &column_to_solve,
  fl::dense::Matrix<Precision_t, false> &primal_variables) {

  // The SVD factors: it might be able to share the SVD factors among
  // normal equaions that are sequentially solved. We use only the
  // left singular vectors because the matrix is symmetric.
  fl::dense::Matrix<Precision_t, false> left_singular_vectors;
  fl::dense::Matrix<Precision_t, true> singular_values;
  fl::dense::Matrix<Precision_t, false>
  right_singular_vectors_transposed;
  success_t success_flag;
  fl::dense::ops::SVD<fl::la::Init>(left_hand_side, &singular_values,
                                    &left_singular_vectors,
                                    &right_singular_vectors_transposed,
                                    &success_flag);

  // Initialize the row index that corresponds to the active variable
  // to zero.
  for (index_t i = 0; i < primal_active_set.n_rows(); i++) {
    if (primal_active_set.get(i, column_to_solve)) {
      primal_variables.set(i, column_to_solve, 0.0);
    }
  }

  for (index_t j = 0; j < singular_values.length() &&
       singular_values[j] > 1e-5; j++) {

    // First compute the dot product between each left singular
    // vector and the right hand side.
    Precision_t dot_product = 0;
    index_t row_index = 0;
    for (index_t i = 0; i < primal_active_set.n_rows(); i++) {
      if (primal_active_set.get(i, column_to_solve)) {
        dot_product += left_singular_vectors.get(row_index, j) *
                       right_hand_side[row_index];
        row_index++;
      }
    }
    row_index = 0;
    for (index_t i = 0; i < primal_active_set.n_rows(); i++) {
      if (primal_active_set.get(i, column_to_solve)) {
        primal_variables.set(i, column_to_solve,
                             primal_variables.get(i, column_to_solve) +
                             dot_product / singular_values[j] *
                             left_singular_vectors.get(row_index, j));
        row_index++;
      }
    }
  }
}

template<typename Table_t>
template<typename Precision_t>
void BppNnlsNmf<Table_t>::ComputeVariables_(
  const fl::dense::Matrix<Precision_t, false> &normal_left_hand_side,
  const fl::dense::Matrix<Precision_t, false> &normal_right_hand_side,
  const fl::dense::Matrix<index_t, true> &num_active_primal_variables,
  const fl::dense::Matrix<bool, false> &primal_active_set,
  fl::dense::Matrix<Precision_t, false> &primal_variables,
  fl::dense::Matrix<Precision_t, false> &dual_variables) {

  // Right now, the problems are solved sequentially. I will add in
  // the tree amortized solving later with the incremental Cholesky/SVD
  // updating.
  for (index_t column_to_solve = 0;
       column_to_solve < primal_variables.n_cols(); column_to_solve++) {

    // Intermediate matrices extracted out of the globally precomputed
    // matrices.
    fl::dense::Matrix<Precision_t, false> primal_left_submatrix;
    fl::dense::Matrix<Precision_t, true> primal_right_submatrix;
    fl::dense::Matrix<Precision_t, false> dual_left_submatrix;
    fl::dense::Matrix<Precision_t, true> dual_right_submatrix;

    ExtractSubMatrices_(normal_left_hand_side, normal_right_hand_side,
                        num_active_primal_variables, primal_active_set,
                        column_to_solve, &primal_left_submatrix,
                        &primal_right_submatrix, &dual_left_submatrix,
                        &dual_right_submatrix);

    // Solve the normal equation by SVD for the primal variables.
    if (num_active_primal_variables[column_to_solve] > 0) {
      SolveNormalEquation_(primal_left_submatrix,
                           primal_right_submatrix, primal_active_set,
                           column_to_solve, primal_variables);
    }

    // Update the dual variables from the primal variables.
    if (num_active_primal_variables[column_to_solve] <
        primal_active_set.n_rows() &&
        num_active_primal_variables[column_to_solve] > 0) {
      UpdateDualVariables_(dual_left_submatrix, dual_right_submatrix,
                           primal_active_set, column_to_solve,
                           primal_variables, dual_variables);
    }
  }
}

template<typename Table_t>
void BppNnlsNmf<Table_t>::ApplyExchangeRules_(
  const std::vector< std::vector<index_t> > &primal_violations,
  const std::vector< std::vector<index_t> > &dual_violations,
  fl::dense::Matrix<index_t, true> &num_block_exchange_rules,
  fl::dense::Matrix<index_t, true> &backup_exchange_rules,
  fl::dense::Matrix<index_t, true> &num_active_primal_variables,
  fl::dense::Matrix<bool, false> &primal_active_set) {

  for (index_t j = 0; j < primal_active_set.n_cols(); j++) {

    // Primal violations and dual violations are disjoint, so
    // we can do this.
    index_t union_size = primal_violations[j].size() +
                         dual_violations[j].size();

    // Flag that tells whether violation sets should be exchanged
    // in entirety.
    bool exchange_all = true;

    if (union_size < backup_exchange_rules[j]) {
      num_block_exchange_rules[j] = 3;
      backup_exchange_rules[j] = union_size;
    }
    else {
      if (num_block_exchange_rules[j] > 0) {
        num_block_exchange_rules[j]--;
      }
      else {

        // In this case, choose the largest index from the union
        // of the two violation index sets and flip its boolean
        // flag.
        exchange_all = false;
      }
    }

    // Exchange the sets.
    if (exchange_all) {

      // Move from the primal to the dual.
      for (index_t i = 0; i < (index_t) primal_violations[j].size();
           i++) {
        primal_active_set.set(primal_violations[j][i], j, false);
        num_active_primal_variables[j]--;
      }

      // Move from the dual to the primal.
      for (index_t i = 0; i < (index_t) dual_violations[j].size();
           i++) {
        primal_active_set.set(dual_violations[j][i], j, true);
        num_active_primal_variables[j]++;
      }
    }
    else {
      index_t max_index = -1;
      if (primal_violations[j].size() > 0) {
        max_index = primal_violations[j][primal_violations[j].size() -
                                         1];
      }
      if (dual_violations[j].size() > 0) {
        max_index = std::max(max_index,
                             dual_violations[j]
                             [dual_violations[j].size() - 1]);
      }

      // Flip the boolean flag for the max index.
      if (primal_active_set.get(max_index, j)) {
        primal_active_set.set(max_index, j, false);
        num_active_primal_variables[j]--;
      }
      else {
        primal_active_set.set(max_index, j, true);
        num_active_primal_variables[j]++;
      }
    }
  }
}

template<typename Table_t>
template<typename Precision_t, bool transpose_mode>
void BppNnlsNmf<Table_t>::BlockPrincipalPivotingNNLS_(
  const fl::dense::Matrix<Precision_t, false> &matrix,
  const fl::dense::Matrix<Precision_t, false> &right_hand_side,
  fl::dense::Matrix<Precision_t, false> &solution,
  fl::dense::Matrix<Precision_t, false> &normal_left_hand_side) {

  // Form the factors needed for the normal equation.
  fl::dense::Matrix<Precision_t, false> normal_right_hand_side;

  if (! transpose_mode) {
    fl::dense::ops::Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>
    (matrix, matrix, &normal_left_hand_side);
    fl::dense::ops::Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>
    (matrix, right_hand_side, &normal_right_hand_side);
  }
  else {
    fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::Trans>
    (matrix, matrix, &normal_left_hand_side);
    fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::Trans>
    (matrix, right_hand_side, &normal_right_hand_side);
  }

  // Matrix for maintaining the active variables:
  // primal_active_set: represents F and G in Algorithm 2.
  fl::dense::Matrix<bool, false> primal_active_set;
  fl::dense::Matrix<index_t, true> num_active_primal_variables;

  if (! transpose_mode) {
    primal_active_set.Init(solution.n_rows(), solution.n_cols());
  }
  else {
    primal_active_set.Init(solution.n_cols(), solution.n_rows());
  }
  num_active_primal_variables.Init(primal_active_set.n_cols());
  num_active_primal_variables.SetZero();
  primal_active_set.SetAll(false);

  // The primal variables and the dual variables: the dual
  // variables are set to the negative of the normal right hand
  // side: these correspond to X and Y in Algorithm 2.
  fl::dense::Matrix<Precision_t, false> primal_variables;
  fl::dense::Matrix<Precision_t, false> dual_variables;
  dual_variables.Copy(normal_right_hand_side);
  primal_variables.Init(dual_variables.n_rows(),
                        dual_variables.n_cols());
  primal_variables.SetZero();
  fl::dense::ops::ScaleExpert((Precision_t) - 1.0, &dual_variables);

  // Stuffs related to exchange rules.
  // num_block_exchange_rules: P in Algorithm 2.
  // backup_exchange_rules: T in Algorithm 2.
  fl::dense::Matrix<index_t, true> num_block_exchange_rules;
  fl::dense::Matrix<index_t, true> backup_exchange_rules;
  if (! transpose_mode) {
    num_block_exchange_rules.Init(solution.n_cols());
    backup_exchange_rules.Init(solution.n_cols());
    backup_exchange_rules.SetAll(solution.n_rows() + 1);
  }
  else {
    num_block_exchange_rules.Init(solution.n_rows());
    backup_exchange_rules.Init(solution.n_rows());
    backup_exchange_rules.SetAll(solution.n_cols() + 1);
  }
  num_block_exchange_rules.SetAll(3);

  // Find the set of indices that are violated for both the primal
  // and the dual variables.
  std::vector< std::vector<index_t> > primal_violations;
  std::vector< std::vector<index_t> > dual_violations;
  if (! transpose_mode) {
    primal_violations.resize(solution.n_cols());
    dual_violations.resize(solution.n_cols());
  }
  else {
    primal_violations.resize(solution.n_rows());
    dual_violations.resize(solution.n_rows());
  }

  // Main loop of the algorithm: loop while the solution is
  // infeasible.
  while (! Feasible_(primal_active_set,
                     primal_variables,
                     dual_variables, primal_violations,
                     dual_violations)) {

    // Apply the exchange rule.
    ApplyExchangeRules_(primal_violations, dual_violations,
                        num_block_exchange_rules, backup_exchange_rules,
                        num_active_primal_variables, primal_active_set);

    // Update solutions.
    ComputeVariables_(normal_left_hand_side, normal_right_hand_side,
                      num_active_primal_variables, primal_active_set,
                      primal_variables, dual_variables);
  }

  // Copy over the solution.
  if (! transpose_mode) {
    for (index_t j = 0; j < solution.n_cols(); j++) {
      for (index_t i = 0; i < solution.n_rows(); i++) {
        if (primal_active_set.get(i, j)) {
          solution.set(i, j, primal_variables.get(i, j));
        }
        else {
          solution.set(i, j, 0.0);
        }
      }
    }
  }
  else {
    for (index_t j = 0; j < solution.n_cols(); j++) {
      for (index_t i = 0; i < solution.n_rows(); i++) {
        if (primal_active_set.get(j, i)) {
          solution.set(i, j, primal_variables.get(j, i));
        }
        else {
          solution.set(i, j, 0.0);
        }
      }
    }
  }
}

template<typename Table_t>
template<typename Precision_t>
Precision_t BppNnlsNmf<Table_t>::KktResidual_(
  const fl::dense::Matrix<Precision_t, false> &input_matrix,
  const fl::dense::Matrix<Precision_t, false> &w_factor,
  const fl::dense::Matrix<Precision_t, false> &h_factor,
  const fl::dense::Matrix<Precision_t, false> &w_transposed_times_w,
  const fl::dense::Matrix<Precision_t, false> &h_times_h_transposed) {

  // Accumulated KKT residual.
  Precision_t kkt_residual = 0;

  // Check whether each of the elements in WHH^T - A H^T is
  // non-negative.
  // Gradient of the object function with respect to W. Also check
  // the component-wise multiplication between W and
  // WHH^T - A H^T is zero (complementary slack-ness). This is done
  // by computing the appropriate KKT residual.
  for (index_t j = 0; j < h_factor.n_rows(); j++) {
    for (index_t i = 0; i < w_factor.n_rows(); i++) {

      // Take the dot product between the j-th column of HH^T and
      // the i-th row of W.
      Precision_t first_dot_product = 0;
      for (index_t k = 0; k < w_factor.n_cols(); k++) {
        first_dot_product += w_factor.get(i, k) *
                             h_times_h_transposed.get(k, j);
      }

      // Take the dot product between the j-th column of H^T (i.e.
      // j-th row of H) and the i-th row of A.
      Precision_t second_dot_product = 0;
      for (index_t k = 0; k < input_matrix.n_cols(); k++) {
        second_dot_product += input_matrix.get(i, k) *
                              h_factor.get(j, k);
      }
      kkt_residual += fabs(std::min(first_dot_product -
                                    second_dot_product,
                                    w_factor.get(i, j)));
    }
  }

  // Check whether each of the elements in W^T WH - W^T A is
  // non-negative. Gradient of the object function with
  // respect to H.
  for (index_t j = 0; j < input_matrix.n_cols(); j++) {
    for (index_t i = 0; i < w_factor.n_cols(); i++) {

      // Take the dot product between the j-th column of H and
      // the i-th row of W^T W.
      Precision_t first_dot_product = 0;
      for (index_t k = 0; k < h_factor.n_rows(); k++) {
        first_dot_product += w_transposed_times_w.get(i, k) *
                             h_factor.get(k, j);
      }

      // Take the dot product between the j-th column of A and
      // the i-th row of W^T (i.e. i-th column of W).
      Precision_t second_dot_product = 0;
      for (index_t k = 0; k < input_matrix.n_rows(); k++) {
        second_dot_product += w_factor.get(k, i) *
                              input_matrix.get(k, j);
      }
      kkt_residual += fabs(std::min(first_dot_product -
                                    second_dot_product,
                                    h_factor.get(i, j)));
    }
  }

  // Count how many entries and W and H are non-zeros.
  index_t w_factor_nonzero_count = 0;
  index_t h_factor_nonzero_count = 0;
  for (index_t j = 0; j < w_factor.n_cols(); j++) {
    for (index_t i = 0; i < w_factor.n_rows(); i++) {
      if (w_factor.get(i, j) > 0) {
        w_factor_nonzero_count++;
      }
    }
  }
  for (index_t j = 0; j < h_factor.n_cols(); j++) {
    for (index_t i = 0; i < h_factor.n_rows(); i++) {
      if (h_factor.get(i, j) > 0) {
        h_factor_nonzero_count++;
      }
    }
  }

  // Normalize the KKT residual by the nonzero count.
  kkt_residual = kkt_residual /
                 ((Precision_t)(w_factor_nonzero_count +
                                h_factor_nonzero_count));
  return kkt_residual;
}

template<typename Table_t>
template<typename Precision_t>
void BppNnlsNmf<Table_t>::Normalize_(
  fl::dense::Matrix<Precision_t, false> &w_factor,
  fl::dense::Matrix<Precision_t, false> &h_factor) {

  for (index_t j = 0; j < w_factor.n_cols(); j++) {

    // Compute the L2 norm of each column of W and divide by
    // the length.
    double l2_norm = 0;
    for (index_t i = 0; i < w_factor.n_rows(); i++) {
      l2_norm += fl::math::Sqr(w_factor.get(i, j));
    }
    l2_norm = sqrt(l2_norm);
    if (l2_norm > 0) {
      for (index_t i = 0; i < w_factor.n_rows(); i++) {
        w_factor.set(i, j, w_factor.get(i, j) / l2_norm);
      }
    }

    // Scale the corresponding row of the H by the L2 norm.
    for (index_t i = 0; i < h_factor.n_cols(); i++) {
      h_factor.set(j, i, l2_norm * h_factor.get(j, i));
    }
  }
}

template<typename Table_t>
template<typename Precision_t>
bool BppNnlsNmf<Table_t>::KktConditionSatisfied_(
  const Precision_t &kkt_residual,
  const Precision_t &initial_kkt_residual) {

  // Tolerance level.
  const Precision_t tolerance = 1e-4;
  return kkt_residual <= tolerance * initial_kkt_residual;
}

template<typename Table_t>
template<typename Precision_t>
void BppNnlsNmf<Table_t>::Compute(
  const fl::dense::Matrix<Precision_t, false> &input_matrix,
  const index_t &rank,
  int num_iterations,
  fl::dense::Matrix<Precision_t, false> *w_factor,
  fl::dense::Matrix<Precision_t, false> *h_factor) {

  // Allocate space for W and H.
  w_factor->Init(input_matrix.n_rows(), rank);
  h_factor->Init(rank, input_matrix.n_cols());

  // Compute the Frobenius norm of the matrix to be factored.
  Precision_t input_matrix_squared_frobenius_norm =
    fl::dense::ops::Dot(input_matrix.n_elements(),
                        input_matrix.ptr(), input_matrix.ptr());

  // Initialize w_factor and h factor: each element
  // initialized to random number.
  boost::uniform_real<double> random_uniform_generator
  (0.0, sqrt(input_matrix_squared_frobenius_norm) /
   ((double) w_factor->n_elements()));
  boost::lagged_fibonacci19937 engine;
  for (index_t j = 0; j < rank; j++) {
    for (index_t i = 0; i < input_matrix.n_rows(); i++) {
      w_factor->set(i, j, random_uniform_generator.operator()
                    <boost::lagged_fibonacci19937>(engine));
    }
  }
  h_factor->SetAll(0.0);

  // The left hand side of the normal equations that are computed
  // during iterations.
  fl::dense::Matrix<Precision_t, false> w_transposed_times_w;
  fl::dense::Matrix<Precision_t, false> h_times_h_transposed;
  w_transposed_times_w.Init(w_factor->n_cols(), w_factor->n_cols());
  w_transposed_times_w.SetZero();
  for (int i = 0; i < w_factor->n_rows(); i++) {
    for (int j = 0; j < w_factor->n_cols(); j++) {
      for (int k = 0; k < w_factor->n_cols(); k++) {
        w_transposed_times_w.set(j, k, w_transposed_times_w.get(j, k) +
                                 w_factor->get(i, j) * w_factor->get(i, k));
      }
    }
  }
  h_times_h_transposed.Init(h_factor->n_rows(), h_factor->n_rows());
  h_times_h_transposed.SetZero();
  for (int i = 0; i < h_factor->n_cols(); i++) {
    for (int j = 0; j < h_factor->n_rows(); j++) {
      for (int k = 0; k < h_factor->n_rows(); k++) {
        h_times_h_transposed.set(j, k, h_times_h_transposed.get(j, k) +
                                 h_factor->get(j, i) * h_factor->get(k, i));
      }
    }
  }

  // The initial KKT residual.
  Precision_t initial_kkt_residual =
    KktResidual_(input_matrix, *w_factor, *h_factor,
                 w_transposed_times_w, h_times_h_transposed);

  // The computed KKT residual.
  Precision_t kkt_residual = 0;
  fl::logger->Message()<<"Starting block coordient optimization"<<std::endl;
  fl::logger->Message()<<"Frobenious norm of the input matrix V: "<<
    fl::math::Pow<double,1,2>(input_matrix_squared_frobenius_norm)<<std::endl;

  // Loop until the KKT residual is sufficiently small.
  for (int iteration_num = 0; iteration_num < num_iterations ||
       num_iterations < 0; iteration_num++) {

    // Fix W and solve for H.
    BlockPrincipalPivotingNNLS_<Precision_t, false>(*w_factor,
        input_matrix, *h_factor, w_transposed_times_w);

    // Fix H and solve for W.
    BlockPrincipalPivotingNNLS_<Precision_t, true>(*h_factor,
        input_matrix, *w_factor, h_times_h_transposed);

    // Compute KKT residual.
    kkt_residual = KktResidual_(input_matrix, *w_factor, *h_factor,
                                w_transposed_times_w,
                                h_times_h_transposed);
    fl::logger->Message()<< "iteration: "<< iteration_num+1<<", kkt residual: " << kkt_residual
      <<", relative error (to ||V||): "
      << 100 * kkt_residual/fl::math::Pow<double,1,2>(input_matrix_squared_frobenius_norm)
      <<"%" << std::endl;
    // Normalize the W factor to have unit-norm columns.
    Normalize_(*w_factor, *h_factor);

    // Break out if the KKT conditions are satisfied.
    if (KktConditionSatisfied_(kkt_residual, initial_kkt_residual)) {
      break;
    }

  } // end of the outer main loop
}
};
};

#endif
