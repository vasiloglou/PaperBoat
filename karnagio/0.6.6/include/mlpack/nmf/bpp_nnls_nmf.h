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
/** @file bpp_nnls_nmf.h
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

#ifndef MLPACK_NMF_BPP_NNLS_NMF_H
#define MLPACK_NMF_BPP_NNLS_NMF_H

#include "boost/random/lagged_fibonacci.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/program_options.hpp"
#include "fastlib/dense/matrix.h"
#include "boost/utility.hpp"

namespace fl {
namespace ml {

template<typename TableType>
class BppNnlsNmf : boost::noncopyable {

  private:

    template<typename Precision_t>
    static void ExtractSubVectorHelper_(const fl::dense::Matrix
                                        <Precision_t, false> &matrix,
                                        const fl::dense::Matrix
                                        <index_t, true>
                                        &num_active_primal_variables,
                                        const fl::dense::Matrix
                                        <bool, false> &primal_active_set,
                                        const index_t &column_to_solve,
                                        const bool index_row_by_active_vars,
                                        fl::dense::Matrix
                                        <Precision_t, true> *submatrix);

    template<typename Precision_t>
    static void ExtractSubMatrixHelper_(const fl::dense::Matrix
                                        <Precision_t, false> &matrix,
                                        const fl::dense::Matrix
                                        <index_t, true>
                                        &num_active_primal_variables,
                                        const fl::dense::Matrix
                                        <bool, false> &primal_active_set,
                                        const index_t &column_to_solve,
                                        const bool index_row_by_active_vars,
                                        const bool index_col_by_active_vars,
                                        fl::dense::Matrix
                                        <Precision_t, false> *submatrix);

    template<typename Precision_t>
    static void ExtractSubMatrices_(const fl::dense::Matrix < Precision_t,
                                    false > &normal_left_hand_side,
                                    const fl::dense::Matrix < Precision_t,
                                    false > &normal_right_hand_side,
                                    const fl::dense::Matrix<index_t, true>
                                    &num_active_primal_variables,
                                    const fl::dense::Matrix<bool, false>
                                    &primal_active_set,
                                    const index_t &column_to_solve,
                                    fl::dense::Matrix<Precision_t, false>
                                    *primal_left_submatrix,
                                    fl::dense::Matrix<Precision_t, true>
                                    *primal_right_submatrix,
                                    fl::dense::Matrix<Precision_t, false>
                                    *dual_left_submatrix,
                                    fl::dense::Matrix<Precision_t, true>
                                    *dual_right_submatrix);

    template<typename Precision_t>
    static bool Feasible_(
      const fl::dense::Matrix<bool, false>     &primal_active_set,
      const fl::dense::Matrix<Precision_t, false> &primal_variables,
      const fl::dense::Matrix<Precision_t, false> &dual_variables,
      std::vector< std::vector<index_t> > &primal_violations,
      std::vector< std::vector<index_t> > &dual_violations);

    template<typename Precision_t>
    static void UpdateDualVariables_(const fl::dense::Matrix < Precision_t,
                                     false > &dual_left_submatrix,
                                     const fl::dense::Matrix < Precision_t,
                                     true > &dual_right_submatrix,
                                     const fl::dense::Matrix<bool, false>
                                     &primal_active_set,
                                     const index_t &column_to_solve,
                                     const fl::dense::Matrix < Precision_t,
                                     false > &primal_variables,
                                     fl::dense::Matrix<Precision_t, false>
                                     &dual_variables);

    template<typename Precision_t>
    static void SolveNormalEquation_(const fl::dense::Matrix < Precision_t,
                                     false > &left_hand_side,
                                     const fl::dense::Matrix < Precision_t,
                                     true > &right_hand_side,
                                     const fl::dense::Matrix<bool, false>
                                     &primal_active_set,
                                     const index_t &column_to_solve,
                                     fl::dense::Matrix<Precision_t, false>
                                     &primal_variables);

    template<typename Precision_t>
    static void ComputeVariables_(const fl::dense::Matrix < Precision_t,
                                  false > &normal_left_hand_side,
                                  const fl::dense::Matrix < Precision_t,
                                  false > &normal_right_hand_side,
                                  const fl::dense::Matrix<index_t, true>
                                  &num_active_primal_variables,
                                  const fl::dense::Matrix<bool, false>
                                  &primal_active_set,
                                  fl::dense::Matrix<Precision_t, false>
                                  &primal_variables,
                                  fl::dense::Matrix<Precision_t, false>
                                  &dual_variables);

    static void ApplyExchangeRules_(const std::vector
                                    < std::vector<index_t> >
                                    &primal_violations,
                                    const std::vector
                                    < std::vector<index_t> >
                                    &dual_violations,
                                    fl::dense::Matrix<index_t, true>
                                    &num_block_exchange_rules,
                                    fl::dense::Matrix<index_t, true>
                                    &backup_exchange_rules,
                                    fl::dense::Matrix<index_t, true>
                                    &num_active_primal_variables,
                                    fl::dense::Matrix<bool, false>
                                    &primal_active_set);

    template<typename Precision_t, bool transpose_mode>
    static void BlockPrincipalPivotingNNLS_(
      const fl::dense::Matrix<Precision_t, false> &matrix,
      const fl::dense::Matrix<Precision_t, false> &right_hand_side,
      fl::dense::Matrix<Precision_t, false> &solution,
      fl::dense::Matrix<Precision_t, false> &normal_left_hand_side);

    template<typename Precision_t>
    static Precision_t KktResidual_(const fl::dense::Matrix < Precision_t,
                                    false > &input_matrix,
                                    const fl::dense::Matrix < Precision_t,
                                    false > &w_factor,
                                    const fl::dense::Matrix < Precision_t,
                                    false > &h_factor,
                                    const fl::dense::Matrix < Precision_t,
                                    false > &w_transposed_times_w,
                                    const fl::dense::Matrix < Precision_t,
                                    false > &h_times_h_transposed);

    template<typename Precision_t>
    static void Normalize_(
      fl::dense::Matrix<Precision_t, false> &w_factor,
      fl::dense::Matrix<Precision_t, false> &h_factor);

    template<typename Precision_t>
    static bool KktConditionSatisfied_(const Precision_t &kkt_residual,
                                       const Precision_t
                                       &initial_kkt_residual);

  public:

    template<typename Precision_t>
    static void Compute(
      const fl::dense::Matrix<Precision_t, false> &input_matrix,
      const index_t &rank,
      int num_iterations,
      fl::dense::Matrix<Precision_t, false> *w_factor,
      fl::dense::Matrix<Precision_t, false> *h_factor);

    template<typename DataAccessType>
    static int Main(DataAccessType *data,
                    const std::vector<std::string> &args);
};

template<>
class BppNnlsNmf<boost::mpl::void_> : boost::noncopyable {
  public:
    template<typename TableType>
    class Core {
      public:
        template<typename DataAccessType>
        static int Main(DataAccessType *data,
                        boost::program_options::variables_map &vm);
    };
    template<typename DataAccessType, typename BranchType>
    static int Main(DataAccessType *data,
                    const std::vector<std::string> &args);
  private:
    // these are mpl structs to help in the matrix copying
    struct CopyMatrix {
      template<typename TableType, typename MatrixType>
      static void Init(const TableType &table, MatrixType * const matrix) {
        matrix->Init(table.n_attributes(), table.n_entries());
        fl::logger->Message() << "Copying the data into a matrix" << std::endl;
        for (index_t i = 0; i < table.n_entries(); ++i) {
          typename TableType::Point_t point;
          table.get(i, &point);
          for (index_t j = 0; j < matrix->n_rows(); ++j) {
            matrix->set(j, i, point[j]);
          }
        }
      }
    };
    struct AliasMatrix {
      template<typename TableType, typename MatrixType>
      static void Init(const TableType &table, MatrixType * const matrix) {
        matrix->Alias(
          table.get_point_collection().dense->template get<double>());
      }
    };

};

}
} // namespaces
#endif
#include "bpp_nnls_nmf_defs.h"

