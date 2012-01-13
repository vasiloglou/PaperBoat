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
/** @file gauss_seidel_lasso.h
 *
 */

#ifndef MLPACK_LASSO_GAUSS_SEIDEL_LASSO_H
#define MLPACK_LASSO_GAUSS_SEIDEL_LASSO_H

#include "fastlib/dense/matrix.h"
#include "boost/mpl/void.hpp"
#include "boost/utility.hpp"
#include "boost/program_options.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/random/lagged_fibonacci.hpp"

namespace fl {
namespace ml {

template<typename TableType>
class LassoModel: boost::noncopyable {

  public:

    typedef typename TableType::Dataset_t::Point_t::CalcPrecision_t
    Precision_t;

    TableType *table;

    std::vector<index_t> predictor_indices;

    fl::dense::Matrix<Precision_t, true> coefficients;

  public:

    void PrintCoefficients(const char *file_name);

    template<typename TableType1>
    void Export(TableType1 *coeff_table);
};

template <typename TableType>
class GaussSeidelLasso: boost::noncopyable {

  private:

    template<typename Precision_t>
    static void ExtractNormalEquations_(
      TableType &table,
      const std::vector<index_t> &predictor_indices,
      const index_t &prediction_index,
      fl::dense::Matrix<Precision_t, false> *normal_left_hand_side,
      fl::dense::Matrix<Precision_t, true> *normal_right_hand_side);

    template<typename Precision_t>
    static void ExtractNormalEquations_(
      TableType &table,
      const index_t &prediction_index,
      fl::dense::Matrix<Precision_t, false> *normal_left_hand_side,
      fl::dense::Matrix<Precision_t, true> *normal_right_hand_side);

    template<typename Precision_t>
    static void ComputeViolations_(
      const fl::dense::Matrix<Precision_t, true> &coefficients,
      const double &threshold, const double &lambda,
      const fl::dense::Matrix<Precision_t, true> &gradient,
      fl::dense::Matrix<Precision_t, true> *violations);

    template<typename Precision_t>
    static Precision_t MaxSlope_(
      const fl::dense::Matrix<Precision_t, true> &coefficients,
      const double &zero_threshold, const double &lambda,
      const fl::dense::Matrix<Precision_t, true> &gradient,
      const index_t &max_position);

    template<typename Precision_t>
    static void MaxElement_(
      const fl::dense::Matrix<Precision_t, true> &violations,
      Precision_t *max_value, index_t *max_position);

    template<bool inner_loop, typename Precision_t>
    static void MaxViolation_(
      const fl::dense::Matrix<Precision_t, true> &coefficients,
      const fl::dense::Matrix<Precision_t, true> &violations,
      const Precision_t &zero_threshold,
      Precision_t *max_value, index_t *max_position);

    template<typename Precision_t>
    static bool DifferentSigns_(const Precision_t &first,
                                const Precision_t &second);

    template<typename Precision_t>
    static void UpdateGradient_(
      const fl::dense::Matrix<Precision_t, false> &normal_left_hand_side,
      const index_t &column_index,
      const Precision_t &coeff_original,
      const Precision_t &coeff_new,
      fl::dense::Matrix<Precision_t, true> &gradient);

    template<typename Precision_t>
    static void ComputeGradient_(
      const fl::dense::Matrix<Precision_t, false> &normal_left_hand_side,
      const fl::dense::Matrix<Precision_t, true> &normal_right_hand_side,
      const fl::dense::Matrix<Precision_t, true> &coefficients,
      fl::dense::Matrix<Precision_t, true> &gradient);

    template<typename Precision_t>
    static bool SolutionFound_(const fl::dense::Matrix<Precision_t, true>
                               &violations,
                               const double &optimal_tolerance);

  public:

    /** @brief Computes the LASSO coefficient using the
     *         Gauss-Seidel method.
     *
     *  The algorithm assumes that the data is standardized.
     */
    template<typename Precision_t, bool initial_strategy_is_zero>
    static void Compute(
      TableType &table,
      Precision_t violation_tolerance,
      Precision_t gradient_tolerance,
      index_t iterations,
      const std::vector<index_t> &predictor_indices,
      const index_t &prediction_index,
      const double &lambda,
      fl::ml::LassoModel<TableType> *model_in);

};

template<>
class GaussSeidelLasso<boost::mpl::void_> : boost::noncopyable  {
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
};
}
}

#endif
