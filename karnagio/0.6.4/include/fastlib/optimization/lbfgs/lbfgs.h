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

#ifndef FL_LITE_MLPACK_KDE_LBFGS_H
#define FL_LITE_MLPACK_KDE_LBFGS_H

#include "fastlib/data/monolithic_point.h"
#include "fastlib/dense/matrix.h"

namespace fl {
namespace ml {
template<typename FunctionType>
class Lbfgs {

  public:

    /** @brief The set of parameters for the L-BFGS routine.
     */
    class LbfgsParam {
      private:

        /** @brief Parameter to control the accuracy of the line search
         *         routine for determining the Armijo condition.
         */
        double armijo_constant_;

        /** @brief The minimum step of the line search routine.
         */
        double min_step_;

        /** @brief The maximum step of the line search routine.
         */
        double max_step_;

        /** @brief The maximum number of trials for the line search.
         */
        int max_line_search_;

        /** @brief Parameter for detecting Wolfe condition.
         */
        double wolfe_;

      public:

	void set_max_num_line_searches(int max_num_line_searches_in);

        double armijo_constant() const;

        double min_step() const;

        double max_step() const;

        int max_line_search() const;

        double wolfe() const;

        LbfgsParam();
    };

  private:

    LbfgsParam param_;

    FunctionType *function_;

    fl::data::MonolithicPoint<double> new_iterate_tmp_;

    fl::dense::Matrix<double, false> s_lbfgs_;

    fl::dense::Matrix<double, false> y_lbfgs_;

    index_t num_basis_;

    std::pair< fl::data::MonolithicPoint<double>, double > min_point_iterate_;

  private:

    double Evaluate_(const fl::data::MonolithicPoint<double> &iterate);

    double ChooseScalingFactor_(
      int iteration_num,
      const fl::data::MonolithicPoint<double> &gradient);

    bool GradientNormTooSmall_(
      const fl::data::MonolithicPoint<double> &gradient);

    bool LineSearch_(double &function_value,
                     fl::data::MonolithicPoint<double> &iterate,
                     fl::data::MonolithicPoint<double> &gradient,
                     const fl::data::MonolithicPoint<double> &search_direction,
                     double &step_size);

    void SearchDirection_(const fl::data::MonolithicPoint<double> &gradient,
                          int iteration_num, double scaling_factor,
                          fl::data::MonolithicPoint<double> *search_direction);

    void UpdateBasisSet_(
      int iteration_num,
      const fl::data::MonolithicPoint<double> &iterate,
      const fl::data::MonolithicPoint<double> &old_iterate,
      const fl::data::MonolithicPoint<double> &gradient,
      const fl::data::MonolithicPoint<double> &old_gradient);

  public:

    const std::pair< fl::data::MonolithicPoint<double>, double > &min_point_iterate() const;

    void Init(FunctionType &function_in, int num_basis);

    void set_max_num_line_searches(int max_num_line_searches_in);

    bool Optimize(int num_iterations,
                  fl::data::MonolithicPoint<double> *iterate);
};
};
};

#endif
