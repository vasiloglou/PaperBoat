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
#ifndef FL_LITE_FASTLIB_OPTIMIZATION_LBFGS_LBFGS_H_
#define FL_LITE_FASTLIB_OPTIMIZATION_LBFGS_LBFGS_H_
#include "boost/mpl/at.hpp"
#include "boost/mpl/has_key.hpp"
#include <vector>
#include <string>
#include "fastlib/base/base.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/dense/matrix.h"
/**
 * @author Nikolaos Vasiloglou (nvasil@ieee.org)
 * @file lbfgs.h
 *
 * This class implements the L-BFGS method as desribed in:
 *
 * @book{nocedal1999no,
 *       title={{Numerical Optimization}},
 *       author={Nocedal, J. and Wright, S.J.},
 *       year={1999},
 *       publisher={Springer}
 * }
 * */


namespace fl {
namespace optim  {
class LbfgsTypeOpts {
  public:
    typedef boost::mpl::void_ OptimizedFunctionType;
};
template<typename CalcPrecisionType>
struct LbfgsOpts {
  LbfgsOpts();
  index_t num_of_points;
  // The number of points for the optimization variable.
  CalcPrecisionType sigma;
  // The initial penalty parameter on the augmented lagrangian
  CalcPrecisionType objective_factor;
  // obsolete
  CalcPrecisionType eta;
  // wolfe parameter
  CalcPrecisionType gamma;
  // sigma increase rate, after inner loop is done sigma is multiplied by gamma
  index_t new_dimension;
  // The dimension of the points
  CalcPrecisionType desired_feasibility;
  // Since this is used with augmented lagrangian, we need to know
  // when the  feasibility is sufficient.
  CalcPrecisionType feasibility_tolerance;
  // if the feasibility is not improved by that quantity, then it stops.
  CalcPrecisionType wolfe_sigma1;
  // wolfe parameter
  CalcPrecisionType wolfe_sigma2;
  //  wolfe parameter
  CalcPrecisionType min_beta;
  //wolfe parameter
  CalcPrecisionType wolfe_beta;
  // wolfe parameter
  CalcPrecisionType step_size;
  // Initial step size for the wolfe search
  bool silent;
  // if true then it doesn't emmit updates
  bool show_warnings;
  //  if true then it does show warnings
  bool use_default_termination;
  // let this module decide where to terminate. If false then
  // the objective function decides
  CalcPrecisionType norm_grad_tolerance;
  // If the norm of the gradient doesn't change more than
  // this quantity between two iterations and the use_default_termination
  // is set, the algorithm terminates
  index_t max_iterations;
  // maximum number of iterations required
  index_t mem_bfgs;
  // the limited memory of BFGS
};

template<typename TemplateMap>
class Lbfgs {
  public:
    typedef typename TemplateMap::OptimizedFunctionType OptimizedFunction_t;
    typedef typename OptimizedFunction_t::CalcPrecision_t CalcPrecision_t;
    typedef typename OptimizedFunction_t::ResultTable_t Container_t;

    void Init(OptimizedFunction_t *optimized_function, LbfgsOpts<CalcPrecision_t> &opts);
    void Destruct();
    void ComputeLocalOptimumBFGS();
    void ReportProgress();
    void CopyCoordinates(Container_t *result);
    void Reset();
    void set_coordinates(Container_t &coordinates);
    void set_desired_feasibility(CalcPrecision_t desired_feasibility);
    void set_feasibility_tolerance(CalcPrecision_t feasibility_tolerance);
    void set_norm_grad_tolerance(CalcPrecision_t norm_grad_tolerance);
    void set_max_iterations(index_t max_iterations);
    Container_t *coordinates();
    CalcPrecision_t sigma();
    void set_sigma(CalcPrecision_t sigma);

  private:
    void InitOptimization_();
    void ComputeWolfeStep_();
    void UpdateLagrangeMult_();
    success_t ComputeWolfeStep_(CalcPrecision_t *step, Container_t &direction);
    success_t ComputeBFGS_(CalcPrecision_t *step, Container_t &grad, index_t memory);
    success_t UpdateBFGS_();
    success_t UpdateBFGS_(index_t index_bfgs);
    void BoundConstrain();
    std::string ComputeProgress_();
    void ReportProgressFile_();

    OptimizedFunction_t  *optimized_function_;
    index_t num_of_iterations_;
    index_t num_of_points_;
    index_t new_dimension_;
    CalcPrecision_t sigma_;
    CalcPrecision_t initial_sigma_;
    CalcPrecision_t objective_factor_;
    CalcPrecision_t eta_;
    CalcPrecision_t gamma_;
    CalcPrecision_t step_;
    CalcPrecision_t desired_feasibility_;
    CalcPrecision_t feasibility_tolerance_;
    CalcPrecision_t norm_grad_tolerance_;
    CalcPrecision_t wolfe_sigma1_;
    CalcPrecision_t wolfe_sigma2_;
    CalcPrecision_t wolfe_beta_;
    CalcPrecision_t min_beta_;
    bool silent_;
    bool show_warnings_;
    bool use_default_termination_;
    std::vector<Container_t> s_bfgs_;
    std::vector<Container_t> y_bfgs_;
    fl::dense::Matrix<CalcPrecision_t, true> ro_bfgs_;
    index_t index_bfgs_;
    Container_t coordinates_;
    Container_t previous_coordinates_;
    Container_t gradient_;
    Container_t previous_gradient_;
    index_t max_iterations_;
    CalcPrecision_t step_size_;
    // the memory of bfgs
    index_t mem_bfgs_;
};
}
}

#include "fastlib/optimization/augmented_lagrangian/lbfgs_dev.h"
#endif //LBFGS_H_
