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

#ifndef FL_LITE_MLPACK_KDE_LBFGS_DEV_H
#define FL_LITE_MLPACK_KDE_LBFGS_DEV_H

#include "fastlib/la/linear_algebra.h"
#include "fastlib/optimization/lbfgs/lbfgs.h"

namespace fl {
namespace ml {

template<typename FunctionType>
void Lbfgs<FunctionType>::LbfgsParam::set_max_num_line_searches(
  int max_num_line_searches_in) {

  max_line_search_ = max_num_line_searches_in;
}

template<typename FunctionType>
double Lbfgs<FunctionType>::LbfgsParam::armijo_constant() const {
  return armijo_constant_;
}

template<typename FunctionType>
double Lbfgs<FunctionType>::LbfgsParam::min_step() const {
  return min_step_;
}

template<typename FunctionType>
double Lbfgs<FunctionType>::LbfgsParam::max_step() const {
  return max_step_;
}

template<typename FunctionType>
int Lbfgs<FunctionType>::LbfgsParam::max_line_search() const {
  return max_line_search_;
}

template<typename FunctionType>
double Lbfgs<FunctionType>::LbfgsParam::wolfe() const {
  return wolfe_;
}

template<typename FunctionType>
Lbfgs<FunctionType>::LbfgsParam::LbfgsParam() {
  armijo_constant_ = 1e-4;
  min_step_ = 1e-20;
  max_step_ = 1e20;
  max_line_search_ = 20;
  wolfe_ = 0.9;
}

template<typename FunctionType>
double Lbfgs<FunctionType>::ChooseScalingFactor_(
  int iteration_num,
  const fl::data::MonolithicPoint<double> &gradient) {

  double scaling_factor = 1.0;
  if (iteration_num > 0) {
    int previous_pos = (iteration_num - 1) % num_basis_;
    fl::data::MonolithicPoint<double> s_basis;
    fl::data::MonolithicPoint<double> y_basis;
    s_lbfgs_.MakeColumnVector(previous_pos, &s_basis);
    y_lbfgs_.MakeColumnVector(previous_pos, &y_basis);
    scaling_factor = fl::la::Dot(s_basis, y_basis) /
                     fl::la::Dot(y_basis, y_basis);
  }
  else {
    scaling_factor = 1.0 / sqrt(fl::la::Dot(gradient, gradient));
  }
  return scaling_factor;
}

template<typename FunctionType>
bool Lbfgs<FunctionType>::GradientNormTooSmall_(
  const fl::data::MonolithicPoint<double> &gradient) {

  const double threshold = 1e-5;
  return fl::la::LengthEuclidean(gradient) < threshold;
}

template<typename FunctionType>
void Lbfgs<FunctionType>::Init(FunctionType &function_in, int num_basis) {
  function_ = &function_in;
  new_iterate_tmp_.Init(index_t(function_->num_dimensions()));
  s_lbfgs_.Init(function_->num_dimensions(), num_basis);
  y_lbfgs_.Init(function_->num_dimensions(), num_basis);
  num_basis_ = num_basis;

  // Allocate the pair holding the min iterate information.
  min_point_iterate_.first.Init(index_t(function_->num_dimensions()));
  min_point_iterate_.first.SetZero();
  min_point_iterate_.second = std::numeric_limits<double>::max();
}

template<typename FunctionType>
bool Lbfgs<FunctionType>::LineSearch_(
  double &function_value,
  fl::data::MonolithicPoint<double> &iterate,
  fl::data::MonolithicPoint<double> &gradient,
  const fl::data::MonolithicPoint<double> &search_direction,
  double &step_size) {

  // Implements the line search with back-tracking.

  // The initial linear term approximation in the direction of the
  // search direction.
  double initial_search_direction_dot_gradient =
    fl::la::Dot(gradient, search_direction);

  // If it is not a descent direction, just report failure.
  if (initial_search_direction_dot_gradient > 0.0) {
    return false;
  }

  // Save the initial function value.
  double initial_function_value = function_value;

  // Unit linear approximation to the decrease in function value.
  double linear_approx_function_value_decrease = param_.armijo_constant() *
      initial_search_direction_dot_gradient;

  // The number of iteration in the search.
  int num_iterations = 0;

  // Armijo step size scaling factor for increase and decrease.
  const double inc = 2.1;
  const double dec = 0.5;
  double width = 0;
  for (; ;) {

    // Perform a step and evaluate the gradient and the function
    // values at that point.
    new_iterate_tmp_.CopyValues(iterate);
    fl::la::AddExpert(step_size, search_direction, &new_iterate_tmp_);
    function_value = Evaluate_(new_iterate_tmp_);
    function_->Gradient(new_iterate_tmp_, &gradient);
    num_iterations++;

    if (function_value > initial_function_value + step_size *
        linear_approx_function_value_decrease) {
      width = dec;
    }
    else {

      // Check Wolfe's condition.
      double search_direction_dot_gradient =
        fl::la::Dot(gradient, search_direction);

      if (search_direction_dot_gradient < param_.wolfe() *
          initial_search_direction_dot_gradient) {
        width = inc;
      }
      else {
        if (search_direction_dot_gradient > -param_.wolfe() *
            initial_search_direction_dot_gradient) {
          width = dec;
        }
        else {
          break;
        }
      }
    }

    // Terminate when the step size gets too small or too big or it
    // exceeds the max number of iterations.
    if (step_size < param_.min_step()) {
      fl::logger->Message() << "Step size is too small!";
      return false;
    }
    if (step_size > param_.max_step()) {
      fl::logger->Message() << "Step size is too big!";
      return false;
    }
    if (num_iterations >= param_.max_line_search()) {
      fl::logger->Message() << "Bailing out: too many search trials!";
      return false;
    }

    // Scale the step size.
    step_size *= width;
  }

  // Move to the new iterate.
  iterate.CopyValues(new_iterate_tmp_);
  return true;
}

template<typename FunctionType>
void Lbfgs<FunctionType>::SearchDirection_(
  const fl::data::MonolithicPoint<double> &gradient,
  int iteration_num, double scaling_factor,
  fl::data::MonolithicPoint<double> *search_direction) {

  fl::data::MonolithicPoint<double> q;
  q.Copy(gradient);

  // Temporary variables.
  fl::data::MonolithicPoint<double> rho;
  fl::data::MonolithicPoint<double> alpha;
  rho.Init(num_basis_);
  alpha.Init(num_basis_);

  fl::data::MonolithicPoint<double> y_basis, s_basis;
  index_t limit = std::max(iteration_num - num_basis_, index_t(0));
  for (int i = iteration_num - 1; i >= limit; i--) {
    int translated_position = i % num_basis_;
    s_lbfgs_.MakeColumnVector(translated_position, &s_basis);
    y_lbfgs_.MakeColumnVector(translated_position, &y_basis);
    rho[ iteration_num - i - 1 ] = 1.0 / fl::la::Dot(y_basis, s_basis);
    alpha[ iteration_num - i - 1 ] = rho [ iteration_num - i - 1] *
                                     fl::la::Dot(s_basis, q);
    fl::la::AddExpert(-alpha[iteration_num - i - 1], y_basis, &q);
   // q=q-alpha[iteration_num - i - 1]*y_basis;
  }
  fl::dense::ops::Scale< fl::la::Overwrite >(scaling_factor, q,
      search_direction);
  for (int i = limit; i <= iteration_num - 1; i++) {
    int translated_position = i % num_basis_;
    s_lbfgs_.MakeColumnVector(translated_position, &s_basis);
    y_lbfgs_.MakeColumnVector(translated_position, &y_basis);
    double beta = rho[ iteration_num - i - 1 ] *
                  fl::la::Dot(y_basis, *search_direction);
    fl::la::AddExpert(alpha [ iteration_num - i - 1 ] - beta, s_basis,
                      search_direction);
  }

  // Negate the search direction so that it is a descent direction.
  fl::la::SelfScale(-1.0, search_direction);
}

template<typename FunctionType>
void Lbfgs<FunctionType>::UpdateBasisSet_(
  int iteration_num,
  const fl::data::MonolithicPoint<double> &iterate,
  const fl::data::MonolithicPoint<double> &old_iterate,
  const fl::data::MonolithicPoint<double> &gradient,
  const fl::data::MonolithicPoint<double> &old_gradient) {

  int overwrite_pos = iteration_num % num_basis_;
  fl::data::MonolithicPoint<double> s_basis;
  fl::data::MonolithicPoint<double> y_basis;
  s_lbfgs_.MakeColumnVector(overwrite_pos, &s_basis);
  y_lbfgs_.MakeColumnVector(overwrite_pos, &y_basis);
  fl::la::Sub<fl::la::Overwrite>(old_iterate, iterate, &s_basis);
  fl::la::Sub<fl::la::Overwrite>(old_gradient, gradient, &y_basis);
}

template<typename FunctionType>
double Lbfgs<FunctionType>::Evaluate_(
  const fl::data::MonolithicPoint<double> &iterate) {

  // Evaluate the function and keep track of the minimum function
  // value encountered during the optimization.
  double function_value = function_->Evaluate(iterate);

  if (function_value < min_point_iterate_.second) {
    min_point_iterate_.first.CopyValues(iterate);
    min_point_iterate_.second = function_value;
  }
  return function_value;
}

template<typename FunctionType>
const std::pair< fl::data::MonolithicPoint<double>, double > &
Lbfgs<FunctionType>::min_point_iterate() const {
  return min_point_iterate_;
}

template<typename FunctionType>
void Lbfgs<FunctionType>::set_max_num_line_searches(
  int max_num_line_searches_in) {

  param_.set_max_num_line_searches(max_num_line_searches_in);
}

template<typename FunctionType>
bool Lbfgs<FunctionType>::Optimize(int num_iterations,
                                   fl::data::MonolithicPoint<double> *iterate) {
  FL_SCOPED_LOG(Lbfgs);
  // The old iterate to be saved.
  fl::data::MonolithicPoint<double> old_iterate;
  old_iterate.Init(index_t(function_->num_dimensions()));
  old_iterate.SetZero();

  // Whether to optimize until convergence.
  bool optimize_until_convergence = (num_iterations <= 0);

  // The initial function value.
  double function_value = Evaluate_(*iterate);

  // The gradient: the current and the old.
  fl::data::MonolithicPoint<double> gradient;
  fl::data::MonolithicPoint<double> old_gradient;
  gradient.Init(index_t(function_->num_dimensions()));
  gradient.SetZero();
  old_gradient.Init(index_t(function_->num_dimensions()));
  old_gradient.SetZero();

  // The search direction.
  fl::data::MonolithicPoint<double> search_direction;
  search_direction.Init(index_t(function_->num_dimensions()));
  search_direction.SetZero();

  // The initial gradient value.
  function_->Gradient(*iterate, &gradient);

  // The boolean flag telling whether the line search succeeded at
  // least once.
  bool line_search_successful_at_least_once = false;

  // The main optimization loop.
  int it_num;
  for (it_num = 0; optimize_until_convergence ||
       it_num < num_iterations; it_num++) {

    // Break when the norm of the gradient becomes too small.
    if (GradientNormTooSmall_(gradient)) {
      break;
    }

    // Choose the scaling factor.
    double scaling_factor = ChooseScalingFactor_(it_num, gradient);

    // Build an approximation to the Hessian and choose the search
    // direction for the current iteration.
    SearchDirection_(gradient, it_num, scaling_factor, &search_direction);

    // Save the old iterate and the gradient before stepping.
    old_iterate.CopyValues(*iterate);
    old_gradient.CopyValues(gradient);

    // Do a line search and take a step.
    double step_size = 1.0;
    bool search_is_success =
      LineSearch_(function_value, *iterate, gradient, search_direction,
                  step_size);

    if (search_is_success == false) {
      fl::logger->Warning() << "Could not find an adequate step size.";
      break;
    }

    line_search_successful_at_least_once = search_is_success;

    // Overwrite an old basis set.
    UpdateBasisSet_(it_num, *iterate, old_iterate, gradient, old_gradient);

  } // end of the optimization loop.

  // fl::logger->Debug() << "[LBFGS] Finished at iteration num " << it_num;

  return line_search_successful_at_least_once;
}

};
};

#endif
