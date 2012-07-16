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
#ifndef FL_LITE_FASTLIB_OPTIMIZATION_LBFGS_LBFG_DEV_H_
#define FL_LITE_FASTLIB_OPTIMIZATION_LBFGS_LBFG_DEV_H_
#include "fastlib/optimization/augmented_lagrangian/lbfgs.h"

namespace fl {
namespace optim {
template<typename CalcPrecisionType>
LbfgsOpts<CalcPrecisionType>::LbfgsOpts() {
  num_of_points = 1000;
  sigma = 10;
  objective_factor = 1.0;
  eta = 0.99;
  gamma = 5;
  new_dimension = 2;
  feasibility_tolerance = 0.01;
  desired_feasibility = 100;
  wolfe_sigma1 = 0.1;
  wolfe_sigma2 = 0.9;
  step_size = 3.0;
  silent = false;
  show_warnings = true;
  use_default_termination = true;
  norm_grad_tolerance = 0.1;
  wolfe_beta = 0.8;
  min_beta = 1e-40;
  max_iterations = 10000;
  // the memory of bfgs
  mem_bfgs = 20;
}

template<typename TemplateMap>
void Lbfgs<TemplateMap>::Init(
  typename Lbfgs<TemplateMap>::OptimizedFunction_t *optimized_function,
  LbfgsOpts<typename Lbfgs<TemplateMap>::CalcPrecision_t> &opts) {
  optimized_function_ = optimized_function;
  num_of_points_ = opts.num_of_points;
  sigma_ = opts.sigma;
  initial_sigma_ = opts.sigma;
  objective_factor_ = opts.objective_factor;
  eta_ = opts.eta;
  gamma_ = opts.gamma;
  new_dimension_ = opts.new_dimension;
  feasibility_tolerance_ = opts.feasibility_tolerance;
  desired_feasibility_ = opts.desired_feasibility;
  wolfe_sigma1_ = opts.wolfe_sigma1;
  wolfe_sigma2_ = opts.wolfe_sigma2;
  step_size_ = opts.step_size;
  silent_ = opts.silent;
  show_warnings_ = opts.show_warnings;
  use_default_termination_ = opts.use_default_termination;
  if (unlikely(wolfe_sigma1_ >= wolfe_sigma2_)) {
    fl::logger->Die() << "Wolfe sigma1 "
      << wolfe_sigma1_ 
      <<"should be less than sigma2 "
      <<wolfe_sigma2_;
  }
  DEBUG_ASSERT(wolfe_sigma1_ > 0);
  DEBUG_ASSERT(wolfe_sigma1_ < 1);
  DEBUG_ASSERT(wolfe_sigma2_ > wolfe_sigma1_);
  DEBUG_ASSERT(wolfe_sigma2_ < 1);
  if (unlikely(wolfe_sigma1_ >= wolfe_sigma2_)) {
    fl::logger->Die()<<"Wolfe sigma1 "
      <<wolfe_sigma1_ 
      <<" should be less than sigma2 "
      <<wolfe_sigma2_;
  }
  DEBUG_ASSERT(wolfe_sigma1_ > 0);
  DEBUG_ASSERT(wolfe_sigma1_ < 1);
  DEBUG_ASSERT(wolfe_sigma2_ > wolfe_sigma1_);
  DEBUG_ASSERT(wolfe_sigma2_ < 1);
  norm_grad_tolerance_ = opts.norm_grad_tolerance;
  wolfe_beta_   = opts.wolfe_beta;
  min_beta_ = opts.min_beta;
  max_iterations_ = opts.max_iterations;
  // the memory of bfgs
  mem_bfgs_ = opts.mem_bfgs;
  optimized_function_->set_sigma(sigma_);
  InitOptimization_();
}

template<typename TemplateMap>
void Lbfgs<TemplateMap>::Destruct() {
  CalcPrecision_t objective;
  CalcPrecision_t feasibility_error;
  optimized_function_->ComputeFeasibilityError(coordinates_,
      &feasibility_error);
  optimized_function_->ComputeObjective(coordinates_, &objective);
}

template<typename TemplateMap>
void Lbfgs<TemplateMap>::ComputeLocalOptimumBFGS() {
  CalcPrecision_t feasibility_error=0;
  if (silent_ == false) {
    fl::logger->Message() << "Starting optimization" <<std::endl;
    // Run a few iterations with gradient descend to fill the memory of BFGS
    fl::logger->Message() << "Initializing BFGS" << std::endl;
  }
  index_bfgs_ = 0;
  // You have to compute also the previous_gradient_ and previous_coordinates_
  // tha are needed only by BFGS
  optimized_function_->ComputeGradient(coordinates_, &gradient_);
  previous_gradient_.CopyValues(gradient_);
  previous_coordinates_.CopyValues(coordinates_);
  ComputeWolfeStep_(&step_, gradient_);
  optimized_function_->ComputeGradient(coordinates_, &gradient_);
  fl::la::Sub<fl::la::Overwrite>(previous_coordinates_, coordinates_, &s_bfgs_[0]);
  fl::la::Sub<fl::la::Overwrite>(previous_gradient_, gradient_, &y_bfgs_[0]);
  ro_bfgs_[0] = fl::la::Dot(s_bfgs_[0], y_bfgs_[0]);
  CalcPrecision_t old_feasibility_error = std::numeric_limits<CalcPrecision_t>::max();
  for (index_t i = 0; i < mem_bfgs_; i++) {
    success_t success = ComputeBFGS_(&step_, gradient_, i);
    if (success == SUCCESS_FAIL) {
      if (silent_ == false) {
        fl::logger->Warning() << "LBFGS failed to find a direction, continuing with gradient descent"<<std::endl;
      }
      ComputeWolfeStep_(&step_, gradient_);
    }
    optimized_function_->ComputeGradient(coordinates_, &gradient_);
    UpdateBFGS_();
    previous_gradient_.CopyValues(gradient_);
    previous_coordinates_.CopyValues(coordinates_);
    num_of_iterations_++;
    if (silent_ == false) {
      ReportProgressFile_();
    }
    /*    if (use_default_termination_== true) {
           optimized_function_->ComputeFeasibilityError(coordinates_,
              &feasibility_error);
          if (feasibility_error < desired_feasibility_) {
            NOTIFY("feasibility error %lg less than desired feasibility %lg",
                feasibility_error, desired_feasibility_);
            return;
          }
        } else {
          if (optimized_function_->IsOptimizationOver(
                coordinates_, gradient_, step_)==true) {
            return;
          }
        }
    */
  }
  if (silent_ == false) {
    fl::logger->Message() << "Now starting optimizing with BFGS" <<std::endl;
  }
  //  index_t failed_tries=0;
  for (index_t it1 = 0; it1 < max_iterations_; it1++) {
    for (index_t it2 = 0; it2 < max_iterations_; it2++) {
      success_t success_bfgs = ComputeBFGS_(&step_, gradient_, mem_bfgs_);
      optimized_function_->ComputeGradient(coordinates_, &gradient_);
      optimized_function_->ComputeFeasibilityError(coordinates_,
          &feasibility_error);
      CalcPrecision_t norm_grad = la::Dot(gradient_, gradient_);
      num_of_iterations_++;
      if (success_bfgs == SUCCESS_FAIL) {
        fl::logger->Warning() << "LBFGS failed to find a direction, continuing with gradient descent"<<std::endl;
        if (ComputeWolfeStep_(&step_, gradient_) == SUCCESS_FAIL) {
          fl::logger->Warning() << "Gradient descent failed too"<<std::endl;
        }
        break;
      }
      if (silent_ == false) {
        ReportProgressFile_();
      }
      // NOTIFY("feasibility_error:%lg desired_feasibility:%lg", feasibility_error, desired_feasibility_);
      if (use_default_termination_ == true) {
        if (feasibility_error < desired_feasibility_) {
          break;
        }
        if (step_*norm_grad / sigma_ < norm_grad_tolerance_) {
          break;
        }
      }
      else {
        if (optimized_function_->IsIntermediateStepOver(
              coordinates_, gradient_, step_) == true) {
          break;
        }
      }
      /*      if (unlikely(UpdateBFGS_()==SUCCESS_FAIL)) {
              failed_tries++;
              if (failed_tries==mem_bfgs_) {
                failed_tries=0;
                break;
              }
            };
      */
      previous_coordinates_.CopyValues(coordinates_);
      previous_gradient_.CopyValues(gradient_);
      // Do this check to make sure the method has not started diverging
      CalcPrecision_t objective;
      optimized_function_->ComputeObjective(coordinates_, &objective);
      if (optimized_function_->IsDiverging(objective)) {
        sigma_ *= gamma_;
        optimized_function_->set_sigma(sigma_);
        optimized_function_->ComputeGradient(coordinates_, &gradient_);
        // break;
      }
    }

    if (silent_ == false) {
      fl::logger->Message() << "Inner loop done, increasing sigma..." << std::endl;
    }

    if (use_default_termination_ == true) {
      if (fabs(old_feasibility_error - feasibility_error)
          / (old_feasibility_error + 1e-20) < feasibility_tolerance_ ||
          feasibility_error < desired_feasibility_) {
        break;
      }
    }
    else {
      if (optimized_function_->IsOptimizationOver(
            coordinates_, gradient_, step_) == true) {
        break;
      }
    }
    old_feasibility_error = feasibility_error;
    UpdateLagrangeMult_();
    optimized_function_->ComputeGradient(coordinates_, &gradient_);
  }

}

template<typename TemplateMap>
void Lbfgs<TemplateMap>::CopyCoordinates(typename Lbfgs<TemplateMap>::Container_t *result) {
  result->Copy(coordinates_);
}

template<typename TemplateMap>
void  Lbfgs<TemplateMap>::set_coordinates(typename Lbfgs<TemplateMap>::Container_t &coordinates) {
  coordinates_.CopyValues(coordinates);
}
template<typename TemplateMap>
typename Lbfgs<TemplateMap>::Container_t *Lbfgs<TemplateMap>::coordinates() {
  return &coordinates_;
}

template<typename TemplateMap>
typename Lbfgs<TemplateMap>::CalcPrecision_t Lbfgs<TemplateMap>::sigma() {
  return sigma_;
}

template<typename TemplateMap>
void Lbfgs<TemplateMap>::set_sigma(typename Lbfgs<TemplateMap>::CalcPrecision_t sigma) {
  sigma_ = sigma;
  optimized_function_->set_sigma(sigma_);
}

template<typename TemplateMap>
void Lbfgs<TemplateMap>::Reset() {
  sigma_ = initial_sigma_;
  optimized_function_->set_sigma(sigma_);
}

template<typename TemplateMap>
void Lbfgs<TemplateMap>::set_max_iterations(index_t max_iterations) {
  max_iterations_ = max_iterations;
}

template<typename TemplateMap>
void Lbfgs<TemplateMap>::InitOptimization_() {
  if (unlikely(new_dimension_ < 0)) {
    fl::logger->Die()<<"You forgot to set the new dimension";
  }
  if (silent_ == false) {
    fl::logger->Message()<<"Initializing optimization"<<std::endl;
  }
  optimized_function_->GiveInitMatrix(&coordinates_);
  num_of_points_ = coordinates_.n_cols();
  new_dimension_ = coordinates_.n_rows();
  previous_coordinates_.Init(new_dimension_, num_of_points_);
  gradient_.Init(new_dimension_, num_of_points_);
  previous_gradient_.Init(new_dimension_, num_of_points_);
  if (unlikely(mem_bfgs_ < 0)) {
    fl::logger->Die()<<"You forgot to initialize the memory for BFGS";
  }
  // Init the memory for BFGS
  s_bfgs_.resize(mem_bfgs_);
  y_bfgs_.resize(mem_bfgs_);
  ro_bfgs_.Init(mem_bfgs_);
  ro_bfgs_.SetAll(0.0);
  for (index_t i = 0; i < mem_bfgs_; i++) {
    s_bfgs_[i].Init(new_dimension_, num_of_points_);
    y_bfgs_[i].Init(new_dimension_, num_of_points_);
  }
  num_of_iterations_ = 0;
}

template<typename TemplateMap>
void Lbfgs<TemplateMap>::UpdateLagrangeMult_() {
  optimized_function_->UpdateLagrangeMult(coordinates_);
  sigma_ *= gamma_;
  optimized_function_->set_sigma(sigma_);
}

// for optimization purposes the direction  is always the negative of what it is supposed
// in the wolfe form. so for example if the direction is the negative gradient the direction
// should be the gradient and not the -gradient
template<typename TemplateMap>
success_t Lbfgs<TemplateMap>::ComputeWolfeStep_(typename Lbfgs<TemplateMap>::CalcPrecision_t *step,
    typename Lbfgs<TemplateMap>::Container_t &direction) {
  success_t success = SUCCESS_PASS;
  Container_t temp_coordinates;
  Container_t temp_gradient;
  temp_gradient.Init(new_dimension_, num_of_points_);
  temp_coordinates.Init(coordinates_.n_rows(), coordinates_.n_cols());
  CalcPrecision_t lagrangian1 = optimized_function_->ComputeLagrangian(coordinates_);
  CalcPrecision_t lagrangian2 = 0;
  CalcPrecision_t beta = wolfe_beta_;
  CalcPrecision_t dot_product = -fl::la::Dot(gradient_, direction);
  CalcPrecision_t wolfe_factor =  dot_product * wolfe_sigma1_ * wolfe_beta_ * step_size_;
  for (index_t i = 0; beta > min_beta_ / (1.0 + sigma_); i++) {
    temp_coordinates.CopyValues(coordinates_);
    fl::la::AddExpert(-step_size_*beta, direction, &temp_coordinates);
    optimized_function_->Project(&temp_coordinates);
    lagrangian2 = optimized_function_->ComputeLagrangian(temp_coordinates);
    // NOTIFY("direction:%lg", la::Dot(direction.n_elements(), direction.ptr(), direction.ptr()) );
    // NOTIFY("step_size:%lg beta:%lg min_beta:%lg", step_size_, beta, min_beta_);
    // NOTIFY("********lagrangian2:%lg lagrangian1:%lg wolfe_factor:%lg", lagrangian2, lagrangian1, wolfe_factor);
    if (lagrangian2 <= lagrangian1 + wolfe_factor)  {
      optimized_function_->ComputeGradient(temp_coordinates, &temp_gradient);
      CalcPrecision_t dot_product_new = -fl::la::Dot(temp_gradient, direction);
      //  NOTIFY("dot_product_new:%lg wolfe_sigma2:%lg dot_product:%lg", dot_product_new,wolfe_sigma2_, dot_product);
      if (dot_product_new >= wolfe_sigma2_*dot_product) {
        success = SUCCESS_PASS;
      }
      else {
        success = SUCCESS_FAIL;
      }
      break;
    }
    beta *= wolfe_beta_;
    wolfe_factor *= wolfe_beta_;
  }
  // optimized_function_->Project(&temp_coordinates);

  if (beta <= min_beta_ / (1.0 + sigma_)) {
    *step = 0;
    return SUCCESS_FAIL;
  }
  else {
    *step = step_size_ * beta;
    coordinates_.CopyValues(temp_coordinates);
    if (success == SUCCESS_FAIL) {
      return SUCCESS_FAIL;
    }
    else {
      return SUCCESS_PASS;
    }
  }
}

template<typename TemplateMap>
success_t Lbfgs<TemplateMap>::ComputeBFGS_(typename Lbfgs<TemplateMap>::CalcPrecision_t *step,
    typename Lbfgs<TemplateMap>::Container_t  &grad, index_t memory) {
  Container_t alpha;
  alpha.Init(mem_bfgs_);
  Container_t scaled_y;
  scaled_y.Init(new_dimension_, num_of_points_);
  Container_t temp_direction(grad);
  for (index_t i = index_bfgs_, num = 0; num < memory; i = (i + 1 + mem_bfgs_) % mem_bfgs_, num++) {
    // printf("i:%i  index_bfgs_:%i\n", i, index_bfgs_);
    alpha[i] = fl::la::Dot(s_bfgs_[i], temp_direction);
    alpha[i] *= ro_bfgs_[i];
    scaled_y.CopyValues(y_bfgs_[i]);
    fl::la::SelfScale(alpha[i], &scaled_y);
    fl::la::SubFrom(scaled_y, &temp_direction);
  }
  // We need to scale the gradient here
  CalcPrecision_t s_y = fl::la::Dot(y_bfgs_[index_bfgs_],
                                    s_bfgs_[index_bfgs_]);
  CalcPrecision_t y_y = fl::la::Dot(y_bfgs_[index_bfgs_],
                                    y_bfgs_[index_bfgs_]);
  if (show_warnings_ == true && unlikely(y_y < 1e-40)) {
    fl::logger->Warning()<<"Gradient differences close to singular...norm="<<y_y<<std::endl;
  }
  CalcPrecision_t norm_scale = s_y / (y_y + 1e-40);
  fl::la::SelfScale(norm_scale, &temp_direction);
  Container_t scaled_s;
  CalcPrecision_t beta;
  scaled_s.Init(new_dimension_, num_of_points_);
  index_t num = 0;
  for (index_t j = (index_bfgs_ + memory - 1) % mem_bfgs_; num < memory;
       j = (j - 1 + mem_bfgs_) % mem_bfgs_) {
    // printf("j:%i  index_bfgs_:%i\n", j, index_bfgs_);

    beta = fl::la::Dot(y_bfgs_[j], temp_direction);
    beta *= ro_bfgs_[j];
    scaled_s.CopyValues(s_bfgs_[j]);
    fl::la::SelfScale(alpha[j] - beta, &scaled_s);
    fl::la::AddTo(scaled_s, &temp_direction);
    num++;
  }
  success_t success = ComputeWolfeStep_(step, temp_direction);
  /*  if (step==0) {
      NONFATAL("BFGS Failed looking in the other direction...\n");
      la::Scale(-1.0, &temp_direction);
      ComputeWolfeStep_(step, temp_direction);
      *step=-*step;
      fx_timer_stop(module_, "bfgs_step");
      return SUCCESS_FAIL;
    }
  */
  return success;
}

template<typename TemplateMap>
success_t Lbfgs<TemplateMap>::UpdateBFGS_() {
  index_t try_index_bfgs = (index_bfgs_ - 1 + mem_bfgs_) % mem_bfgs_;
  if (UpdateBFGS_(try_index_bfgs) == SUCCESS_FAIL) {
    return SUCCESS_FAIL;
  }
  else {
    index_bfgs_ = try_index_bfgs;
    return SUCCESS_PASS;
  }
}

template<typename TemplateMap>
success_t Lbfgs<TemplateMap>::UpdateBFGS_(index_t index_bfgs) {
  // shift all values
  Container_t temp_s_bfgs;
  Container_t temp_y_bfgs;
  fl::la::Sub<fl::la::Init>(previous_coordinates_, coordinates_, &temp_s_bfgs);
  fl::la::Sub<fl::la::Init>(previous_gradient_, gradient_, &temp_y_bfgs);
  CalcPrecision_t temp_ro = fl::la::Dot(temp_s_bfgs,
                                        temp_y_bfgs);
  CalcPrecision_t y_norm = fl::la::Dot(temp_y_bfgs, temp_y_bfgs);
  if (temp_ro < 1e-70*y_norm) {
    if (show_warnings_ == true) {
      fl::logger->Warning()<<"Rejecting s, y they don't satisfy curvature condition "
               "s*y="<<temp_ro  << " < 1e-70 *||y||^2="<< 1e-70*y_norm<<std::endl;
    }
    return SUCCESS_FAIL;
  }
  s_bfgs_[index_bfgs].CopyValues(temp_s_bfgs);
  y_bfgs_[index_bfgs].CopyValues(temp_y_bfgs);
  ro_bfgs_[index_bfgs] = 1.0 / temp_ro;

  return SUCCESS_PASS;
}

template<typename TemplateMap>
std::string Lbfgs<TemplateMap>::ComputeProgress_() {
  CalcPrecision_t lagrangian = optimized_function_->ComputeLagrangian(coordinates_);
  CalcPrecision_t objective;
  optimized_function_->ComputeObjective(coordinates_, &objective);
  CalcPrecision_t feasibility_error;
  optimized_function_->ComputeFeasibilityError(
    coordinates_, &feasibility_error);
  CalcPrecision_t norm_grad = fl::math::Pow<CalcPrecision_t, 1, 2>(fl::la::Dot(gradient_, gradient_));
  char buffer[1024];
  sprintf(buffer, "iteration:%lli sigma:%lg lagrangian:%lg objective:%lg error:%lg "
          "grad_norm:%lg step:%lg",
          (long long)num_of_iterations_, sigma_, lagrangian, objective,
          feasibility_error, norm_grad, step_);
  return std::string(buffer);
}

template<typename TemplateMap>
void Lbfgs<TemplateMap>::ReportProgressFile_() {
  std::string progress = ComputeProgress_();
  fl::logger->Message()<<progress<<std::endl;
}

}
} //namespaces

#endif
