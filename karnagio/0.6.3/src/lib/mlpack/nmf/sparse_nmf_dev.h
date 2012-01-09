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

#ifndef FL_LITE_MLPACK_NMF_LBFGS_NMF_DEV_H_
#define FL_LITE_MLPACK_NMF_LBFGS_NMF_DEV_H_
#include <deque>
#include <algorithm>
#include "mlpack/nmf/sparse_nmf.h"
#include "fastlib/optimization/lbfgs/lbfgs_dev.h"
#include "boost/math/special_functions/fpclassify.hpp"

namespace fl {
namespace ml {

template<typename NmfArgs>
NmfWFactorFunction<NmfArgs>::NmfWFactorFunction() {
  table_ = NULL;
  k_rank_ = 0;
  current_h_factor_ = NULL;
}

template<typename NmfArgs>
void NmfWFactorFunction<NmfArgs>::Init(
  InputTable_t *table_in, int k_rank_in,
  const fl::data::MonolithicPoint<CalcPrecision_t> *current_h_factor) {

  table_ = table_in;
  k_rank_ = k_rank_in;
  current_h_factor_ = current_h_factor;
}

template<typename NmfArgs>
typename NmfWFactorFunction<NmfArgs>::CalcPrecision_t
NmfWFactorFunction<NmfArgs>::Evaluate(
  const fl::data::MonolithicPoint<CalcPrecision_t> &w_factor) {

  // The squared loss.
  CalcPrecision_t objective_function = 0;

  // Evaluate the loss on the nonzero components of the table.
  for (int i = 0; i < table_->n_entries(); i++) {

    typename InputTable_t::Point_t point;
    table_->get(i, &point);

    // Iterate over the nonzero components of the current point.
    for (typename InputTable_t::Point_t::iterator it = point.begin();
         it != point.end(); ++it) {

      // (row, i)-th element of the table matches the dot product
      // between the row-th row of w factor and the i-th column of the
      // h-factor.
      objective_function += fl::math::Sqr(it.value()
          - fl::dense::ops::Dot(
                        k_rank_,
                        w_factor.ptr()+i*k_rank_,
                        current_h_factor_->ptr()+it.attribute()*k_rank_));
    }
  }
  return objective_function;
}

template<typename NmfArgs>
void NmfWFactorFunction<NmfArgs>::Gradient(
  const fl::data::MonolithicPoint<CalcPrecision_t> &w_factor,
  fl::data::MonolithicPoint<CalcPrecision_t> *gradient) {

  // Set the gradient to zero so that we can accumulate it.
  gradient->SetZero();

  // Loop through each point on the table to compute (V - WH).
  for (int i = 0; i < table_->n_entries(); i++) {

    typename InputTable_t::Point_t point;
    table_->get(i, &point);

    // Iterate over the nonzero components of the current point.
    for (typename InputTable_t::Point_t::iterator it = point.begin();
         it != point.end(); ++it) {

      CalcPrecision_t diff=-(it.value() - fl::dense::ops::Dot(k_rank_,
            w_factor.ptr()+i*k_rank_, 
            current_h_factor_->ptr()+it.attribute()*k_rank_));
      fl::dense::ops::AddExpert(k_rank_, 
          diff,
          current_h_factor_->ptr()+it.attribute()*k_rank_,
          gradient->ptr()+i*k_rank_);
    }
  }
}

template<typename NmfArgs>
int NmfWFactorFunction<NmfArgs>::num_dimensions() const {
  return table_->n_entries()*k_rank_;
}

template<typename NmfArgs>
NmfHFactorFunction<NmfArgs>::NmfHFactorFunction() {
  table_ = NULL;
  k_rank_ = 0;
  current_w_factor_ = NULL;
}

template<typename NmfArgs>
void NmfHFactorFunction<NmfArgs>::Init(
  InputTable_t *table_in, int k_rank_in,
  const fl::data::MonolithicPoint<CalcPrecision_t> *current_w_factor) {

  table_ = table_in;
  k_rank_ = k_rank_in;
  current_w_factor_ = current_w_factor;
}

template<typename NmfArgs>
typename NmfHFactorFunction<NmfArgs>::CalcPrecision_t
NmfHFactorFunction<NmfArgs>::Evaluate(
  const fl::data::MonolithicPoint<CalcPrecision_t> &h_factor) {

  // The squared loss.
  CalcPrecision_t objective_function = 0;

  // Evaluate the loss on the nonzero components of the table.
  for (int i = 0; i < table_->n_entries(); i++) {

    typename InputTable_t::Point_t point;
    table_->get(i, &point);

    // Iterate over the nonzero components of the current point.
    for (typename InputTable_t::Point_t::iterator it = point.begin();
         it != point.end(); ++it) {

      // (row, i)-th element of the table matches the dot product
      // between the row-th row of w factor and the i-th column of the
      // h-factor.
      objective_function += fl::math::Sqr(
                              it.value()
                              -fl::dense::ops::Dot(k_rank_, 
                                 h_factor.ptr()+it.attribute()*k_rank_,
                                 current_w_factor_->ptr()+i*k_rank_));
    }
  }
  return objective_function;
}

template<typename NmfArgs>
void NmfHFactorFunction<NmfArgs>::Gradient(
  const fl::data::MonolithicPoint<CalcPrecision_t> &h_factor,
  fl::data::MonolithicPoint<CalcPrecision_t> *gradient) {

  // Set the gradient to zero so that we can accumulate it.
  gradient->SetZero();

  // Loop through each point on the table to compute (V - WH).
  for (int i = 0; i < table_->n_entries(); i++) {

    typename InputTable_t::Point_t point;
    table_->get(i, &point);

    // Iterate over the nonzero components of the current point.
    for (typename InputTable_t::Point_t::iterator it = point.begin();
         it != point.end(); ++it) {

      // The row index of the nonzero component of the table.
      CalcPrecision_t diff=-(it.value() - fl::dense::ops::Dot(k_rank_,
          h_factor.ptr()+it.attribute()*k_rank_,
          current_w_factor_->ptr()+i*k_rank_));
      fl::dense::ops::AddExpert(k_rank_, diff, 
          current_w_factor_->ptr()+i*k_rank_,
          gradient->ptr()+it.attribute()*k_rank_);
    }
  }
}

template<typename NmfArgs>
int NmfHFactorFunction<NmfArgs>::num_dimensions() const {
  return table_->n_attributes()*k_rank_;
}

}} // namspaces

namespace fl {
namespace ml {

template<typename NmfArgs>
SparseNmf<NmfArgs>::SparseNmf() {

  table_ = NULL;
  k_rank_ = 0;
  w_sparsity_factor_ = 0;
  h_sparsity_factor_ = 0;
  w_factor_ = NULL;
  h_factor_ = NULL;
  iterations_ = -1;
  lbfgs_rank_ = 3;
  lbfgs_steps_=3;
  epochs_=-1;
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::Init(
  typename SparseNmf<NmfArgs>::InputTable_t *table_in,
  typename SparseNmf<NmfArgs>::FactorsTable_t *w_factor_in,
  typename SparseNmf<NmfArgs>::FactorsTable_t *h_factor_in) {

  table_ = table_in;
  w_factor_ = w_factor_in;
  h_factor_ = h_factor_in;
}

template<typename NmfArgs>
typename SparseNmf<NmfArgs>::CalcPrecision_t SparseNmf<NmfArgs>::L1Norm_(
  const fl::data::MonolithicPoint<CalcPrecision_t> &point) const {

  CalcPrecision_t l1_norm = 0;
  for (int i = 0; i < point.length(); i++) {
    l1_norm += fabs(point[i]);
  }
  return l1_norm;
}

template<typename NmfArgs>
typename SparseNmf<NmfArgs>::CalcPrecision_t SparseNmf<NmfArgs>::L1Norm_(
  const CalcPrecision_t *point, int length) const {

  CalcPrecision_t l1_norm = 0;
  for (int i = 0; i < length; i++) {
    l1_norm += fabs(point[i]);
  }
  return l1_norm;
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::SparseProjection_(
  CalcPrecision_t *point, index_t length, CalcPrecision_t sparsity_factor) {

  // Compute the L2 norm and the L1 norm of the starting vector.
  CalcPrecision_t l2_norm = fl::dense::ops::LengthEuclidean(length, point);
  CalcPrecision_t l1_norm = L1Norm_(point, length);

  // The required L1 norm given the sparsity level.
  CalcPrecision_t required_l1_norm =
    l2_norm * (sqrt((CalcPrecision_t)length) - sparsity_factor * (sqrt((CalcPrecision_t)length) - 1));

  // Temporary vector which will be copied back.
  fl::data::MonolithicPoint<CalcPrecision_t> s_point;
  s_point.Init(length);
  for (int i = 0; i < length; i++) {
    s_point[i] = point[i] + (required_l1_norm - l1_norm) /
                      ((CalcPrecision_t) length);
  }

  std::vector<bool> zeta(length, false);
  index_t zeta_size=0;
  while (true) {
    fl::data::MonolithicPoint<CalcPrecision_t> m_point;
    m_point.Init(length);
    m_point.SetZero();
    for(index_t i=0; i<length; ++i) {
      DEBUG_ASSERT(length-zeta_size);
      if (zeta[i]==false) {
        m_point[i]=required_l1_norm/(length-zeta_size);
      }
    }
    // calculating alpha
    // We need to form a second order equation first
    CalcPrecision_t a2=0;
    CalcPrecision_t a1=0;
    CalcPrecision_t a0=-l2_norm*l2_norm;
    for(index_t i=0; i<length; ++i) {
      a2+=fl::math::Sqr(s_point[i]-m_point[i]);
      a1+=2*s_point[i]*(s_point[i]-m_point[i]);
      a0+=fl::math::Sqr(s_point[i]);
    }
    CalcPrecision_t det=a1*a1-4*a2*a0;
    //DEBUG_ASSERT(det>=0);
    if (a2==0) {
      break;
    }
    CalcPrecision_t alpha=0;
    alpha=(-a1+fl::math::Pow<CalcPrecision_t, 1,2>(det>0?det:0))/(2*a2);
    //DEBUG_ASSERT(alpha>=0);
    bool all_nonneg=true;
    zeta_size=0;
    for(index_t i=0; i<length; ++i) {
      s_point[i]=s_point[i]+alpha*(s_point[i]-m_point[i]);
      if (s_point[i]<0) {
        all_nonneg=false;
        zeta[i]=true;
        zeta_size++;
      }
      if (zeta[i]==true) {
        s_point[i]=0;
      }
    }
    if (all_nonneg) {
      break;
    }
    CalcPrecision_t c=0;
    for(index_t i=0; i<length; ++i) {
      c+=s_point[i];
    }
    c=(c-required_l1_norm)/(length-zeta_size);
    DEBUG_ASSERT(!boost::math::isnan(c));
    for(index_t i=0; i<length; ++i) {
      if (zeta[i]==false) {
        s_point[i]-=c;
      }
    }
    /* 
    double temp_l1=0;
    double temp_l2=0;
    for(index_t i=0; i<length; ++i) {
      temp_l1+=s_point[i];
      temp_l2+=s_point[i]*s_point[i];
    }
    temp_l2=fl::math::Pow<CalcPrecision_t, 1, 2>(temp_l2);
    fl::logger->Message()<<"req l1: "<<required_l1_norm <<", new l1:"<< temp_l1<<std::endl;
    fl::logger->Message()<<"past l2: "<<l2_norm<<", new l2: "<<temp_l2<<std::endl;
    */
  };

  // Copy the resulting vector back.
  for (int i = 0; i < length; i++) {
    DEBUG_ASSERT(!boost::math::isnan(s_point[i]));
    point[i] = s_point[i];
  }
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::BatchSparseProjection_(
  fl::data::MonolithicPoint<CalcPrecision_t> &factor,
  int step_length,
  CalcPrecision_t sparsity_factor) {

  CalcPrecision_t *ptr = factor.ptr();
  
  for (int i = 0; i < factor.length(); i += step_length,
       ptr += step_length) {
    SparseProjection_(ptr, step_length, sparsity_factor);
  }
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::Train(const std::string &mode) {

  // Work with the temporary vectors in place for optimization. The W
  // factor is layed out as the column major vector, the H factor as
  // the row major vector.
  fl::data::MonolithicPoint<CalcPrecision_t> current_w_factor;
  fl::data::MonolithicPoint<CalcPrecision_t> current_h_factor;
  current_w_factor.Init(k_rank_ * table_->n_entries());
  current_h_factor.Init(k_rank_ * table_->n_attributes());

  // we compute the l2 norm of the data so that we can report the relative error
  // of the objective 
  CalcPrecision_t table_l2_norm = 0;
  index_t num_of_elements=0;
  // Evaluate the loss on the nonzero components of the table.
  for (int i = 0; i < table_->n_entries(); i++) {
    typename InputTable_t::Point_t point;
    table_->get(i, &point);
    // Iterate over the nonzero components of the current point.
    for (typename InputTable_t::Point_t::iterator it = point.begin();
         it != point.end(); ++it) {
      // (row, i)-th element of the table matches the dot product
      // between the row-th row of w factor and the i-th column of the
      // h-factor.
      table_l2_norm += fl::math::Sqr(it.value());
      num_of_elements++;
    }
  }

  // Initialize the starting vectors. For the H factor, have each of
  // its row to be unit L2 norm, apply the sparseness constraint.
  for (int i = 0; i < current_w_factor.length(); i++) {
    current_w_factor[i] = fl::math::Random(0.0, 1.0);
  }
  for (int i = 0; i < current_h_factor.length(); i++) {
    current_h_factor[i] = fl::math::Random(0.0, 1.0);
  }

  // Stochastic gradient descent
  fl::ml::NmfWFactorFunction<NmfArgs> temp_factor_function;
  temp_factor_function.Init(table_, k_rank_,
                            &current_h_factor);
 CalcPrecision_t initial_objective=temp_factor_function.Evaluate(current_w_factor);
 fl::logger->Message() <<
   "Initial Objective:  " << initial_objective
    << ", Relative error: "
    << 100*fl::math::Pow<CalcPrecision_t,1, 2>(initial_objective/table_l2_norm) <<"%"
    <<std::endl;
  CalcPrecision_t step=step0_;
  if (mode.find("stoc")!=std::string::npos) {
    fl::logger->Message() << "Running stochastic gradient descent for "
      << epochs_ << " epochs." << std::endl;
    double last_non_contributing_updates=1;
    for(index_t j=0; j<epochs_; ++j) {
      index_t non_contributing_updates=0;
      step=step0_/(j+1);
      for (int i = 0; i < table_->n_entries(); i++) {
        typename InputTable_t::Point_t point;
        table_->get(i, &point);
        // Iterate over the nonzero components of the current point.
        for(typename InputTable_t::Point_t::iterator it = point.begin();
          it != point.end(); ++it) {
          // (row, i)-th element of the table matches the dot product
          // between the row-th row of w factor and the i-th column of the
          // h-factor.
          CalcPrecision_t diff=-(it.value() - fl::dense::ops::Dot(k_rank_,
          current_w_factor.ptr()+i*k_rank_, 
          current_h_factor.ptr()+it.attribute()*k_rank_));
          // Update the w factor temporarily
          fl::data::MonolithicPoint<CalcPrecision_t> temp_w, temp_h;
          temp_w.Copy(current_w_factor.ptr()+i*k_rank_, k_rank_);
          temp_h.Copy(current_h_factor.ptr()+it.attribute()*k_rank_, k_rank_);
          fl::dense::ops::AddExpert(k_rank_, 
          -diff*step,
          current_h_factor.ptr()+it.attribute()*k_rank_,
          temp_w.ptr());
          // Update the h_factor
          fl::dense::ops::AddExpert(k_rank_, 
          -diff*step,
          current_w_factor.ptr()+i*k_rank_,
          temp_h.ptr());
          // now do the projections
          // W
          if (w_sparsity_factor_>0) {
            SparseProjection_(temp_w.ptr(), k_rank_,
               w_sparsity_factor_);
          } else {
            for(index_t l=0; l<k_rank_; ++l) {
              temp_w[l]=fl::math::ClampNonNegative(
                  temp_w[l]);
            }
          }
          // H
          if (h_sparsity_factor_>0) {
            // temporarily we don't have any solution for that so we just do a nonnegative 
            // projection
            for(index_t l=0; l<k_rank_; ++l) {
              temp_h[l]=fl::math::ClampNonNegative(temp_h[l]);
            } 
          } else {
            for(index_t l=0; l<k_rank_; ++l) {
              temp_h[l]=fl::math::ClampNonNegative(temp_h[l]);
            } 
          }


          // Let's see if we recorded any progress, if we did let's copy it back
          CalcPrecision_t new_diff=(it.value() - fl::dense::ops::Dot(temp_w,
                temp_h));
          if (fabs(diff)>fabs(new_diff)) {
            for(index_t l=0;l<k_rank_; ++l) {
              current_w_factor[i*k_rank_+l]=temp_w[l];
              current_h_factor[it.attribute()*k_rank_+l]=temp_h[l];      
            }
          } else {
            non_contributing_updates++;
          }
        }
      }
      temp_factor_function.Init(table_, k_rank_,
                                &current_h_factor);
      CalcPrecision_t objective=temp_factor_function.Evaluate(current_w_factor);
      last_non_contributing_updates=
          1.0*non_contributing_updates/num_of_elements;
      fl::logger->Message() << "Epoch: " 
        << j
        << ", step: "
        << step
        << ", Objective: "
        << objective
        << ", Relative error: "
        << 100*fl::math::Pow<CalcPrecision_t,1, 2>(objective/table_l2_norm) <<"%"
        << ", Non contributing updates: "
        << 100.0 *  last_non_contributing_updates<< "%"
        << std::endl;

    }
  }
  
  if (mode.find("lbfgs")!=std::string::npos) {
    fl::logger->Message() << "Running LBFGS .."<<std::endl; 
    // Do a sparse projection.
    if (w_sparsity_factor_>0) {
      BatchSparseProjection_(current_w_factor, k_rank_,
                           w_sparsity_factor_);
    } else {
      for(index_t i=0; i<current_w_factor.size(); ++i) {
        current_w_factor[i]=fl::math::ClampNonNegative(current_w_factor[i]);
      }
    }
      // Do a sparse projection for each row of the H factor.
      if (h_sparsity_factor_>0) {
        fl::data::MonolithicPoint<CalcPrecision_t> temp(current_h_factor);
        for(index_t i=0; i<k_rank_; ++i) {
          for(index_t j=0; j<table_->n_attributes(); ++j) {
            temp.set(i*table_->n_attributes()+j, current_h_factor.get(j*k_rank_+i));
          }
        }
        BatchSparseProjection_(temp, table_->n_attributes(),
                               h_sparsity_factor_);
        for(index_t i=0; i<k_rank_; ++i) {
          for(index_t j=0; j<table_->n_attributes(); ++j) {
            current_h_factor.set(j*k_rank_+i, temp.get(i*table_->n_attributes()+j));
          }
        }
      } else {
        for(index_t i=0; i<current_h_factor.size(); ++i) {
          current_h_factor[i]=fl::math::ClampNonNegative(current_h_factor[i]);
        } 
      }
  
    if (iterations_ < 0) {
      fl::logger->Message() << "Running until convergence."<<std::endl;
    }
    else {
      fl::logger->Message() << "Running for " << iterations_ << " iterations."<<std::endl;
    }
  
    int iteration_num = 0;
    for (iteration_num = 0; iterations_ < 0 ||
         iteration_num < iterations_; iteration_num++) {
  
      // Optimize the W factor.
      fl::ml::Lbfgs<fl::ml::NmfWFactorFunction<NmfArgs> > nmf_w_factor_engine;
      fl::ml::NmfWFactorFunction<NmfArgs> nmf_w_factor_function;
      nmf_w_factor_function.Init(table_, k_rank_,
                                 &current_h_factor);
      nmf_w_factor_engine.Init(nmf_w_factor_function, lbfgs_rank_);
      bool w_factor_optimized =
        nmf_w_factor_engine.Optimize(lbfgs_steps_, &current_w_factor);
  
      // Optimize the H factor.
      fl::ml::Lbfgs<fl::ml::NmfHFactorFunction<NmfArgs> > nmf_h_factor_engine;
      fl::ml::NmfHFactorFunction<NmfArgs> nmf_h_factor_function;
      nmf_h_factor_function.Init(table_, k_rank_, 
                                 &current_w_factor);
      nmf_h_factor_engine.Init(nmf_h_factor_function, lbfgs_rank_);
      bool h_factor_optimized =
        nmf_h_factor_engine.Optimize(lbfgs_steps_, &current_h_factor);
  
      // Do a sparse projection for each column of the W factor.
      if (w_sparsity_factor_>0) {
        BatchSparseProjection_(current_w_factor, k_rank_,
                               w_sparsity_factor_);
      } else {
        for(index_t i=0; i<current_w_factor.size(); ++i) {
         current_w_factor[i]=fl::math::ClampNonNegative(current_w_factor[i]);
        }
      }
  
      // Do a sparse projection for each row of the H factor.
      if (h_sparsity_factor_>0) {
        fl::data::MonolithicPoint<CalcPrecision_t> temp;
        temp.Init(index_t(current_h_factor.size()));
        for(index_t i=0; i<k_rank_; ++i) {
          for(index_t j=0; j<table_->n_attributes(); ++j) {
            temp.set(i*table_->n_attributes()+j, current_h_factor.get(j*k_rank_+i));
          }
        }
        BatchSparseProjection_(temp, table_->n_attributes(),
                               h_sparsity_factor_);
        for(index_t i=0; i<k_rank_; ++i) {
          for(index_t j=0; j<table_->n_attributes(); ++j) {
            current_h_factor.set(j*k_rank_+i, temp.get(i*table_->n_attributes()+j));
          }
        }
      } else {
        for(index_t i=0; i<current_h_factor.size(); ++i) {
          current_h_factor[i]=fl::math::ClampNonNegative(current_h_factor[i]);
        } 
      }
      CalcPrecision_t objective=nmf_w_factor_function.Evaluate(current_w_factor);
      fl::logger->Message() << "iteration: " << iteration_num+1 <<", "
        <<"Objective: " << objective
        <<", Relative error: "
        << 100*fl::math::Pow<CalcPrecision_t,1, 2>(objective/table_l2_norm) <<"%"<<std::endl;
      // If no progress is being made, then break out.
      if (w_factor_optimized == false || h_factor_optimized == false) {
        break;
      }
    }
    fl::logger->Message() << "Ran for " << iteration_num << " iterations.";
  }
  // Copy the result to the tables.

  typename FactorsTable_t::Point_t w_factor_point;
  for (int i = 0; i < current_w_factor.length()/k_rank_; i++) {
    w_factor_->get(i, &w_factor_point);
    for(index_t j=0; j<k_rank_; ++j) {
      w_factor_point.set(j, current_w_factor[i*k_rank_+j]);
    }
  }
  
  typename FactorsTable_t::Point_t h_factor_point;
  for (int i = 0; i < current_h_factor.length()/k_rank_; i++) {
    h_factor_->get(i, &h_factor_point);
    for(index_t j=0; j<k_rank_; ++j) {
      h_factor_point.set(j, current_h_factor[i*k_rank_+j]);
    }
  }  
}

template<typename NmfArgs>
typename SparseNmf<NmfArgs>::CalcPrecision_t SparseNmf<NmfArgs>::Evaluate(
  index_t i, index_t j) {

}

template<typename NmfArgs>
template<typename ContainerType>
void SparseNmf<NmfArgs>::Evaluate(std::vector<std::pair<index_t, index_t> > &indices,
                                 ContainerType *values) {
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::set_rank(index_t k_rank_in) {
  k_rank_ = k_rank_in;
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::set_iterations(index_t iterations_in) {
  iterations_ = iterations_in;
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::set_w_sparsity_factor(
  CalcPrecision_t sparsity_factor_in) {
  w_sparsity_factor_ = sparsity_factor_in;
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::set_h_sparsity_factor(
  CalcPrecision_t sparsity_factor_in) {
  h_sparsity_factor_ = sparsity_factor_in;
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::set_lbfgs_rank(index_t lbfgs_rank) {
  lbfgs_rank_=lbfgs_rank;
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::set_lbfgs_steps(index_t lbfgs_steps) {
  lbfgs_steps_=lbfgs_steps;
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::set_epochs(index_t epochs) {
  epochs_=epochs;
}

template<typename NmfArgs>
void SparseNmf<NmfArgs>::set_step0(CalcPrecision_t step0) {
  step0_=step0;
}

template<typename NmfArgs>
typename SparseNmf<NmfArgs>::FactorsTable_t *SparseNmf<NmfArgs>::get_w() {
  return w_factor_;
}

template<typename NmfArgs>
typename SparseNmf<NmfArgs>::FactorsTable_t *SparseNmf<NmfArgs>::get_h() {
  return h_factor_;
}


}
} // namespace

#endif
