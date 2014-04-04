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
#ifndef FL_LITE_MLPACK_MVU_MVU_OBJECTIVES_DEV_H_
#define FL_LITE_MLPACK_MVU_MVU_OBJECTIVES_DEV_H_
#include "mlpack/mvu/mvu_objectives.h"
#include "mlpack/mvu/mvu_objectives_defs.h"
#include "mlpack/allkn/allkn_dev.h"

namespace fl {
namespace ml {
template<typename TemplateOpts>
void MaxVariance<TemplateOpts>::Init(typename MaxVariance<TemplateOpts>::MVUOpts &opts) {
  knns_ = opts.knns;
  new_dimension_ = opts.new_dimension;
  num_of_points_ = opts.num_of_points;
  infeasibility1_ = std::numeric_limits<CalcPrecision_t>::max();
  previous_infeasibility1_ = std::numeric_limits<CalcPrecision_t>::max();
  desired_feasibility_error_ = opts.desired_feasibility_error;
  infeasibility_tolerance_ = opts.infeasibility_tolerance;
  grad_tolerance_ = opts.grad_tolerance;

  if (opts.auto_tune == true) {
    fl::logger->Message() << "Auto-tuning the knn" << std::endl;
    MaxVarianceUtils::EstimateKnns(*opts.from_tree_neighbors,
                                   *opts.from_tree_distances,
                                   opts.knns,
                                   num_of_points_,
                                   new_dimension_,
                                   &knns_);
    fl::logger->Message() << "Optimum knns is " << knns_ << std::endl;
  }
  fl::logger->Message() << "Consolidating neighbors" << std::endl;
  MaxVarianceUtils::ConsolidateNeighbors(
    *opts.from_tree_neighbors,
    *opts.from_tree_distances,
    opts.knns,
    knns_,
    &nearest_neighbor_pairs_,
    &nearest_distances_,
    &num_of_nearest_pairs_);

  eq_lagrange_mult_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult_.SetAll(1.0);
  CalcPrecision_t max_nearest_distance = 0;
  for (index_t i = 0; i < num_of_nearest_pairs_; i++) {
    max_nearest_distance = std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_ = -max_nearest_distance *
                               num_of_points_ * num_of_points_;

  fl::logger->Message() << "Lower bound for optimization " << sum_of_furthest_distances_
    << std::endl;
}

template<typename TemplateOpts>
void MaxVariance<TemplateOpts>::Destruct() {
  eq_lagrange_mult_.Destruct();

}

template<typename TemplateOpts>
void MaxVariance<TemplateOpts>::ComputeGradient(
  typename MaxVariance<TemplateOpts>::ResultTable_t &coordinates,
  typename MaxVariance<TemplateOpts>::ResultTable_t *gradient) {

  gradient->CopyValues(coordinates);
  // we need to use -CRR^T because we want to maximize CRR^T
  fl::la::SelfScale(-1.0, gradient);
  index_t dimension = coordinates.n_rows();
  CalcPrecision_t *a_i_r = new CalcPrecision_t[dimension];
  for (index_t i = 0; i < num_of_nearest_pairs_; i++) {
    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    CalcPrecision_t *point1 = coordinates.GetColumnPtr(n1);
    CalcPrecision_t *point2 = coordinates.GetColumnPtr(n2);
    CalcPrecision_t dist_diff = fl::dense::ops::DistanceSqEuclidean(dimension, point1, point2)
                                - nearest_distances_[i];
    // Make sure this change if right (used SubExpert instead of SubOverwrite)
    fl::dense::ops::SubExpert(dimension, point2, point1, a_i_r);

    // equality constraints
    fl::dense::ops::AddExpert(dimension,
                              -eq_lagrange_mult_[i] + dist_diff*sigma_,
                              a_i_r,
                              gradient->GetColumnPtr(n1));
    fl::dense::ops::AddExpert(dimension,
                              eq_lagrange_mult_[i] - dist_diff*sigma_,
                              a_i_r,
                              gradient->GetColumnPtr(n2));
  }
  delete[] a_i_r;
}

template<typename TemplateOpts>
void MaxVariance<TemplateOpts>::ComputeObjective(
  typename MaxVariance<TemplateOpts>::ResultTable_t &coordinates,
  typename MaxVariance<TemplateOpts>::CalcPrecision_t *objective) {
  *objective = 0;
  index_t dimension = coordinates.n_rows();
  for (index_t i = 0; i < coordinates.n_cols(); i++) {
    *objective -= fl::dense::ops::Dot(dimension,
                                      coordinates.GetColumnPtr(i),
                                      coordinates.GetColumnPtr(i));
  }
}

template<typename TemplateOpts>
void MaxVariance<TemplateOpts>::ComputeFeasibilityError(
  typename MaxVariance<TemplateOpts>::ResultTable_t &coordinates,
  typename MaxVariance<TemplateOpts>::CalcPrecision_t *error) {

  index_t dimension = coordinates.n_rows();
  *error = 0;
  for (index_t i = 0; i < num_of_nearest_pairs_; i++) {
    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    CalcPrecision_t *point1 = coordinates.GetColumnPtr(n1);
    CalcPrecision_t *point2 = coordinates.GetColumnPtr(n2);
    *error += fl::math::Sqr(fl::dense::ops::DistanceSqEuclidean(dimension,
                            point1, point2) - nearest_distances_[i]);
  }
}

template<typename TemplateOpts>
typename MaxVariance<TemplateOpts>::CalcPrecision_t MaxVariance<TemplateOpts>::ComputeLagrangian(
  typename MaxVariance<TemplateOpts>::ResultTable_t &coordinates) {
  index_t dimension = coordinates.n_rows();
  CalcPrecision_t lagrangian = 0;
  ComputeObjective(coordinates, &lagrangian);
  for (index_t i = 0; i < num_of_nearest_pairs_; i++) {
    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    CalcPrecision_t *point1 = coordinates.GetColumnPtr(n1);
    CalcPrecision_t *point2 = coordinates.GetColumnPtr(n2);
    CalcPrecision_t dist_diff = fl::dense::ops::DistanceSqEuclidean(dimension, point1, point2)
                                - nearest_distances_[i];
    lagrangian += dist_diff * dist_diff * sigma_
                  -eq_lagrange_mult_[i] * dist_diff;
  }
  return lagrangian;
}

template<typename TemplateOpts>
void MaxVariance<TemplateOpts>::UpdateLagrangeMult(
  typename MaxVariance<TemplateOpts>::ResultTable_t &coordinates) {
  index_t dimension = coordinates.n_rows();
  for (index_t i = 0; i < num_of_nearest_pairs_; i++) {
    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    CalcPrecision_t *point1 = coordinates.GetColumnPtr(n1);
    CalcPrecision_t *point2 = coordinates.GetColumnPtr(n2);
    CalcPrecision_t dist_diff =
      fl::dense::ops::DistanceSqEuclidean(dimension, point1, point2)
      - nearest_distances_[i];
    eq_lagrange_mult_[i] -= sigma_ * dist_diff;
  }
}

template<typename TemplateOpts>
void MaxVariance<TemplateOpts>::set_sigma(
  typename MaxVariance<TemplateOpts>::CalcPrecision_t sigma) {
  sigma_ = sigma;
}

template<typename TemplateOpts>
bool MaxVariance<TemplateOpts>::IsDiverging(
  typename MaxVariance<TemplateOpts>::CalcPrecision_t objective) {
  if (objective < sum_of_furthest_distances_) {
    fl::logger->Message() << "objective("<< objective<< ") < sum_of_furthest_distances ("
      << sum_of_furthest_distances_<< ")" <<std::endl; 
    return true;
  }
  else {
    return false;
  }
}

template<typename TemplateOpts>
bool MaxVariance<TemplateOpts>::IsOptimizationOver(
  typename MaxVariance<TemplateOpts>::ResultTable_t &coordinates,
  typename MaxVariance<TemplateOpts>::ResultTable_t &gradient,
  typename MaxVariance<TemplateOpts>::CalcPrecision_t step) {
  ComputeFeasibilityError(coordinates, &infeasibility1_);
  if (infeasibility1_ < desired_feasibility_error_ ||
      fabs(infeasibility1_ -previous_infeasibility1_) < infeasibility_tolerance_)  {
    fl::logger->Message() << "Optimization is over" <<std::endl;
    return true;
  }
  else {
    previous_infeasibility1_ = infeasibility1_;
    return false;
  }

}
template<typename TemplateOpts>
bool MaxVariance<TemplateOpts>::IsIntermediateStepOver(
  typename MaxVariance<TemplateOpts>::ResultTable_t &coordinates,
  typename MaxVariance<TemplateOpts>::ResultTable_t &gradient, CalcPrecision_t step) {
  CalcPrecision_t norm_gradient = math::Pow<CalcPrecision_t, 1, 2>(
                                    fl::dense::ops::Dot(gradient.n_elements(),
                                                        gradient.ptr(),
                                                        gradient.ptr()));
  CalcPrecision_t feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);
  if (norm_gradient*step < grad_tolerance_
      ||  feasibility_error < desired_feasibility_error_) {
    return true;
  }
  return false;


}


template<typename TemplateOpts>
void MaxVariance<TemplateOpts>::Project(
  typename MaxVariance<TemplateOpts>::ResultTable_t *coordinates) {
  fl::optim::OptUtils::RemoveMean(coordinates);
}

template<typename TemplateOpts>
index_t  MaxVariance<TemplateOpts>::num_of_points() {
  return num_of_points_;
}

template<typename TemplateOpts>
void MaxVariance<TemplateOpts>::GiveInitMatrix(
  typename MaxVariance<TemplateOpts>::ResultTable_t *init_data) {
  init_data->Init(new_dimension_, num_of_points_);
  for (index_t i = 0; i < num_of_points_; i++) {
    for (index_t j = 0; j < new_dimension_ ; j++) {
      init_data->set(j, i, math::Random(0.0, 1.0));
    }
  }
}


///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
template<typename TemplateOpts>
void MaxFurthestNeighbors<TemplateOpts>::Init(
  typename MaxFurthestNeighbors<TemplateOpts>::MFNOpts &opts) {
  MaxVariance<TemplateOpts>::Init(
    static_cast<typename MaxVariance<TemplateOpts>::MVUOpts &>(opts));

  MaxVarianceUtils::ConsolidateNeighbors(*opts.from_tree_furhest_neighbors,
                                         *opts.from_tree_furthest_distances,
                                         1,
                                         1,
                                         &furthest_neighbor_pairs_,
                                         &furthest_distances_,
                                         &num_of_furthest_pairs_);
}

template<typename TemplateOpts>
void MaxFurthestNeighbors<TemplateOpts>::Destruct() {
  MaxVariance<TemplateOpts>::Destruct();
  furthest_neighbor_pairs_.clear();
  furthest_distances_.clear();
}

template<typename TemplateOpts>
void MaxFurthestNeighbors<TemplateOpts>::ComputeGradient(
  typename MaxFurthestNeighbors<TemplateOpts>::ResultTable_t &coordinates,
  typename MaxFurthestNeighbors<TemplateOpts>::ResultTable_t *gradient) {
  index_t dimension = coordinates.n_rows();
  gradient->SetAll(0.0);
  // objective
  CalcPrecision_t *a_i_r = new CalcPrecision_t[dimension];
  for (index_t i = 0; i < num_of_furthest_pairs_; i++) {
    index_t n1 = furthest_neighbor_pairs_[i].first;
    index_t n2 = furthest_neighbor_pairs_[i].second;
    CalcPrecision_t *point1 = coordinates.GetColumnPtr(n1);
    CalcPrecision_t *point2 = coordinates.GetColumnPtr(n2);
//    fl::dense::ops::SubOverwrite(dimension, point2, point1, a_i_r);
    fl::dense::ops::SubExpert(dimension, point2, point1, a_i_r);

    fl::dense::ops::AddExpert(dimension, CalcPrecision_t(-1.0), a_i_r,
                              gradient->GetColumnPtr(n1));
    fl::dense::ops::AddExpert(dimension, CalcPrecision_t(1.0),  a_i_r,
                              gradient->GetColumnPtr(n2));
  }
  // equality constraints
  for (index_t i = 0; i < Parent_t::num_of_nearest_pairs_; i++) {
    index_t n1 = Parent_t::nearest_neighbor_pairs_[i].first;
    index_t n2 = Parent_t::nearest_neighbor_pairs_[i].second;
    CalcPrecision_t *point1 = coordinates.GetColumnPtr(n1);
    CalcPrecision_t *point2 = coordinates.GetColumnPtr(n2);
    CalcPrecision_t dist_diff = fl::dense::ops::DistanceSqEuclidean(dimension, point1, point2)
                                - Parent_t::nearest_distances_[i];
//    fl::dense::ops::SubOverwrite(dimension, point2, point1, a_i_r);
    fl::dense::ops::SubExpert(dimension, point2, point1, a_i_r);

    fl::dense::ops::AddExpert(dimension,
                              -Parent_t::eq_lagrange_mult_[i] + dist_diff*Parent_t::sigma_,
                              a_i_r,
                              gradient->GetColumnPtr(n1));
    fl::dense::ops::AddExpert(dimension,
                              Parent_t::eq_lagrange_mult_[i] - dist_diff*Parent_t::sigma_,
                              a_i_r,
                              gradient->GetColumnPtr(n2));
  }
  delete[] a_i_r;
}

template<typename TemplateOpts>
void MaxFurthestNeighbors<TemplateOpts>::ComputeObjective(
  typename MaxFurthestNeighbors<TemplateOpts>::ResultTable_t &coordinates,
  typename MaxFurthestNeighbors<TemplateOpts>::CalcPrecision_t *objective) {
  *objective = 0;
  index_t dimension = coordinates.n_rows();
  for (index_t i = 0; i < num_of_furthest_pairs_; i++) {
    index_t n1 = furthest_neighbor_pairs_[i].first;
    index_t n2 = furthest_neighbor_pairs_[i].second;
    CalcPrecision_t *point1 = coordinates.GetColumnPtr(n1);
    CalcPrecision_t *point2 = coordinates.GetColumnPtr(n2);
    CalcPrecision_t diff = fl::dense::ops::DistanceSqEuclidean(dimension, point1, point2);
    *objective -= diff;
  }
}

template<typename TemplateOpts>
typename MaxFurthestNeighbors<TemplateOpts>::CalcPrecision_t
MaxFurthestNeighbors<TemplateOpts>::ComputeLagrangian(
  typename MaxFurthestNeighbors<TemplateOpts>::ResultTable_t &coordinates) {
  index_t dimension = coordinates.n_rows();
  CalcPrecision_t lagrangian = 0;
  ComputeObjective(coordinates, &lagrangian);
  for (index_t i = 0; i < Parent_t::num_of_nearest_pairs_; i++) {
    index_t n1 = Parent_t::nearest_neighbor_pairs_[i].first;
    index_t n2 = Parent_t::nearest_neighbor_pairs_[i].second;
    CalcPrecision_t *point1 = coordinates.GetColumnPtr(n1);
    CalcPrecision_t *point2 = coordinates.GetColumnPtr(n2);
    CalcPrecision_t dist_diff = fl::dense::ops::DistanceSqEuclidean(dimension, point1, point2)
                                - Parent_t::nearest_distances_[i];
    lagrangian += dist_diff * dist_diff * Parent_t::sigma_
                  -Parent_t::eq_lagrange_mult_[i] * dist_diff;
  }
  return lagrangian;
}


template<typename TemplateOpts>
bool MaxFurthestNeighbors<TemplateOpts>::IsDiverging(
  typename MaxFurthestNeighbors<TemplateOpts>::CalcPrecision_t objective) {
  if (objective < Parent_t::sum_of_furthest_distances_) {
    fl::logger->Message() << "objective("<< objective << ") < sum_of_furthest_distances ("<< 
      Parent_t::sum_of_furthest_distances_<< ")" <<std::endl;
    return true;
  }
  else {
    return false;
  }
}



///////////////////////////////////////////////////////////////
template < typename IndexContainerType,
typename DistanceContainerType,
typename IndexIndexContainerType >
void MaxVarianceUtils::ConsolidateNeighbors(IndexContainerType &from_tree_ind,
    DistanceContainerType  &from_tree_dist,
    index_t num_of_neighbors,
    index_t chosen_neighbors,
    IndexIndexContainerType *neighbor_pairs,
    DistanceContainerType *distances,
    index_t *num_of_pairs) {

  *num_of_pairs = 0;
  index_t num_of_points = from_tree_ind.size() / num_of_neighbors;
  neighbor_pairs->clear();
  distances->clear();
  bool skip = false;
  for (index_t i = 0; i < num_of_points; i++) {
    for (index_t k = 0; k < chosen_neighbors; k++) {
      index_t n1 = i;                       //neighbor 1
      index_t n2 = from_tree_ind[i*num_of_neighbors+k];  //neighbor 2
      if (n1 > n2) {
        for (index_t n = 0; n < chosen_neighbors; n++) {
          if (from_tree_ind[n2*num_of_neighbors+n] == n1) {
            skip = true;
            break;
          }
        }
      }
      if (skip == false) {
        *num_of_pairs += 1;
        neighbor_pairs->push_back(std::make_pair(n1, n2));
        distances->push_back(from_tree_dist[i*num_of_neighbors+k]);
      }
      skip = false;
    }
  }
}

template < typename IndexContainerType,
typename DistanceContainerType >
void MaxVarianceUtils::EstimateKnns(IndexContainerType &neares_neighbors,
                                    DistanceContainerType &nearest_distances,
                                    index_t maximum_knns,
                                    index_t num_of_points,
                                    index_t dimension,
                                    index_t *optimum_knns) {
  typedef typename DistanceContainerType::value_type CalcPrecision_t;
  CalcPrecision_t max_loocv_score = -std::numeric_limits<CalcPrecision_t>::max();
  CalcPrecision_t loocv_score = 0;
  //CalcPrecision_t unit_sphere_volume=math::SphereVolume(1.0, dimension);
  *optimum_knns = 0;
  for (index_t k = 2; k < maximum_knns; k++) {
    loocv_score = 0.0;
    CalcPrecision_t mean_band = 0.0;
    for (index_t i = 0; i < num_of_points; i++) {
//      CalcPrecision_t scale_factor=
//        pow(nearest_distances[i*maximum_knns+k], dimension/2);
      CalcPrecision_t probability = 0;
      for (index_t j = 0; j < k; j++) {
        probability += exp(-nearest_distances[i*maximum_knns+j]
                           / (2 * nearest_distances[i*maximum_knns+k]));
      }
      loocv_score += log(probability)
                     - (dimension / 2) * nearest_distances[i*maximum_knns+k]
                     - log(static_cast<CalcPrecision_t>(num_of_points));
      mean_band += nearest_distances[i*maximum_knns+k];
    }
    fl::logger->Message() << "Knn="<< k 
      << " mean_band="<< mean_band / num_of_points
      << " score="<< loocv_score<< ", dimension="<<dimension<<std::endl;
    if (loocv_score > max_loocv_score) {
      max_loocv_score = loocv_score;
      *optimum_knns = k;
    }
  }
}


}
} //namespaces

#endif
