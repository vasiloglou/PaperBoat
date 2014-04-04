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
#ifndef FL_LITE_FASTLIB_TREE_METRIC_TREE_H
#define FL_LITE_FASTLIB_TREE_METRIC_TREE_H

#include "fastlib/base/base.h"
#include "fastlib/dense/matrix.h"
#include "fastlib/metric_kernel/hellinger_metric.h"
#include "spacetree.h"
#include <vector>
#include <deque>

namespace fl {
namespace tree {

class ComputeCentroid {
  public:
    template<typename MetricType, typename TreeIteratorType, typename CenterType>
    static inline void Compute(
      MetricType &metric,
      TreeIteratorType &it,
      typename TreeIteratorType::CalcPrecision_t *radius,
      CenterType *center);

    template<typename MetricType, typename TreeIteratorType, typename PrecisionType>
    static inline void Compute(
      MetricType &metric,
      TreeIteratorType &it,
      typename TreeIteratorType::CalcPrecision_t *radius,
      fl::data::MonolithicPoint<PrecisionType> *center);
   
    template<typename TreeIteratorType, typename CenterType>
    static inline void Compute(
      fl::math::HellingerMetric &metric,
      TreeIteratorType &it,
      typename TreeIteratorType::CalcPrecision_t *radius,
      CenterType *center); 
};

/** @brief The generic ball-tree for any metric.
 */
class MetricTree {
  public:
    static const bool is_binary = true;
    template <typename TableType, typename TreeType>
    static void ComputeLevel(TableType & table,
                             TreeType * node,
                             TreeType * node_parent,
                             index_t * node_level) {
      if (node_parent != NULL) {
        index_t parent_level;
        node_parent->level(&parent_level);
        (*node_level) = parent_level - 1;
      }
      else {
        *node_level = 0;
      }
    }

    template<typename TableType, typename ContainerType1, typename ContainerType2>
    static void ComputeAverageRadiusPerLevel(TableType &table,
        typename TableType::Tree_t *node,
        ContainerType1 *radius_sum, ContainerType2 *num_of_nodes_per_level, index_t level) {
      if (num_of_nodes_per_level->size() == static_cast<size_t>(level)) {
        num_of_nodes_per_level->push_back(1);
        radius_sum->push_back(0);
      }
      else {
        (*num_of_nodes_per_level)[level] += 1;
        (*radius_sum)[level] += table.get_node_bound(node).radius();
      }
      if (!table.node_is_leaf(node)) {
        level++;
        ComputeAverageRadiusPerLevel(table, table.get_node_left_child(node),
                                     radius_sum, num_of_nodes_per_level, level);
        ComputeAverageRadiusPerLevel(table, table.get_node_right_child(node),
                                     radius_sum, num_of_nodes_per_level, level);
      }
    }

    class SplitRule {
      public:
        template < typename TreeIteratorType,
        typename MetricType,
        typename PointType1,
        typename PointType2 >
        static void FurthestPoint(
          const MetricType &metric,
          const PointType1 &pivot,
          TreeIteratorType &it,
          typename TreeIteratorType::CalcPrecision_t *furthest_distance,
          PointType2 *furthest_point) {

          *furthest_distance = -1.0;
          typename TreeIteratorType::CalcPrecision_t
             distance_between_center_and_point;
          // Reset the iterator.
          it.Reset();
          PointType2 point;
          while (it.HasNext()) {
            index_t point_id;
            it.Next(&point, &point_id);
             //  the order of the arguments in the Distance function
             //  is critical, don't change it, if you do then
             //  it will not work for assymetric divergences
            distance_between_center_and_point =
              metric.Distance(point, pivot);
            if ((*furthest_distance) <
                distance_between_center_and_point) {
              *furthest_distance = distance_between_center_and_point;
              furthest_point->Alias(point);
            }
          }
          DEBUG_ASSERT(*furthest_distance!=-1);
        }

        template < typename MetricType,
        typename TreeIteratorType,
        typename BoundType >
        static void FindBoundFromMatrix(MetricType &metric,
                                        TreeIteratorType & it,
                                        BoundType * bound) {
          typedef typename BoundType::Point_t Point_t;
          typedef typename Point_t::CalcPrecision_t CalcPrecision_t;
          CalcPrecision_t radius;
          ComputeCentroid::Compute(metric, it, &radius, &bound->center());
          bound->PostProcessCenter(metric);
          bound->set_radius(radius);
        }

        template < typename MetricType,
        typename TreeIteratorType,
        typename TreeType >
        static bool Partition(MetricType &metric,
                              TreeIteratorType &it,
                              TreeType *node,
                              std::deque<bool> *membership) {

          typedef typename TreeIteratorType::Point_t Point_t;
          typedef typename Point_t::CalcPrecision_t CalcPrecision_t;
          // Pick a random row.
          Point_t random_row_vec;
          it.RandomPick(&random_row_vec);

          // Now figure out the furthest point from the random row picked
          // above.
          CalcPrecision_t furthest_distance;
          Point_t furthest_from_random_row_vec;
          FurthestPoint(metric, random_row_vec, it, &furthest_distance,
                        &furthest_from_random_row_vec);

          // Then figure out the furthest point from the furthest point.
          CalcPrecision_t furthest_from_furthest_distance;
          Point_t furthest_from_furthest_random_row_vec;
          FurthestPoint(metric, furthest_from_random_row_vec, it,
                        &furthest_from_furthest_distance,
                        &furthest_from_furthest_random_row_vec);
          if (furthest_from_furthest_distance <=
              std::numeric_limits <CalcPrecision_t>::min()) {
            return false;
          }
          // Compute the distance to the two pivots, and determine the
          // membership.
          membership->resize(it.count());

          Point_t point;

          // Reset the iterator.
          it.Reset();
          int left_count = 0;
          for (index_t left = 0; left < it.count(); left++) {
            DEBUG_ASSERT(it.HasNext());
            // Make alias of the current point.
            index_t point_id;
            it.Next(&point, &point_id);

            // Compute the distances from the two pivots.
            CalcPrecision_t distance_from_left_pivot =
              metric.Distance(point, furthest_from_random_row_vec);
            CalcPrecision_t distance_from_right_pivot =
              metric.Distance(point,
                              furthest_from_furthest_random_row_vec);

            // Set the boolean vector indicating whether the point is
            // closer to the left pivot than the right pivot.
            (*membership)[left] =
              (distance_from_left_pivot <= distance_from_right_pivot);
            if ((*membership)[left]) {
              left_count++;
            }
          }
          return 0 < left_count && left_count < it.count();
        }
    };
};

template<typename MetricType, typename TreeIteratorType, typename PrecisionType>
void ComputeCentroid::Compute(
  MetricType &metric,
  TreeIteratorType &it,
  typename TreeIteratorType::CalcPrecision_t *radius,
  fl::data::MonolithicPoint<PrecisionType> *center) {

  typedef typename TreeIteratorType::Point_t Point_t;
  typedef typename TreeIteratorType::CalcPrecision_t CalcPrecision_t;
  typedef fl::data::MonolithicPoint<PrecisionType> CenterType;
  // Reset the iterator.
  it.Reset();
  typename TreeIteratorType::Point_t col;
  index_t col_id;
  it.Next(&col, &col_id);
  Point_t random_point;
  it.RandomPick(&random_point);
  center->Copy(random_point);
  center->SetAll(0);
  it.Reset();
  while (it.HasNext()) {
    it.Next(&col, &col_id);
    fl::la::AddTo(col, center);
  }
  fl::la::SelfScale(1.0 / ((CalcPrecision_t)it.count()), center);
//          int num_of_zeros=0;
//          for(index_t i=0; i<center->size(); i++) {
//            if ((*center)[i]==0) {
//             num_of_zeros+=1;
//            }
//          }
//          NOTIFY("size:%u, zeros:%i, %g %% ", center->size(), num_of_zeros, 100.0*num_of_zeros/center->size());
  // we only need the radius so we just put a dummy point
  Point_t dummy;
  MetricTree::SplitRule::FurthestPoint(metric, *center,
                                       it, radius, &dummy);
}

template<typename MetricType, typename TreeIteratorType, typename CenterType>
void ComputeCentroid::Compute(
  MetricType &metric,
  TreeIteratorType &it,
  typename TreeIteratorType::CalcPrecision_t *radius,
  CenterType *center) {

  // Compute the centroid as usual
  typedef typename TreeIteratorType::Point_t Point_t;
  typedef typename TreeIteratorType::CalcPrecision_t CalcPrecision_t;
  fl::data::MonolithicPoint<CalcPrecision_t> dummy_center;
  // Reset the iterator.
  it.Reset();
  typename TreeIteratorType::Point_t col;

  index_t col_id;
  it.Next(&col, &col_id);
  CenterType random_point;
  it.RandomPick(&random_point);
  dummy_center.Copy(random_point);
  dummy_center.SetAll(0);
  it.Reset();
  while (it.HasNext()) {
    it.Next(&col, &col_id);
    fl::la::AddTo(col, &dummy_center);
  }
  fl::la::SelfScale(1.0 / ((CalcPrecision_t)it.count()), &dummy_center);
  // Now find the point from the set that is closest to that centroid
  it.Reset();
  CalcPrecision_t best_distance=std::numeric_limits<CalcPrecision_t>::max();
  Point_t dummy_center1;
  while (it.HasNext()) {
    it.Next(&col, &col_id);
    CalcPrecision_t dist=metric.DistanceSq(dummy_center, col);
    if (dist<best_distance) {
      best_distance=dist;
      dummy_center1.Alias(col);
    }
  }
  center->Copy(dummy_center1);
  // we only need the radius so we just put a dummy point
  Point_t dummy;
  MetricTree::SplitRule::FurthestPoint(metric, *center,
                                       it, radius, &dummy);

}

template<typename TreeIteratorType, typename CenterType>
void ComputeCentroid::Compute(
  fl::math::HellingerMetric &metric,
  TreeIteratorType &it,
  typename TreeIteratorType::CalcPrecision_t *radius,
  CenterType *center) {

  // Compute the centroid as usual
  typedef typename TreeIteratorType::Point_t Point_t;
  typedef typename TreeIteratorType::CalcPrecision_t CalcPrecision_t;
  fl::data::MonolithicPoint<CalcPrecision_t> dummy_center;
  // Reset the iterator.
  it.Reset();
  typename TreeIteratorType::Point_t col;

  index_t col_id;
  it.Next(&col, &col_id);
  CenterType random_point;
  it.RandomPick(&random_point);
  dummy_center.Copy(random_point);
  dummy_center.SetAll(0);
  it.Reset();
  while (it.HasNext()) {
    it.Next(&col, &col_id);
    fl::la::AddTo(col, &dummy_center);
  }
  CalcPrecision_t l2=fl::la::LengthEuclidean(dummy_center);
  fl::la::SelfScale(1.0 / (l2), &dummy_center);
  // Now find the point from the set that is closest to that centroid
  it.Reset();
  CalcPrecision_t best_distance=std::numeric_limits<CalcPrecision_t>::max();
  Point_t dummy_center1;
  while (it.HasNext()) {
    it.Next(&col, &col_id);
    CalcPrecision_t dist=metric.DistanceSq(dummy_center, col);
    if (dist<best_distance) {
      best_distance=dist;
      dummy_center1.Copy(col);
    }
  }

  center->Copy(dummy_center1);
  // we only need the radius so we just put a dummy point
  Point_t dummy;
  MetricTree::SplitRule::FurthestPoint(metric, *center,
                                       it, radius, &dummy);
}

}    // tree namespace
}    // fl namespace

#endif
