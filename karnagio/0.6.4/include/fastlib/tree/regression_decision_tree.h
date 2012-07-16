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
#ifndef FL_LITE_FASTLIB_TREE_REGRESSION_DECISION_TREE_H
#define FL_LITE_FASTLIB_TREE_REGRESSION_DECISION_TREE_H

#include "fastlib/base/base.h"
#include "spacetree.h"
#include "bounds.h"
#include <vector>
#include <deque>
#include <algorithm>
#include "boost/type_traits/is_same.hpp"

namespace fl {
namespace tree {

template < typename StoragePrecision = double,
typename CalcPrecision = double,
int t_pow = 2 >
class RDTreeGenHrectBound : public GenHrectBound < StoragePrecision,
      CalcPrecision, pow > {
  public:
    RDTreeGenHrectBound() : mean_target_value_(0) {
    }
    StoragePrecision &mean_target_value() {
      return mean_target_value_;
    }
    const StoragePrecision &mean_target_value() {
      return mean_target_value_;
    }
  private:
    StoragePrecision mean_target_value_;

};

/** @brief Classification Decision Tree  KD-tree.
 */
class RegressionnDecisionTree {

  public:
    /** @brief Computes the label of the given node given its
     *         parent.
     *
     *  @param table The table.
     *  @param node The current node.
     *  @param node_parent The parent of the current node.
     *  @param node_level The output of the computed level.
     */
    static const bool is_binary = true;
    template<typename TableType, typename TreeType>
    static void ComputeLevel(TableType &table,
                             TreeType *node,
                             TreeType *node_parent,
                             index_t *node_level) {

      // If the parent node is present, just set the node level to
      // one less than the level of the parent. Otherwise, set it to
      // zero.
      if (node_parent != NULL) {
        index_t parent_level;
        node_parent->level(&parent_level);
        (*node_level) = parent_level - 1;
      }
      else {
        *node_level = 0;
      }
    }

    class SplitRule {
      public:
        template < typename MetricType,
        typename TreeIteratorType,
        typename BoundType >
        static void FindBoundFromMatrix(MetricType &metric,
                                        TreeIteratorType &it,
                                        BoundType *bound) {
          if (!boost::is_same<MetricType, fl::math::LMetric<2> >::value) {
            fl::logger->Die() << "This kd-tree supports only L2 metric";
          }
          typedef typename TreeIteratorType::Point_t Point_t;
          // Reset the iterator so that it starts from the beginning
          // point.
          it.Reset();
          while (it.HasNext()) {
            index_t col_id;
            Point_t col;
            it.Next(&col, &col_id);
            *bound |= col;
            bound->mean_target_value() += col.meta_data().get<2>();
          }
          bound->mean_target_value() /= it.count();
        }

        template < typename MetricType,
        typename TreeIteratorType,
        typename TreeType >
        static bool Partition(MetricType &metric,
                              TreeIteratorType &it,
                              TreeType *node,
                              std::deque<bool> *membership) {

          // we need to do this check in case we have garbage
          CalcPrecision_t max_width = std::numeric_limit<CalcPrecision_t>::min();
          for (index_t d = 0; d < it.table().n_attributes(); d++) {
            const CalcPrecision_t w =
              it.table().get_node_bound(node).get(d).width();

            if (unlikely(w > max_width)) {
              max_width = w;
            }
          }

          if (max_width <= std::numeric_limits<CalcPrecision_t>::min()) {
            return false;
          }


          typedef typename TreeIteratorType::Point_t Point_t;
          typedef typename Point_t::CalcPrecision_t CalcPrecision_t;
          index_t num_points = it.count();
          std::vector<std::pair<CalcPrecision_t, index_t > > dimension(num_points);
          std::vector<CalcPrecision_t> cumulative_targets_up(num_points);
          std::vector<CalcPrecision_t> cumulative_targets_down(num_points);
          CalcPrecision_t global_min_mse = std::numeric_limits<CalcPrecision_t>::max();
          index_t split_dim = std::numeric_limits<index_t>::max();
          CalcPrecision_t global_split_value = std::numeric_limits<CalcPrecision_t>::max();
          for (index_t d = 0; d < it.table().n_attributes(); d++) {
            // put the d dimension of every point along with the point_id
            // in a vector
            it.Reset();
            index_t count = 0;
            index_t point_id;
            Point_t point;
            while (it.HasNext()) {
              index_t point_id;
              Point_t point;
              it.Next(&point, &point_id);
              dimension[count].first = point.get(d);
              dimension[count].second = point_id;
              ++count;
            }
            // sort the points according to the d-dimension
            std::sort(dimension.begin(), dimension.end());
            it.get(dimensions.begin()->second, &point);
            // compute the cummulative sum of the targets in an ascending order
            cummulative_targets_up[0] = point.meta_data().get<2>();
            // compute the cummulative sum of the targets in an descending order
            it.get(dimensions.end->second, &point);
            cummulative_targets_down[0] = point.meta_data().get<2>();

            for (index_t i = 1; i < targets.size(); i++) {
              it.get(dimensions[i].second, &point);
              cummulative_targets_up[i] =
                cummulative_targets[i-1] + point.meta_data().get<2>();
              it.get(dimensions[dimensions.size()-i-1].second, &point);
              cummulative_targets_down[i] =
                cummulative_targets_down[i-1] + point.meta_data().get<2>();
            }
            // now find the best split for this dimension
            CalcPrecision_t min_mse = std::numeric_limits<CalcPrecision_t>::max();
            CalcPrecision_t split_value = std::numeric_limits<CalcPrecision_t>::max();
            for (index_t i = 0; i < cummulative_targets_up.size(); i++) {
              CalcPrecision_t mse = 1.0 * cummulative_targets_up[i] / (i + 1)
                                    + 1.0 * cummulative_targets_down[num_points-i-1] / (num_points - i);
              if (mse < min_mse) {
                min_mse = mse;
                split_value = dimensions[i].first;
              }
            }
            if (min_mse < global_min_mse) {
              global_min_mse = min_mse;
              split_dimension = d;
              global_split_value = split_value;
            }
          }
          membership->resize(it.count());
          for (index_t i = 0; i < it.count(); i++) {
            index_t point_id;
            Point_t point;
            DEBUG_ASSERT(it.HasNext());
            it.Next(&point, &point_id);
            (*membership)[i] = (point[split_dimension] <= global_split_value);
          }
          return true;
        }
    };
};
}; // tree namespace
}; // fl namespace

#endif
