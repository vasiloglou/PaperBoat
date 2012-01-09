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
#ifndef MLPACK_ORTHO_RANGE_SEARCH_HYPER_RECTANGLE_TREE_H
#define MLPACK_ORTHO_RANGE_SEARCH_HYPER_RECTANGLE_TREE_H

#include "fastlib/dense/matrix.h"
#include "fastlib/la/linear_algebra.h"
#include "boost/tuple/tuple.hpp"
#include <vector>
#include <queue>

namespace fl {
namespace tree {

class HyperRectangleTree {

  public:

    static const bool is_binary = true;

    template<typename TableType, typename TreeType>
    static void ComputeLevel(TableType &table,
                             TreeType *node,
                             TreeType *node_parent,
                             index_t *node_level) {
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
        template < typename MetricType, typename TreeIteratorType,
        typename BoundType >
        static void FindBoundFromMatrix(MetricType &metric,
                                        TreeIteratorType &it,
                                        BoundType *bound) {

          // Here we assume that the BoundType is the same as
          // Point_t.

          // Typedefine the Point_t and Precision_t.
          typedef typename TreeIteratorType::Point_t Point_t;
          typedef typename Point_t::CalcPrecision_t Precision_t;

          // Reset the iterator so that it starts from the
          // beginning point.
          it.Reset();
          index_t limit = it.table().n_attributes() / 2;
          for (index_t i = 0; i < limit; i++) {
            (*bound)[i] = std::numeric_limits<Precision_t>::max();
            (*bound)[i + limit] = -std::numeric_limits<Precision_t>::max();
          }
          while (it.HasNext()) {
            index_t col_id;
            Point_t col;
            it.Next(&col, &col_id);

            // The first half of the point is for the lower bound
            // of the bounding box for the bounding boxes. The
            // second half of the point is for the upper bound of
            // the bounding box for the bounding boxes.
            for (index_t i = 0; i < limit; i++) {
              (*bound)[i] = std::min((*bound)[i], col[i]);
              (*bound)[i + limit] = std::max((*bound)[i + limit],
                                             col[i + limit]);
            }
          }
        }

        template < typename MetricType, typename TreeIteratorType,
        typename TreeType >
        static bool Partition(MetricType &metric,
                              TreeIteratorType &it,
                              TreeType *node,
                              std::deque<bool> *membership) {

          // We choose the widest dimension to be the splitting
          // dimension with the split value being the middle.
          typedef typename TreeIteratorType::Point_t Point_t;
          typedef typename Point_t::CalcPrecision_t CalcPrecision_t;
          index_t split_dim = std::numeric_limits<index_t>::max();
          const typename TreeType::Bound_t &bound = it.table().get_node_bound(node);

          CalcPrecision_t max_width = -1;
          index_t limit = bound.length() / 2;
          for (index_t d = 0; d < limit; d++) {
            CalcPrecision_t w = bound[d + limit] - bound[d];
            if (unlikely(w > max_width)) {
              max_width = w;
              split_dim = d;
            }
          }
          if (max_width <= std::numeric_limits<CalcPrecision_t>::min()) {
            return false;
          }

          // Split value is the midpoint.
          membership->resize(it.count());
          const typename Point_t::CalcPrecision_t split_value =
            (bound[split_dim + limit] + bound[split_dim]) * 0.5;

          // Iterate each hyperrectangle. If the lower limit is at
          // most the split value, then put it in the left.
          it.Reset();
          index_t left_count = 0;
          for (index_t left = 0; left < it.count(); left++) {
            index_t point_id;
            Point_t point;
            it.Next(&point, &point_id);
            (*membership)[left] = (point[split_dim] <= split_value);
            if ((*membership)[left]) {
              left_count++;
            }
          }

          // If the right child is empty, then do not split it.
          return left_count > 0 && left_count < it.count();
        }
    };
};
};
};

#endif
