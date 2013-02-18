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
#ifndef FL_LITE_FASTLIB_TREE_KDTREE_H
#define FL_LITE_FASTLIB_TREE_KDTREE_H

#include "fastlib/base/base.h"
#include "spacetree.h"
#include "bounds.h"
#include <vector>
#include <deque>
#include <algorithm>
#include "boost/type_traits/is_same.hpp"

namespace fl {
namespace tree {

/** @brief The generic midpoint split KD-tree.
 */
class MidpointKdTree {

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
          bound->Init(it.table());
          for (index_t d = 0;d < it.count(); d++) {
            Point_t point;
            it.get(d, &point);
            *bound |= point;
          }
        }

        template < typename MetricType,
        typename TreeIteratorType,
        typename TreeType >
        static bool Partition(MetricType &metric,
                              TreeIteratorType &it,
                              TreeType *node,
                              std::deque<bool> *membership) {

          typedef typename TreeIteratorType::Point_t Point_t;
          index_t split_dim = std::numeric_limits<index_t>::max();
          typedef typename TreeIteratorType::CalcPrecision_t CalcPrecision_t;
          CalcPrecision_t max_width = 0;

          for (index_t d = 0; d < it.table().n_attributes(); d++) {
            const CalcPrecision_t w =
              it.table().get_node_bound(node).get(d).width();

            if (unlikely(w > max_width)) {
              max_width = w;
              split_dim = d;
            }
          }
          if (max_width == 0) {
            // Jedi master says:
            //   "Zero width, all dimensions have; unsplittable they are."
            return false;
          }

          GenRange<CalcPrecision_t> split_dim_range =
            it.table().get_node_bound(node).get(split_dim);
          CalcPrecision_t split_value = split_dim_range.mid();

          membership->resize(it.count());
          bool has_members[2] = {false,false};
	  //#pragma omp parallel for		
	  //shared(membership)
          for (index_t i = 0; i < it.count(); i++) {
            Point_t point;
            it.get(i, &point);
            (*membership)[i] = (point[split_dim] < split_value);
            has_members[(point[split_dim] < split_value)] = true;
          }

          // "Fail to split, a dimension may still."
          return has_members[0] && has_members[1];
        }
    };
};


/** @brief The generic median split KD-tree.
 */
class MedianKdTree {
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
      private:
        template<typename PrecisionType>
        static int qsort_compar_(const void *a, const void *b) {
          PrecisionType *a_dbl = (PrecisionType *) a;
          PrecisionType *b_dbl = (PrecisionType *) b;

          if (*a_dbl < *b_dbl) {
            return -1;
          }
          else {
            if (*a_dbl > *b_dbl) {
              return 1;
            }
            else {
              return 0;
            }
          }
        }

        template<typename TreeIteratorType>
        static typename TreeIteratorType::CalcPrecision_t
        ChooseSplitValue_(TreeIteratorType &it,
                          int split_dim,
                          GenRange<typename TreeIteratorType::CalcPrecision_t> range) {

          typedef typename TreeIteratorType::Point_t Point_t;
          typedef typename TreeIteratorType::CalcPrecision_t CalcPrecision_t;
          fl::dense::Matrix<CalcPrecision_t, true> coordinate_vals;
          coordinate_vals.Init(it.count());

          index_t start = it.start();
	  //#pragma  omp for
          for (index_t i = 0; i < it.count(); i++) {
            Point_t v;
            it.get(i, &v);
            coordinate_vals[i] = v[split_dim];
          }


          // sort coordinate value
          std::nth_element(coordinate_vals.ptr(),
                           coordinate_vals.ptr() + it.count() / 2,
                           coordinate_vals.ptr() + it.count());

          CalcPrecision_t split_val =
            (CalcPrecision_t) coordinate_vals[it.count() / 2];
          if (split_val == range.lo ||
              split_val == range.hi) {
            split_val = range.mid();
          }
          return split_val;
        }

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
          // Reset the iterator.
          it.Reset();
          while (it.HasNext()) {
            index_t col_id;
            Point_t col;
            it.Next(&col, &col_id);
            *bound |= col;
          }
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
          index_t split_dim = std::numeric_limits<index_t>::max();
          CalcPrecision_t max_width = 0;

          for (index_t d = 0; d < it.table().n_attributes(); d++) {
            CalcPrecision_t w = it.table().get_node_bound(node).get(d).width();
            if (unlikely(w > max_width)) {
              max_width = w;
              split_dim = d;
            }
          }
          if (max_width == 0) {
            // Jedi master says:
            //   "Zero width, all dimensions have; unsplittable they are."
            return false;
          }

          GenRange<CalcPrecision_t> split_dim_range =
            it.table().get_node_bound(node).get(split_dim);
          CalcPrecision_t split_value =
            ChooseSplitValue_(it, split_dim, split_dim_range);
          if (split_dim_range.lo == split_value ||
              split_dim_range.hi == split_value) {
            // "Still fails to split node, split_val does."
            return false;
          }

          membership->resize(it.count());
	  //#pragma omp parallel for		
	  //shared(membership)			
	  //firstprivate(split_dim)		
	  //firstprivate(split_value)
          for (index_t i = 0; i < it.count(); i++) {
            Point_t point;
            it.get(i, &point);
            (*membership)[i] = (point[split_dim] < split_value);
          }
          return true;
        }
    };
};
} // tree namespace
} // fl namespace

#endif
