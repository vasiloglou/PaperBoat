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
#ifndef FASTLIB_TREE_COVER_TREE_H
#define FASTLIB_TREE_COVER_TREE_H

#include "fastlib/fastlib.h"
#include "spacetree.h"
#include "bounds.h"

#include <vector>

namespace fl {
namespace tree {
/** @brief The generic cover tree for L_p metric.
 */
template < typename StoragePrecision, typename CalcPrecision,
typename Metric, int BaseNominator, int BaseDenominator >
class CoverTree {

  public:

    typedef StoragePrecision StoragePrecision_t;

    typedef StoragePrecision Precision_t;

    typedef CalcPrecision CalcPrecision_t;

    typedef Metric Metric_t;

    static const bool is_binary = false;

    static const bool store_level = true;

    typedef fl::tree::DBallBound < StoragePrecision, CalcPrecision, Metric_t,
    fl::la::GenMatrix<StoragePrecision, true> > Bound_t;

    static const CalcPrecision_t base =
      1.0*BaseNominator / BaseDenominator;

  private:

    static inline CalcPrecision_t inverse_log_base_() {
      CalcPrecision_t inverse_log_base  = 1.0 / log(base);
      return inverse_log_base;
    }

    static inline index_t scale_of_distance_(CalcPrecision_t distance) {
      return (index_t) ceil(log(distance) * inverse_log_base_());
    }

  public:

    template<typename TableType, typename TreeType>
    static void ComputeLevel(TableType &table,
                             const TreeType *node,
                             const TreeType *node_parent,
                             index_t *node_level) {
      if (node_parent != NULL) {
        index_t parent_level;
        node_parent->level(&parent_level);
        (*node_level) = std::min(parent_level - 1,
                                 scale_of_distance_(table.get_node_bound(node).radius()));
      }
      else {
        *node_level = scale_of_distance_(table.get_node_bound(node).radius());
      }
    }

    class SplitRule {

      private:
        template<typename TreeIterator>
        static void FurthestPoint_(
          const fl::la::GenMatrix<StoragePrecision, true> &pivot,
          TreeIterator &it,
          CalcPrecision *furthest_distance,
          fl::la::GenMatrix<StoragePrecision, true> *furthest_point) {

          *furthest_distance = -1.0;

          fl::la::GenMatrix<StoragePrecision, true> point;
          it.Reset();
          while (it.HasNext()) {
            index_t point_id;
            it.Next(&point, &point_id);
            CalcPrecision distance_between_center_and_point =
              Metric_t::Distance(pivot, point);

            if ((*furthest_distance) < distance_between_center_and_point) {
              *furthest_distance = distance_between_center_and_point;
              furthest_point->Alias(point);
            }
          }
        }

      public:

        template<typename TreeIterator>
        static void FindBoundFromMatrix(
          TreeIterator &it,
          Bound_t *bound) {

          // Choose the first point in the list as the pivot.
          fl::la::GenMatrix<StoragePrecision, true> first_vector;
          it.Reset();
          index_t point_id;
          it.Next(&first_vector, &point_id);
          bound->center().CopyValues(first_vector);

          // Compute the furthest distance.
          CalcPrecision furthest_distance;
          fl::la::GenMatrix<StoragePrecision, true> furthest_point;
          FurthestPoint_(bound->center(), it,
                         &furthest_distance, &furthest_point);
          bound->set_radius(furthest_distance);
        }

        template<typename TreeIterator, typename TreeType>
        static bool Partition(
          TreeIterator &it,
          TreeType *node,
          std::vector<bool> *membership) {

          if (it.table().get_node_bound(node).radius() <
              std::numeric_limits<typename TreeType::Precision_t>::max()) {
            return false;
          }

          // Pick the first row in the list for the first child (for
          // self-branch), and otherwise a random row as the pivot and
          // swap it with the first point in the list.
          fl::la::GenMatrix<StoragePrecision, true> random_row_vec;
          it.RandomPick(&random_row_vec);
          fl::la::GenMatrix<StoragePrecision, true> first_row_vec;
          it.Reset();
          index_t point_id;
          it.Next(&first_row_vec, &point_id);

          if (random_row_vec.ptr() != first_row_vec.ptr()) {
            first_row_vec.SwapValues(&random_row_vec);
          }

          // Now, the first vector in the list should hold the pivot.
          index_t node_level;

          // I need to somehow change the Partition function definition to get
          // the level of the node.

          // node->level(&node_level);
          CalcPrecision distance_limit = pow(base, node_level - 1);

          // Initialize the membership variable.
          membership->resize(it.count());

          // The distance limit is BASE^(parent scale - 1)
          index_t true_count = 0;

          fl::la::GenMatrix<StoragePrecision, true> point;
          it.Reset();
          for (index_t left = 0; left < it.count(); left++) {

            DEBUG_ASSERT(it.HasNext());
            // Make alias of the current point.
            index_t point_id;
            it.Next(&point, &point_id);
            // Compute the distances from the two pivots.
            CalcPrecision distance_from_pivot =
              Metric_t::Distance(first_row_vec, point);

            // Set the boolean vector indicating whether the point is
            // closer to the left pivot than the right pivot.
            (*membership)[left] = (distance_from_pivot <=
                                   distance_limit);

            if ((*membership)[left]) {
              true_count++;
            }
          }
          return true;
        }
    };
};

}; // tree namespace
}; // fl namespace

#endif
