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
#ifndef FL_LITE_FASTLIB_TREE_COSINE_TREE_H
#define FL_LITE_FASTLIB_TREE_COSINE_TREE_H

#include "fastlib/la/linear_algebra.h"
#include "fastlib/dense/matrix.h"
#include "fastlib/metric_kernel/abstract_metric.h"
#include <vector>
#include <deque>
#include <iostream>

namespace fl {
namespace tree {

template<typename PointType>
class CosineBound {

  private:
    PointType mean_vector_;

    typename PointType::CalcPrecision_t frobenius_norm_sum_;

    typename PointType::CalcPrecision_t max_distance_within_bound_;

  public:
    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & mean_vector_;
    }

   template<typename StreamType>
   void Print(StreamType &stream, const std::string &delim) const {
      stream << "*CosineBound*" << "\n";
      stream << "mean:";
      for (index_t i = 0; i < mean_vector_.size(); i++) {
        stream << mean_vector_[i] << delim;
      }
      stream << "\n";
      stream << "frob norm sum:" << frobenius_norm_sum_ << "\n";
      stream << "max dist within:" << max_distance_within_bound_ << "\n";
   }

    typename PointType::CalcPrecision_t frobenius_norm_sum() const {
      return frobenius_norm_sum_;
    }

    const PointType &mean() const {
      return mean_vector_;
    }

    PointType &mean() {
      return mean_vector_;
    }

    void AddTo(const PointType &v) {
      la::AddTo(v, &mean_vector_);
      frobenius_norm_sum_ += fl::la::Dot(v, v);
    }

    template<typename TableType>
    void Init(TableType &table) {
      typename TableType::Point_t point;
      table.get(0, &point);
      mean_vector_.Copy(point);
      mean_vector_.SetAll(0);
    }

    void Scale(index_t count) {
      fl::la::SelfScale((typename PointType::CalcPrecision_t)
                        (1.0 /
                         ((typename PointType::CalcPrecision_t)
                          count)),
                        &mean_vector_);
    }

    void SetZero() {
      mean_vector_.SetAll(0);
      frobenius_norm_sum_ = 0;
      max_distance_within_bound_ = 0;
    }

    void set_max_distance_within_bound(
      typename PointType::CalcPrecision_t max_distance_within_bound_in) {
      
      max_distance_within_bound_ = max_distance_within_bound_in;
    }

    typename PointType::CalcPrecision_t MaxDistanceWithinBound() const {
      return max_distance_within_bound_;
    }

    /** Calculates the midpoint of the range, cetroid must be initialized */
    template<typename TPointType>
    void CalculateMidpoint(TPointType *centroid) const {
      fl::logger->Die() << "Don't use this";
    }

    /** Calculates the midpoint of the range */
    template<typename TPointType>
    void CalculateMidpointOverwrite(TPointType *centroid) const {
      for (index_t i = 0; i < mean_vector_.size(); i++) {
        (*centroid)[i] = mean_vector_[i];
      }
    }
};


/** @brief The generic similarity tree.
 */
class SimilarityTree {

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
        template < typename TreeIteratorType,
        typename TreeType >
        static void PickAccordingToLengthSquareDistribution
        (TreeIteratorType &it,
         TreeType *node,
         typename TreeIteratorType::Point_t *random_row_vec) {

          typedef typename TreeIteratorType::Point_t Point_t;
          typedef typename TreeIteratorType::Point_t::CalcPrecision_t
          CalcPrecision_t;
          std::vector<CalcPrecision_t> length_square_distribution;
          length_square_distribution.resize(it.count());

          it.Reset();
          index_t current_index = 0;
          do {
            Point_t point;
            index_t point_id;
            it.Next(&point, &point_id);
            if (current_index == 0) {
              length_square_distribution[current_index] =
                fl::la::Dot(point, point);
            }
            else {
              length_square_distribution[current_index] =
                length_square_distribution[current_index - 1] +
                fl::la::Dot(point, point);
            }
            current_index++;
          }
          while (it.HasNext());

          // Reset the iterator.
          it.Reset();
          current_index = 0;

          // Generate a random number.
          CalcPrecision_t random_number =
            fl::math::Random(0.0, length_square_distribution[it.count() - 1]);

          do {
            index_t point_index;
            it.Next(random_row_vec, &point_index);
            if (random_number <=
                length_square_distribution[current_index]) {
              break;
            }
            current_index++;
          }
          while (it.HasNext());
        }

      public:
        template < typename MetricType, typename TreeIteratorType,
        typename BoundType >
        static void FindBoundFromMatrix(MetricType &metric,
                                        TreeIteratorType &it,
                                        BoundType *bound) {

          // Compute the mean vector.
          typename TreeIteratorType::Point_t col;
          bound->SetZero();
          it.Reset();
          while (it.HasNext()) {
            index_t col_id;
            it.Next(&col, &col_id);
            bound->AddTo(col);
          }
          bound->Scale(it.count());
        }

        template < typename MetricType, typename TreeIteratorType,
        typename TreeType >
        static bool Partition(MetricType &metric,
                              TreeIteratorType &it,
                              TreeType *node,
                              std::deque<bool> *membership) {

          typedef typename TreeIteratorType::Point_t Point_t;
          typedef typename Point_t::CalcPrecision_t CalcPrecision_t;
          // Pick a random row according to length square distribution.
          Point_t random_row_vec;
          PickAccordingToLengthSquareDistribution(it, node,
                                                  &random_row_vec);

          // Now find the minimum/maximum absolute values of
          // similarities.
          CalcPrecision_t min_abs_similarity =
            std::numeric_limits<CalcPrecision_t>::max();
          CalcPrecision_t max_abs_similarity = 0;

          // Temporary variables.
          Point_t point;
          index_t point_id;
          std::vector<CalcPrecision_t> similarities;
          similarities.resize(it.count());

          // Reset the iterator.
          it.Reset();
          for (int cur_index = 0; it.HasNext(); cur_index++) {
            it.Next(&point, &point_id);
            // Compute the similarity with the chosen pivot.
            CalcPrecision_t abs_similarity =
              metric.Distance(random_row_vec, point);
            abs_similarity = (abs_similarity >= 0) ?
                             abs_similarity : (-abs_similarity);
            similarities[cur_index] = abs_similarity;

            min_abs_similarity = std::min(min_abs_similarity,
                                          abs_similarity);
            max_abs_similarity = std::max(max_abs_similarity,
                                          abs_similarity);
          }

          if (max_abs_similarity - min_abs_similarity <=
              std::numeric_limits<CalcPrecision_t>::min()) {
            return false;
          }

          // Compute the distance to the two pivots, and determine the
          // membership.
          membership->resize(it.count());

          index_t left_count = 0;
          for (index_t left = 0; left < it.count(); left++) {
            // Set the boolean vector indicating whether the point is
            // closer to the left pivot than the right pivot.
            (*membership)[left] =
              ((max_abs_similarity - similarities[left]) <=
               (similarities[left] - min_abs_similarity));
            if ((*membership)[left]) {
              left_count++;
            }
          }
          return left_count < it.count() && left_count > 0;
        }
    };
};
};
};

#endif
