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
#ifndef FL_LITE_FASTLIB_TREE_CLASSIFICATION_DECISION_TREE_PARTITION_H
#define FL_LITE_FASTLIB_TREE_CLASSIFICATION_DECISION_TREE_PARTITION_H

#include "fastlib/base/base.h"
#include "fastlib/tree/spacetree.h"
#include "fastlib/tree/bounds.h"
#include "fastlib/tree/cart_impurity.h"
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
#include "boost/type_traits/is_same.hpp"

namespace fl {
namespace tree {

class ComputeImprovement {
  public:

    template < typename MapSplitValuesType, typename BoundType,
    typename CalcPrecision_t >
    ComputeImprovement(MapSplitValuesType &split_values,
                       const BoundType &left_bound,
                       const BoundType &right_bound,
                       int total_num_points,
                       CalcPrecision_t *delta_impurity) {
      CalcPrecision_t p_right =
        ((CalcPrecision_t) right_bound.n_points()) /
        ((CalcPrecision_t) total_num_points);
      CalcPrecision_t p_left  =
        ((CalcPrecision_t) left_bound.n_points()) /
        ((CalcPrecision_t) total_num_points);
      (*delta_impurity) =
        (split_values.iterator()->table().get_node_bound(
           split_values.node())).impurity(
          *(split_values.impurity()))
        - p_left * left_bound.impurity(*(split_values.impurity()))
        - p_right * right_bound.impurity(*(split_values.impurity()));
    }
};

class SortDimension {
  private:

    template<typename TreeIteratorType>
    class Comparator {
      private:
        TreeIteratorType *it_;

        int *current_dimension_;

      public:
        void Init(TreeIteratorType &it_in, int *current_dimension_in) {
          it_ = &it_in;
          current_dimension_ = current_dimension_in;
        }

        bool operator()(int first_point_index, int second_point_index) {
          typename TreeIteratorType::Point_t first_point, second_point;
          it_->table().get(first_point_index, &first_point);
          it_->table().get(second_point_index, &second_point);

          return (first_point[*current_dimension_] <
                  second_point[*current_dimension_]) ||
                 (first_point[*current_dimension_] ==
                  second_point[*current_dimension_] &&
                  first_point_index < second_point_index);
        }
    };

  public:
    template<typename MapSplitValuesType>
    SortDimension(MapSplitValuesType &split_values,
                  std::vector<int> *sorted_indices) {

      sorted_indices->resize(
        split_values.iterator()->table().get_node_count(split_values.node()));
      split_values.iterator()->Reset();
      for (int i = 0;
           i < split_values.iterator()->table().get_node_count(
             split_values.node()); i++) {
        typename MapSplitValuesType::TreeIteratorT::Point_t point;
        index_t point_id;
        split_values.iterator()->Next(&point, &point_id);
        (*sorted_indices)[i] = point_id;
      }
      Comparator<typename MapSplitValuesType::TreeIteratorT> comp;
      comp.Init(*(split_values.iterator()), split_values.current_dimension());
      std::sort(sorted_indices->begin(), sorted_indices->end(), comp);
    }
};

class PartitionNumericDimension {
  public:

    template<typename TreeIteratorType, typename TreeType, typename T>
    PartitionNumericDimension(TreeIteratorType &it, TreeType *node,
                              const std::pair<int, T> &split_values,
                              std::deque<bool> *membership) {

      // Allocate the membership vector.
      membership->resize(it.count());

      typename TreeIteratorType::Point_t point;
      index_t point_index;
      it.Reset();

      for (int i = 0; it.HasNext(); i++) {
        it.Next(&point, &point_index);
        (*membership)[i] = (point[split_values.first] < split_values.second);
      }
    }

    template<typename MapSplitValuesType, typename T, typename CalcPrecision_t>
    PartitionNumericDimension(MapSplitValuesType &split_values,
                              T *split_value, CalcPrecision_t *best_gain) {

      // Sort the dimension in the increasing order.
      std::vector<int> sorted_indices;
      SortDimension(split_values, &sorted_indices);

      // Initially put everything to the right node.
      typename MapSplitValuesType::TreeT::Bound_t left_bound;
      typename MapSplitValuesType::TreeT::Bound_t right_bound(
        *(split_values.iterator()));

      // Given the sorted indices of points, iterate and generate two
      // partitions. Find the best partioning point.
      typename MapSplitValuesType::TreeIteratorT::Point_t point;
      split_values.iterator()->table().get(sorted_indices[0], &point);

      // Every instance whose attribute values is at least the value
      // of current_split_val is put to the right side.
      T previous_instance_value = point[*(split_values.current_dimension())];
      *split_value = previous_instance_value;
      *best_gain = - std::numeric_limits<CalcPrecision_t>::max();

      // The total number of points we are splitting.
      int total_num_points = split_values.iterator()->count();

      for (int i = 0; i < sorted_indices.size(); i++) {

        // Get the current point.
        split_values.iterator()->table().get(sorted_indices[i], &point);

        // If the attribute value of the current point is above the
        // current split point, compute the current improvement and
        // update the best one.
        if (point[*(split_values.current_dimension())] >
            previous_instance_value) {
          CalcPrecision_t delta_impurity;
          ComputeImprovement(split_values, left_bound, right_bound,
                             total_num_points, &delta_impurity);

          if (delta_impurity > *best_gain) {
            *split_value = 0.5 * (point[*(split_values.current_dimension())] +
                                  previous_instance_value);
            *best_gain = delta_impurity;
          }
        }

        // Remove the current instance from the right bound and put it
        // into the left bound.
        previous_instance_value = point[*(split_values.current_dimension())];
        left_bound |= point;
        right_bound -= point;
      }
    }
};

template<typename T>
class PartitionNominalDimension {
  private:

    template<typename SortPrecision_t>
    class IndexSorter {

      private:
        const std::vector<SortPrecision_t> *sort_according_to_this_;

      public:

        void Init(
          const std::vector<SortPrecision_t> &sort_according_to_this_in) {
          sort_according_to_this_ = &sort_according_to_this_in;
        }

        bool operator()(int first, int second) {
          return (*sort_according_to_this_)[first] <
                 (*sort_according_to_this_)[second];
        }
    };

    void ConvertAttributeValueToString_(T to_be_converted,
                                        std::string *converted) {
      std::stringstream ss;
      ss << to_be_converted;
      ss >> *converted;
    }

    template < typename MapSplitValuesType,
    typename BoundType,
    typename TreeIteratorType,
    typename CalcPrecision_t >
    void Partition_(MapSplitValuesType &split_values,
                    const std::string &trial_split_values,
                    int current_dimension,
                    BoundType &left_bound, BoundType &right_bound,
                    TreeIteratorType &it,
                    std::string *split_value, CalcPrecision_t *best_gain) {

      typename TreeIteratorType::Point_t point;
      index_t point_id;

      // Loop through each point and group.
      for (it.Reset(); it.HasNext();) {
        it.Next(&point, &point_id);

        // Conversion of the attribute value of the current point to
        // a string representation.
        std::string point_attribute_value;
        ConvertAttributeValueToString_(point[current_dimension],
                                       &point_attribute_value);

        if (trial_split_values.find(point_attribute_value) !=
            std::string::npos) {
          left_bound |= point;
        }
        else {
          right_bound |= point;
        }
      } // end of looping through each point.

      // Compute the improvement.
      CalcPrecision_t delta_impurity;
      ComputeImprovement(split_values, left_bound, right_bound,
                         it.count(), &delta_impurity);

      if (delta_impurity > *best_gain) {
        *split_value = trial_split_values;
        *best_gain = delta_impurity;
      }
    }

    template < typename CalcPrecision_t, typename MapSplitValuesType,
    typename TreeIteratorType >
    void SortAndPartition_(
      const std::vector<CalcPrecision_t> &scores,
      const std::vector< std::pair<T, int> > &attribute_value_counts,
      MapSplitValuesType &split_values,
      int current_dimension,
      TreeIteratorType &it,
      std::string *split_value,
      CalcPrecision_t *best_gain) {

      // Sort the attribute values according to the probabilities of
      // the first class.
      IndexSorter<CalcPrecision_t> sorter;
      sorter.Init(scores);
      std::vector<int> sorted_indices(scores.size());
      for (int i = 0; i < scores.size(); i++) {
        sorted_indices[i] = i;
      }
      std::sort(sorted_indices.begin(), sorted_indices.end(), sorter);

      // Loop through each of the sorted attribute values.
      std::string trial_split_values("");
      for (int i = 0; i < sorted_indices.size(); i++) {

        // The split value: append to the trial_split_values string
        // and group every point whose attribute value is among the
        // string goes to the left.
        std::string trial_split_value;
        ConvertAttributeValueToString_(
          attribute_value_counts[ sorted_indices[i] ].first,
          &trial_split_value);

        if (i == 0) {
          trial_split_values = std::string(trial_split_value);
        }
        else {
          trial_split_values += std::string(" | " + trial_split_value);
        }

        // Temporarily partition.
        typename MapSplitValuesType::TreeT::Bound_t left_bound;
        typename MapSplitValuesType::TreeT::Bound_t right_bound;

        Partition_(split_values, trial_split_values, current_dimension,
                   left_bound, right_bound, it,
                   split_value, best_gain);

      } // end of looping over all of the sorted attribute values.
    }

    template<typename MapSplitValuesType, typename CalcPrecision_t>
    void ExhaustiveSearch_(
      const std::vector< std::pair<T, int> > &attribute_value_counts,
      MapSplitValuesType &split_values,
      std::string *split_value, CalcPrecision_t *best_gain) {

      int current_dimension = *(split_values.current_dimension());

      // Generate up to $2^(A - 1)$ combinations.
      for (int i = 0; i < (int) pow(2, attribute_value_counts.size() - 1); i++) {

        // Using the binary representation of $i$, generate the set of
        // strings that must be put to the right node.
        std::string trial_split_values("");
        int mod_remainder = i;
        for (int j = attribute_value_counts.size() - 1; j >= 0; j--) {
          int mod = mod_remainder % 2;
          std::string attribute_value;
          if (mod == 1) {
            ConvertAttributeValueToString_(attribute_value_counts[i].first,
                                           &attribute_value);
            if (trial_split_values == "") {
              trial_split_values = attribute_value;
            }
            else {
              trial_split_values += std::string(" | ") + attribute_value;
            }
          }
          mod_remainder = mod_remainder / 2 ;
        } // end of looping over each attribute value.

        // Temporarily partition.
        typename MapSplitValuesType::TreeT::Bound_t left_bound;
        typename MapSplitValuesType::TreeT::Bound_t right_bound;

        Partition_(split_values, trial_split_values, current_dimension,
                   left_bound, right_bound, *(split_values.iterator()),
                   split_value, best_gain);
      }
    }

    template<typename MapSplitValuesType, typename CalcPrecision_t>
    void HeuristicSearch_(
      const std::vector< std::pair<T, int> > &attribute_value_counts,
      MapSplitValuesType &split_values,
      std::string *split_value, CalcPrecision_t *best_gain) {

      // The iterator of the curret node.
      typename MapSplitValuesType::TreeIteratorT &it =
        *(split_values.iterator());

      // The current node to be split and its associated bound and the
      // class counts.
      typedef typename MapSplitValuesType::TreeT::Point_t Point_t;
      typename MapSplitValuesType::TreeT *node = split_values.node();
      const typename MapSplitValuesType::TreeT::Bound_t &node_bound =
        split_values.iterator()->table().get_node_bound(node);
      typedef typename boost::mpl::at_c < typename Point_t::MetaData_t::TypeList_t, 0 >::type ClassLabelType;
      const std::map<ClassLabelType, int> &class_counts =
        node_bound.class_counts();

      // The current dimension.
      int current_dimension = *(split_values.current_dimension());
      int total_num_points = it.count();

      // Enumerate the class labels in order.
      std::vector< std::pair<ClassLabelType, int> > class_labels;
      for (typename std::map<ClassLabelType, int>::const_iterator
           class_counts_it = class_counts.begin();
           class_counts_it != class_counts.end(); class_counts_it++) {
        class_labels.push_back(std::pair<ClassLabelType, int> (
                                 class_counts_it->first, class_counts_it->second));
      }

      // The class probability matrix.
      fl::dense::Matrix<CalcPrecision_t, false> class_probability_matrix;
      class_probability_matrix.Init(attribute_value_counts.size(),
                                    class_labels.size());
      class_probability_matrix.SetZero();

      Point_t point;
      index_t point_id;
      for (it.Reset(); it.HasNext();) {
        it.Next(&point, &point_id);
        T attribute_value = point[current_dimension];
        ClassLabelType class_label = point.meta_data().template get<0>();

        for (int k = 0; k < class_labels.size(); k++) {
          if (class_labels[k].first == class_label) {
            for (int j = 0; j < attribute_value_counts.size(); j++) {
              if (attribute_value_counts[j].first == attribute_value) {
                class_probability_matrix.set(
                  j, k, class_probability_matrix.get(j, k) + 1.0);
              }
            }
          }
        }
      } // end of looping over each point in the node.

      // Normalize to compute the probability.
      for (int k = 0; k < class_labels.size(); k++) {
        for (int j = 0; j < attribute_value_counts.size(); j++) {
          class_probability_matrix.set(
            j, k, class_probability_matrix.get(j, k) /
            ((CalcPrecision_t) attribute_value_counts[j].second));
        }
      }

      // Compute the covariance matrix and the leading eigenvector.
      fl::dense::Matrix<CalcPrecision_t, false> covariance_matrix;
      covariance_matrix.Init(class_labels.size(), class_labels.size());
      covariance_matrix.SetZero();
      for (int j = 0; j < class_labels.size(); j++) {
        CalcPrecision_t prob_class_j =
          ((CalcPrecision_t) class_labels[j].second) /
          ((CalcPrecision_t) total_num_points);
        for (int i = 0; i < class_labels.size(); i++) {
          CalcPrecision_t prob_class_i =
            ((CalcPrecision_t) class_labels[i].second) /
            ((CalcPrecision_t) total_num_points);
          CalcPrecision_t element = 0;
          for (int k = 0; k < attribute_value_counts.size(); k++) {
            element +=
              (class_probability_matrix.get(k, j) - prob_class_j) *
              (class_probability_matrix.get(k, i) - prob_class_i) *
              attribute_value_counts[k].second;
          }
          covariance_matrix.set(i, j, element);
        }
      }
      success_t success_flag;
      fl::data::MonolithicPoint<CalcPrecision_t> eigenvalues;
      fl::dense::Matrix<CalcPrecision_t, false> eigenvectors;
      fl::dense::ops::Eigenvectors<fl::la::Init>(
        covariance_matrix, &eigenvalues, &eigenvectors,	&success_flag);

      // Find the index of the largest eigenvalue.
      int largest_eigenvalue_index = 0;
      CalcPrecision_t largest_eigenvalue = eigenvalues[0];
      for (int i = 1; i < eigenvalues.length(); i++) {
        if (eigenvalues[i] > largest_eigenvalue) {
          largest_eigenvalue_index = i;
          largest_eigenvalue = eigenvalues[i];
        }
      }

      // Compute the "score" of each attribute value and sort the
      // attribute value according to it.
      std::vector<CalcPrecision_t> projected_scores(
        attribute_value_counts.size(), 0.0);
      for (int i = 0; i < attribute_value_counts.size(); i++) {
        for (int j = 0; j < class_labels.size(); j++) {
          projected_scores[i] += eigenvectors.get(j, largest_eigenvalue_index) *
                                 class_probability_matrix.get(i, j);
        }
      }
      SortAndPartition_(projected_scores,
                        attribute_value_counts, split_values,
                        current_dimension, it, split_value, best_gain);
    }

    template<typename MapSplitValuesType, typename CalcPrecision_t>
    void TwoClass_(
      const std::vector< std::pair<T, int> > &attribute_value_counts,
      MapSplitValuesType &split_values,
      std::string *split_value, CalcPrecision_t *best_gain) {

      typedef typename MapSplitValuesType::TreeIteratorT::Point_t::CalcPrecision_t CalcPrecision_t;

      // The total number of points that we need to split, and the
      // attribute index number.
      int current_dimension = *(split_values.current_dimension());

      // The probability of the first class for each attribute value.
      std::vector<CalcPrecision_t> first_class_probabilities;
      first_class_probabilities.resize(attribute_value_counts.size());

      // Some type defines.
      typedef typename MapSplitValuesType::TreeIteratorT::Point_t Point_t;
      typedef typename boost::mpl::at_c < typename Point_t::MetaData_t::TypeList_t, 0 >::type ClassLabelType;

      // The class distribution for each attribute value.
      std::map<T, std::map<ClassLabelType, int> > class_distributions;

      // The distributions of the class labels for this node.
      const std::map<ClassLabelType, int> &class_counts =
        split_values.iterator()->table().get_node_bound(
          split_values.node()).class_counts();

      // Loop through each point and build the class distribution.
      typename MapSplitValuesType::TreeIteratorT &it =
        *(split_values.iterator());
      typename MapSplitValuesType::TreeIteratorT::Point_t point;
      index_t point_id;
      for (it.Reset(); it.HasNext();) {
        it.Next(&point, &point_id);
        if (class_distributions.find(point[current_dimension]) !=
            class_distributions.end())  {
          class_distributions[point[current_dimension]]
          [point.meta_data().template get<0>()] = 1;
        }
        else {
          class_distributions[point[current_dimension]]
          [point.meta_data().template get<0>()] += 1;
        }
      }

      int index = 0;
      ClassLabelType first_class_label = class_counts.begin()->first;
      for (typename std::map<T, std::map<ClassLabelType, int> >::iterator
           class_outer_it = class_distributions.begin();
           class_outer_it != class_distributions.end(); class_outer_it++,
           index++) {

        int sum = 0;
        for (typename std::map<ClassLabelType, int>::iterator class_inner_it =
               class_outer_it->second.begin();
             class_inner_it != class_outer_it->second.end(); class_inner_it++) {
          sum += (class_inner_it->second);
        }
        if (sum > 0) {
          typename std::map<ClassLabelType, int>::iterator first_class_it =
            class_outer_it->second.find(first_class_label);
          int first_class_count =
            (first_class_it != class_outer_it->second.end()) ?
            (first_class_it->second) : 0;
          first_class_probabilities[index] =
            ((CalcPrecision_t) first_class_count) / ((CalcPrecision_t) sum);
        }
        else {
          first_class_probabilities[index] = 0;
        }
      }

      SortAndPartition_(first_class_probabilities,
                        attribute_value_counts, split_values,
                        current_dimension, it, split_value, best_gain);
    }

  public:

    template<typename TreeIteratorType, typename TreeType>
    PartitionNominalDimension(TreeIteratorType &it, TreeType *node,
                              const std::pair<int, std::string> &split_values,
                              std::deque<bool> *membership) {

      // Allocate the membership vector.
      membership->resize(it.count());

      typename TreeIteratorType::Point_t point;
      index_t point_id;

      // Loop through each point and group.
      it.Reset();
      for (int i = 0; it.HasNext(); i++) {
        it.Next(&point, &point_id);

        // Conversion of the attribute value of the current point to
        // a string representation.
        std::string point_attribute_value;
        ConvertAttributeValueToString_(point[split_values.first],
                                       &point_attribute_value);

        // A point belongs to the left node if its attribute value is
        // among the values in the split values.
        (*membership)[i] = (split_values.second.find(point_attribute_value) !=
                            std::string::npos);
      } // end of looping through each point.
    }

    template<typename MapSplitValuesType, typename CalcPrecision_t>
    PartitionNominalDimension(MapSplitValuesType &split_values,
                              std::string *split_value,
                              CalcPrecision_t *best_gain) {

      // Find out how many examples take a particular attribute value.
      std::map<T, int> attribute_value_counts;
      int current_dimension = *(split_values.current_dimension());
      typename MapSplitValuesType::TreeIteratorT &it =
        *(split_values.iterator());
      typename MapSplitValuesType::TreeIteratorT::Point_t point;
      index_t point_id;
      for (it.Reset(); it.HasNext();) {
        it.Next(&point, &point_id);
        attribute_value_counts[point[current_dimension]] += 1;
      }

      // If there is less than two attribute values for this
      // dimension, no need to split.
      if (attribute_value_counts.size() <= 1) {
        return;
      }

      // Convert the hash table into the vector.
      std::vector< std::pair<T, int> > attribute_value_counts_vector;
      for (typename std::map<T, int>::const_iterator
           attribute_value_counts_it = attribute_value_counts.begin();
           attribute_value_counts_it != attribute_value_counts.end();
           attribute_value_counts_it++) {
        attribute_value_counts_vector.push_back(
          std::pair<T, int>(attribute_value_counts_it->first,
                            attribute_value_counts_it->second));
      }

      // For the two-class problem,
      if (it.table().get_node_bound(split_values.node()).class_counts().size()
          == 2) {
        TwoClass_(attribute_value_counts_vector, split_values, split_value,
                  best_gain);
      }

      // If the number of distinct attribute values is small enough,
      // then do an exhaustive search.
      else if (attribute_value_counts_vector.size() <= 4) {
        ExhaustiveSearch_(attribute_value_counts_vector, split_values,
                          split_value, best_gain);
      }
      else {
        HeuristicSearch_(attribute_value_counts_vector, split_values,
                         split_value, best_gain);
      }
    }
};
};
};

#endif
