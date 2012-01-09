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
#ifndef FL_LITE_FASTLIB_TREE_CLASSIFICATION_DECISION_TREE_H
#define FL_LITE_FASTLIB_TREE_CLASSIFICATION_DECISION_TREE_H

#include "fastlib/base/base.h"
#include "fastlib/tree/spacetree.h"
#include "fastlib/tree/bounds.h"
#include "fastlib/tree/cart_impurity.h"
#include "fastlib/tree/classification_decision_tree_private.h"
#include "fastlib/tree/classification_decision_tree_partition.h"
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
#include "boost/type_traits/is_same.hpp"

namespace fl {
namespace tree {

template < typename StoragePrecision = double,
typename CalcPrecision = double,
int t_pow = 2,
typename ClassLabelType = int >
class CartBound {

  public:

    typedef ClassLabelType ClassLabel_t;

  public:

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & boost::serialization::make_nvp("class_counts", class_counts_);
      ar & boost::serialization::make_nvp("total_counts", total_counts_);
      ar & boost::serialization::make_nvp("majority_class", majority_class_);
      ar & boost::serialization::make_nvp("split_dimension", split_dimension_);
      ar & boost::serialization::make_nvp("numeric_split_value", numeric_split_value_);
      ar & boost::serialization::make_nvp("nominal_split_value", nominal_split_value_);
      ar & boost::serialization::make_nvp("node_is_leaf", node_is_leaf_);
      ar & boost::serialization::make_nvp("subtree_incorrect_count", subtree_incorrect_count_);
      ar & boost::serialization::make_nvp("cost_complexity", cost_complexity_);
    }

    CartBound() {
      Reset();
    }

    template<typename TableType>
    void Init(const TableType &table_in) {
    }

    CalcPrecision MaxDistanceWithinBound() const {
      return 0;
    }

    template<typename TreeIteratorType>
    CartBound(TreeIteratorType &it) {
      Reset();
      it.Reset();
      while (it.HasNext()) {
        typename TreeIteratorType::Point_t point;
        index_t point_id;
        it.Next(&point, &point_id);
        (*this) |= point;
      }
    }

    std::map<ClassLabelType, int> &class_counts() {
      return class_counts_;
    }

    const std::map<ClassLabelType, int> &class_counts() const {
      return class_counts_;
    }

    ClassLabelType majority_class() const {
      return majority_class_;
    }

    int n_points() const {
      return total_counts_;
    }

    template<typename ImpurityType>
    CalcPrecision impurity(const ImpurityType &impurity_function) const {
      return impurity_function.Compute(class_counts_, total_counts_,
                                       majority_class_);
    }

    template<typename PointType>
    void operator |= (const PointType &point) {

      // Increment the class count and the total number of points.
      if (class_counts_.find(point.meta_data().template get<0>()) !=
          class_counts_.end()) {
        class_counts_[point.meta_data().template get<0>()] += 1;
      }
      else {
        class_counts_[point.meta_data().template get<0>()] = 1;
      }
      total_counts_++;
    }

    template<typename PointType>
    void operator -= (const PointType &point) {

      // Take out the point from the class counts and the total number
      // of points.
      class_counts_[point.meta_data().template get<0>()] -= 1;
      total_counts_--;
    }

    /** @brief Loop through the current class list, and find the class
     *         with the most number of points.
     */
    void ComputeMajorityClass() {
      int max_count = -1;
      for (typename std::map<ClassLabelType, int>::const_iterator it =
             class_counts_.begin(); it != class_counts_.end(); it++) {
        if (it->second > max_count) {
          majority_class_ = it->first;
          max_count = it->second;
        }
      }
    }

    void Reset() {
      class_counts_.clear();
      total_counts_ = 0;
      majority_class_ = 0;
      split_dimension_ = -1;
      numeric_split_value_ = std::numeric_limits<double>::max();
      nominal_split_value_ = std::string("");
      node_is_leaf_ = true;
    }

    template<typename T>
    void set_split_information(int split_dimension_in,
                               const T *numeric_split_value_in,
                               const std::string *nominal_split_value_in) {
      split_dimension_ = split_dimension_in;
      if (numeric_split_value_in != NULL) {
        numeric_split_value_ = *numeric_split_value_in;
      }
      if (nominal_split_value_in != NULL) {
        nominal_split_value_ = *nominal_split_value_in;
      }
      node_is_leaf_ = false;
    }

    void reset_split_information() {
      split_dimension_ = -1;
      numeric_split_value_ = std::numeric_limits<double>::max();
      nominal_split_value_ = "";
      node_is_leaf_ = true;
    }

    void split_information(int *split_dimension_out,
                           double *numeric_split_value_out,
                           std::string *nominal_split_value_out,
                           bool *using_numeric_split) const {
      *split_dimension_out = split_dimension_;
      *numeric_split_value_out = numeric_split_value_;
      *nominal_split_value_out = nominal_split_value_;
      *using_numeric_split =
        (numeric_split_value_ < std::numeric_limits<double>::max());
    }

    void mark_it_as_leaf() {
      node_is_leaf_ = true;
    }

    bool node_is_leaf() const {
      return node_is_leaf_;
    }

    void SubtreeIncorrectCount(const CartBound *left_bound,
                               const CartBound *right_bound) {
      if (left_bound != NULL && right_bound != NULL) {
        subtree_incorrect_count_ = left_bound.subtree_incorrect_count() +
                                   right_bound.subtree_incorrect_count();
      }
      else {
        subtree_incorrect_count_ = total_counts_ -
                                   class_counts_[majority_class_];
      }
    }

    int node_incorrect_count() const {
      return total_counts_ - class_counts_[majority_class_];
    }

    int subtree_incorrect_count() const {
      return subtree_incorrect_count_;
    }

    void set_cost_complexity(CalcPrecision cost_complexity_in) {
      cost_complexity_ = cost_complexity_in;
    }

    CalcPrecision cost_complexity() const {
      return cost_complexity_;
    }

  private:

    std::map<ClassLabelType, int> class_counts_;

    int total_counts_;

    ClassLabelType majority_class_;

    int split_dimension_;

    double numeric_split_value_;

    std::string nominal_split_value_;

    bool node_is_leaf_;

    int subtree_incorrect_count_;

    CalcPrecision cost_complexity_;
};


/** @brief Classification decision tree.
 */
class ClassificationDecisionTree {

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

      private:

        template < typename ImpurityType,
        typename TreeIteratorType,
        typename TreeType >
        static void Split_(
          ImpurityType &impurity, TreeIteratorType &it,
          TreeType *node,
          const std::pair<int, float> &float_dimension_split_values,
          const std::pair<int, double> &double_dimension_split_values,
          const std::pair<int, std::string> &bool_dimension_split_values,
          const std::pair<int, std::string> &char_dimension_split_values,
          const std::pair<int, std::string> &int_dimension_split_values,
          int best_dimension,
          const std::string &best_dimension_type,
          std::deque<bool> *membership) {

          if (best_dimension_type == "float") {
            ((typename TreeType::Bound_t &)
             (it.table().get_node_bound(node))).set_split_information(
               best_dimension, &float_dimension_split_values.second,
               (std::string *) NULL);
            PartitionNumericDimension(
              it, node, float_dimension_split_values, membership);
          }
          else if (best_dimension_type == "double") {
            ((typename TreeType::Bound_t &)
             (it.table().get_node_bound(node))).set_split_information(
               best_dimension, &double_dimension_split_values.second,
               (std::string *) NULL);
            PartitionNumericDimension(
              it, node, double_dimension_split_values, membership);
          }
          else if (best_dimension_type == "bool") {
            ((typename TreeType::Bound_t &)
             (it.table().get_node_bound(node))).set_split_information(
               best_dimension, (bool *) NULL,
               &bool_dimension_split_values.second);
            PartitionNominalDimension<bool>(
              it, node, bool_dimension_split_values, membership);
          }
          else if (best_dimension_type == "char") {
            ((typename TreeType::Bound_t &)
             (it.table().get_node_bound(node))).set_split_information(
               best_dimension, (char *) NULL,
               &char_dimension_split_values.second);
            PartitionNominalDimension<char>(
              it, node, char_dimension_split_values, membership);
          }
          else if (best_dimension_type == "int") {
            ((typename TreeType::Bound_t &)
             (it.table().get_node_bound(node))).set_split_information(
               best_dimension, (int *) NULL, &int_dimension_split_values.second);
            PartitionNominalDimension<int>(
              it, node, int_dimension_split_values, membership);
          }
          else {
            fl::logger->Die() << "Unsupported data type encountered " <<
            "in building the classification decision tree.";
          }
        }

      public:
        template < typename ImpurityType,
        typename TreeIteratorType,
        typename BoundType >
        static void FindBoundFromMatrix(ImpurityType &impurity,
                                        TreeIteratorType &it,
                                        BoundType *bound) {

          typedef typename TreeIteratorType::Point_t Point_t;

          // Reset the iterator so that it starts from the beginning
          // point.
          it.Reset();

          // Count the number of points in each class, and form the
          // bounding box for all points.
          while (it.HasNext()) {
            index_t col_id;
            Point_t col;
            it.Next(&col, &col_id);
            (*bound) |= col;
          }

          // Find the majority class after adding all the points.
          bound->ComputeMajorityClass();
        }

        template < typename ImpurityType,
        typename TreeIteratorType,
        typename TreeType >
        static bool Partition(ImpurityType &impurity,
                              TreeIteratorType &it,
                              TreeType *node,
                              std::deque<bool> *membership) {

          typedef typename TreeIteratorType::Point_t::CalcPrecision_t CalcPrecision_t;

          // We need to check this first before we continue and split.
          // If there is only one class we stop.
          if ((it.table().get_node_bound(node)).class_counts().size() <= 1) {
            return false;
          }

          // Loop through the dense dimensions.
          int current_dimension = 0;
          int best_dimension = -1;
          std::string best_dimension_type("");

          CalcPrecision_t best_improvement = 0;
          std::pair<int, double> double_dimension_split_values;
          std::pair<int, float> float_dimension_split_values;
          std::pair<int, std::string> int_dimension_split_values;
          std::pair<int, std::string> char_dimension_split_values;
          std::pair<int, std::string> bool_dimension_split_values;

          boost::mpl::for_each <
          typename TreeIteratorType::Point_t::DenseTypes_t > (
            PartitionEachType<ImpurityType, TreeIteratorType, TreeType, true>(
              impurity,
              it,
              node,
              float_dimension_split_values,
              double_dimension_split_values,
              bool_dimension_split_values,
              char_dimension_split_values,
              int_dimension_split_values,
              &current_dimension,
              &best_dimension,
              best_dimension_type,
              &best_improvement));

          // Loop through the sparse dimensions.
          boost::mpl::for_each <
          typename TreeIteratorType::Point_t::SparseTypes_t > (
            PartitionEachType<ImpurityType, TreeIteratorType, TreeType, false>(
              impurity,
              it,
              node,
              float_dimension_split_values,
              double_dimension_split_values,
              bool_dimension_split_values,
              char_dimension_split_values,
              int_dimension_split_values,
              &current_dimension,
              &best_dimension,
              best_dimension_type,
              &best_improvement));

          // If the maximum width was too small, or the improvement is
          // very little, then do not split.
          if (best_dimension < 0 ||
              best_improvement <= std::numeric_limits<CalcPrecision_t>::min()) {
            return false;
          }
          Split_(impurity, it, node, float_dimension_split_values,
                 double_dimension_split_values, bool_dimension_split_values,
                 char_dimension_split_values, int_dimension_split_values,
                 best_dimension, best_dimension_type, membership);

          int left_assigned_count = 0;
          for (int i = 0; i < it.count(); i++) {
            if ((*membership)[i]) {
              left_assigned_count++;
            }
          }

          // Reset the split information if it fails.
          if (left_assigned_count == 0 || left_assigned_count == it.count()) {
            ((typename TreeType::Bound_t &)
             (it.table().get_node_bound(node))).reset_split_information();
          }
          return left_assigned_count > 0 && left_assigned_count < it.count();
        }
    };
};
}; // tree namespace
}; // fl namespace

#endif
