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

#ifndef FL_LITE_MLPACK_CART_CART_DEV_H
#define FL_LITE_MLPACK_CART_CART_DEV_H

#include "mlpack/cart/cart.h"
#include <string>

namespace fl {
namespace ml {

template<typename TableType>
void Cart<TableType>::Prune_(double cost_complexity) {

  // The root node of the tree.
  typename Cart<TableType>::TreeType *root = table_->get_tree();

  // Compute the training errors at each node and the cost complexity
  // at each node.
  std::vector< typename Cart<TableType>::TreeType * > internal_nodes;
  TrainingErrors_(root, &internal_nodes);
  CostComplexities_(root);
}

template<typename TableType>
int Cart<TableType>::NumberOfLeaves_(
  typename Cart<TableType>::TreeType *node) {

  if (table_->get_node_bound(node).node_is_leaf()) {
    return 1;
  }
  else {
    return NumberOfLeaves_(table_->get_node_left_child(node)) +
           NumberOfLeaves_(table_->get_node_right_child(node));
  }
}

template<typename TableType>
void Cart<TableType>::CostComplexities_(
  typename Cart<TableType>::TreeType *node) {

  if (table_->get_node_bound(node).node_is_leaf()) {

    // Never prune the leaf node, so its cost complexity is infinite.
    table_->get_node_bound(node).set_cost_complexity(
      std::numeric_limits<double>::max());
  }

  // If the current node is an internal node,
  else {

    typename TableType::TreeType::Bound_t &bound =
      table_->get_node_bound(node);

    int improvement = bound.node_incorrect_count() -
                      bound.subtree_incorrect_count();

    // If the splitting increases training error, immediately mark it
    // as a leaf node and set its cost complexity to infinite.
    if (improvement <= 0) {
      bound.mark_it_as_leaf();
      bound.set_cost_complexity(std::numeric_limits<double>::max());
    }
    else {

      // Compute the cost complexity.
      double cost_complexity = improvement /
                               ((double) table_->n_entries() *
                                (NumberOfLeaves_(node) - 1));
      bound.set_cost_complexity(cost_complexity);

      // Recursively compute for the children.
      CostComplexities_(table_->get_node_left_child(node));
      CostComplexities_(table_->get_node_right_child(node));
    }
  }
}

template<typename TableType>
void Cart<TableType>::TrainingErrors_(
  typename Cart<TableType>::TreeType *node,
  std::vector< typename Cart<TableType>::TreeType * > *internal_nodes) {

  if (table_->get_node_bound(node).node_is_leaf()) {
    table_->get_node_bound(node).SubtreeIncorrectCount(
      (const typename TableType::TreeType::Bound_t *) NULL,
      (const typename TableType::TreeType::Bound_t *) NULL);
  }
  else {
    internal_nodes->push_back(node);

    typename TableType::TreeType *left_child =
      table_->get_node_left_child(node);
    typename TableType::TreeType *right_child =
      table_->get_node_right_child(node);

    TrainingErrors_(left_child, internal_nodes);
    TrainingErrors_(right_child, internal_nodes);

    table_->get_node_bound(node).SubtreeIncorrectCount(
      table_->get_node_bound(left_child), table_->get_node_bound(right_child));
  }
}

template<typename TableType>
void Cart<TableType>::CrossValidate(int num_folds) {

}

template<typename TableType>
void Cart<TableType>::Init(TableType *table_in,
                           int leaf_size_in,
                           const std::string &impurity_in) {
  table_ = table_in;

  if (impurity_in == "entropy") {
    typename TableType::template IndexArgs <
    fl::tree::EntropyImpurity > index_args;
    index_args.leaf_size = leaf_size_in;
    table_->IndexData(index_args);
  }
  else if (impurity_in == "gini") {
    typename TableType::template IndexArgs< fl::tree::GiniImpurity > index_args;
    index_args.leaf_size = leaf_size_in;
    table_->IndexData(index_args);
  }
  else if (impurity_in == "misclassification") {
    typename TableType::template IndexArgs <
    fl::tree::MisclassificationImpurity > index_args;
    index_args.leaf_size = leaf_size_in;
    table_->IndexData(index_args);
  }
}

template<typename TableType>
template<typename QueryTableType, typename TableVectorType>
void Cart<TableType>::Classify(
  const QueryTableType &query_table,
  TableVectorType *labels_out) {

  // Perhaps this can be done faster using a tree on the query table.
  for (int i = 0; i < query_table.n_entries(); i++) {
    typename QueryTableType::Point_t query_point;
    query_table.get(i, &query_point);
    Classify(query_point, *labels_out, i);
  }
}

template<typename TableType>
template<typename PointType, typename TableVectorType>
void Cart<TableType>::Classify(
  const PointType &point,
  TableVectorType &label_out,
  int query_point_index) {

  Classify_(point, table_->get_tree(), label_out, query_point_index);
}

template<typename TableType>
template<typename PointType, typename TableVectorType>
void Cart<TableType>::Classify_(
  const PointType &point,
  typename Cart<TableType>::TreeType *node,
  TableVectorType &label_out,
  int query_point_index) {

  // If the node is a leaf node,
  if (table_->get_node_bound(node).node_is_leaf()) {
    label_out.set(query_point_index,
                  table_->get_node_bound(node).majority_class());
  }
  else {

    // Get the split information and traverse accordingly.
    int split_dimension = -1;
    double numeric_split_value = std::numeric_limits<double>::max();
    std::string nominal_split_value("");
    bool using_numeric_split = false;
    table_->get_node_bound(node).split_information(
      &split_dimension, &numeric_split_value,
      &nominal_split_value, &using_numeric_split);

    if (using_numeric_split) {
      if (point[split_dimension] < numeric_split_value) {
        Classify_(point, table_->get_node_left_child(node), label_out,
                  query_point_index);
      }
      else {
        Classify_(point, table_->get_node_right_child(node), label_out,
                  query_point_index);
      }
    }
    else {
      std::string stringified_attribute_value;
      std::stringstream ss;
      ss << point[split_dimension];
      ss >> stringified_attribute_value;
      if (nominal_split_value.find(stringified_attribute_value)
          != std::string::npos) {
        Classify_(point, table_->get_node_left_child(node), label_out,
                  query_point_index);
      }
      else {
        Classify_(point, table_->get_node_right_child(node), label_out,
                  query_point_index);
      }
    }
  }
}

template<typename TableType>
void Cart<TableType>::String(
  typename Cart<TableType>::TreeType *node, int level, std::string *text) {

  // The bounding box for the current node.
  const typename Cart<TableType>::TreeType::Bound_t &bound =
    table_->get_node_bound(node);

  typedef typename Cart<TableType>::ClassLabelType ClassLabelType;

  // If the node is a leaf node,
  if (table_->get_node_bound(node).node_is_leaf()) {

    // Get the class count distribution.
    const std::map<ClassLabelType, int> &class_counts =
      bound.class_counts();

    // Get the majority class label.
    ClassLabelType majority_class_label = bound.majority_class();

    // Get the number of points in the majority class.
    int majority_class_count = class_counts.find(majority_class_label)->second;
    int incorrect_count = table_->get_node_count(node) - majority_class_count;
    std::string stringified;
    std::stringstream ss;
    ss << majority_class_label <<
    "(" << majority_class_count << "/" << incorrect_count << ")";
    ss >> stringified;
    (*text) += ": " + stringified;
  }
  else {

    int split_dimension = -1;
    double numeric_split_value = std::numeric_limits<double>::max();
    std::string nominal_split_value("");
    bool using_numeric_split = false;
    bound.split_information(&split_dimension, &numeric_split_value,
                            &nominal_split_value, &using_numeric_split);

    // Stringify the left child.
    StringInternalNode_(table_->get_node_left_child(node), true,
                        level + 1, text,
                        split_dimension, numeric_split_value,
                        nominal_split_value, using_numeric_split);

    // Stringify the right child.
    StringInternalNode_(table_->get_node_right_child(node), false,
                        level + 1, text,
                        split_dimension, numeric_split_value,
                        nominal_split_value, using_numeric_split);
  }
}

template<typename TableType>
void Cart<TableType>::StringInternalNode_(
  typename Cart<TableType>::TreeType *child_node,
  bool is_left_child_node,
  int level,
  std::string *text,
  int split_dimension,
  double numeric_split_value,
  const std::string &nominal_split_value,
  bool using_numeric_split) {

  const std::vector<std::string> &features = table_->data()->labels();

  // For converting numbers to string.
  std::stringstream ss;
  std::string stringified;

  (*text) += "\n";
  for (int i = 0; i < level; i++) {
    (*text) += "|  ";
  }
  (*text) += features[split_dimension];
  if (using_numeric_split) {
    ss << numeric_split_value;
    ss >> stringified;
    if (is_left_child_node) {
      (*text) += " < " + stringified;
    }
    else {
      (*text) += " >= " + stringified;
    }
  }
  else {
    if (is_left_child_node) {
      (*text) += "=" + nominal_split_value;
    }
    else {
      (*text) += "!=" + nominal_split_value;
    }
  }
  String(child_node, level + 1, text);
}
};
};

#endif
