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

#ifndef FL_LITE_FASTLIB_TREE_TREE_TEST_PRIVATE_H
#define FL_LITE_FASTLIB_TREE_TREE_TEST_PRIVATE_H

#include "boost/mpl/map.hpp"
#include "boost/mpl/if.hpp"
#include "fastlib/base/base.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/table/file_data_access.h"
#include "fastlib/table/table.h"
#include "fastlib/tree/cart_impurity.h"
#include "fastlib/tree/classification_decision_tree.h"
#include "fastlib/tree/kdtree.h"
#include "fastlib/tree/metric_tree.h"
#include "fastlib/tree/similarity_tree.h"
#include "fastlib/tree/bregmanballbound.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "fastlib/metric_kernel/cosine_premetric.h"
#include "fastlib/table/default/dense/labeled/balltree/table.h"
#include "fastlib/table/default/dense/labeled/bregmantree/table.h"

namespace fl {
namespace tree {
namespace tree_test_private {

template<typename BoundType>
class TestPartitionIntegrity;

template < typename StoragePrecision,
typename CalcPrecision,
int t_pow,
typename ClassLabelType >
class TestPartitionIntegrity< fl::tree::CartBound<StoragePrecision, CalcPrecision, t_pow, ClassLabelType> > {

  public:

    template<typename TableType>
    bool PointSplitIntegrity_(const typename TableType::Point_t &point,
                              const TableType &table,
                              typename TableType::Tree_t *node,
                              bool should_belong_to_left_child) {

      // Get the split value, and verify that the node is splitting
      // them into disjoint sets!
      int split_dimension = -1;
      double numeric_split_value = std::numeric_limits<double>::max();
      std::string nominal_split_value("");
      bool using_numeric_split = false;
      table.get_node_bound(node).split_information(
        &split_dimension, &numeric_split_value,
        &nominal_split_value, &using_numeric_split);

      bool belongs_to_left = false;
      if (using_numeric_split) {
        belongs_to_left = (point[split_dimension] < numeric_split_value);
      }
      else {
        std::string stringified_attribute_value;
        std::stringstream ss;
        ss << point[split_dimension];
        ss >> stringified_attribute_value;
        belongs_to_left = (nominal_split_value.find(stringified_attribute_value)
                           != std::string::npos);
      }

      return should_belong_to_left_child == belongs_to_left;
    }

    template<typename ImpurityType, typename TableType>
    TestPartitionIntegrity(const ImpurityType &metric,
                           const TableType &table,
                           typename TableType::Tree_t *node) {

      // Return if the node is leaf.
      if (table.node_is_leaf(node) == false) {

        // The left and the right child.
        typename TableType::Tree_t *left_child =
          table.get_node_left_child(node);
        typename TableType::Tree_t *right_child =
          table.get_node_right_child(node);
        typename TableType::TreeIterator left_it =
          table.get_node_iterator(left_child);
        typename TableType::TreeIterator right_it =
          table.get_node_iterator(right_child);

        // Temporary point and its index.
        typename TableType::Point_t point;
        int point_index;

        // Check whether the left child satisfies the split invariant.
        for (; left_it.HasNext();) {
          left_it.Next(&point, &point_index);
          if (PointSplitIntegrity_(point, table, node, true) == false) {
            fl::logger->Die() << "CART split is corrupted!";
          }
        }

        // Check whether the right child satisfies the split invariant.
        for (; right_it.HasNext();) {
          right_it.Next(&point, &point_index);
          if (PointSplitIntegrity_(point, table, node, false) == false) {
            fl::logger->Die() << "CART split is corrupted!";
          }
        }
      }
    }
};

template<typename PointType>
class TestPartitionIntegrity< fl::tree::BallBound<PointType> > {

  public:
    template<typename MetricType, typename TableType>
    TestPartitionIntegrity(const MetricType &metric,
                           const TableType &table,
                           typename TableType::Tree_t *node) {
    }
};

template<typename PointType>
class TestPartitionIntegrity< fl::tree::BregmanBallBound<PointType> > {

  public:
    template<typename MetricType, typename TableType>
    TestPartitionIntegrity(const MetricType &metric,
                           const TableType &table,
                           typename TableType::Tree_t *node) {
    }
};

template<typename StoragePrecision, typename CalcPrecision, int t_pow>
class TestPartitionIntegrity< fl::tree::GenHrectBound<StoragePrecision, CalcPrecision, t_pow> > {
  public:

    template<typename MetricType, typename TableType>
    TestPartitionIntegrity(const MetricType &metric,
                           const TableType &table,
                           typename TableType::Tree_t *node) {

      if (table.node_is_leaf(node) == false) {

        // The left and the right child and the bounds.
        typename TableType::Tree_t *left_child =
          table.get_node_left_child(node);
        typename TableType::Tree_t *right_child =
          table.get_node_right_child(node);
        typename TableType::TreeIterator left_it =
          table.get_node_iterator(left_child);
        typename TableType::TreeIterator right_it =
          table.get_node_iterator(right_child);

        // Get the bound.
        const typename TableType::Tree_t::Bound_t &left_bound =
          table.get_node_bound(left_child);
        const typename TableType::Tree_t::Bound_t &right_bound =
          table.get_node_bound(right_child);

        // Check whether the bounds are disjoint.
        bool disjoint = false;
        for (int i = 0; disjoint == false && i < table.n_attributes(); i++) {
          const GenRange<StoragePrecision> &left_range = left_bound.get(i);
          const GenRange<StoragePrecision> &right_range = right_bound.get(i);
          disjoint = (left_range.hi <= right_range.lo ||
                      right_range.hi <= left_range.lo);
        }

        if (disjoint == false) {
          fl::logger->Die() << "Kd-tree split is totally wrong!";
        }
      }
    }
};

template<typename BoundType>
class TestBoundIntegrity;

template < typename StoragePrecision,
typename CalcPrecision,
int t_pow,
typename ClassLabelType >
class TestBoundIntegrity< fl::tree::CartBound<StoragePrecision, CalcPrecision, t_pow, ClassLabelType> > {

  public:
    template<typename ImpurityType, typename TableType>
    TestBoundIntegrity(const ImpurityType &metric,
                       const TableType &table,
                       typename TableType::Tree_t *node) {

      // Check the class distribution count.
      typename TableType::TreeIterator it = table.get_node_iterator(node);
      typedef typename TableType::Tree_t::Bound_t::ClassLabel_t
      ClassLabelType;
      const std::map<ClassLabelType, int> &class_counts =
        table.get_node_bound(node).class_counts();
      std::map<ClassLabelType, int> check;

      // Check the total number of points against the class distribution.
      int total_count_check = 0;
      for (typename std::map<ClassLabelType, int>::const_iterator
           class_counts_it = class_counts.begin();
           class_counts_it != class_counts.end();
           class_counts_it++) {
        total_count_check += class_counts_it->second;
      }
      if (total_count_check != table.get_node_count(node)) {
        fl::logger->Die() << "The total number points do not match up!";
      }

      // Now, check the individual class distribution.
      for (; it.HasNext();) {
        typename TableType::Point_t point;
        int point_index;
        it.Next(&point, &point_index);
        if (check.find(point.meta_data().template get<0>()) != check.end()) {
          check[point.meta_data().template get<0>()] += 1;
        }
        else {
          check[point.meta_data().template get<0>()] = 1;
        }
      }
      if (check.size() != class_counts.size()) {
        fl::logger->Die() << "Encountered different number of classes!";
      }
      for (typename std::map<ClassLabelType, int>::const_iterator
           class_counts_it = class_counts.begin();
           class_counts_it != class_counts.end(); class_counts_it++) {
        if (check.find(class_counts_it->first) == check.end() ||
            check[class_counts_it->first] != class_counts_it->second) {
          fl::logger->Die() << "The class count is different!";
        }
      }
    }
};

template<typename PointType>
class TestBoundIntegrity< fl::tree::BallBound<PointType> > {

  public:
    template<typename MetricType, typename TableType>
    TestBoundIntegrity(const MetricType &metric,
                       const TableType &table,
                       typename TableType::Tree_t *node) {

      // Get an iterator to the node.
      typename TableType::TreeIterator it = table.get_node_iterator(node);

      // Get the bound.
      const typename TableType::Tree_t::Bound_t &bound = table.get_node_bound(
            node);

      // The maximum distance to compare against the radius for
      // determining the tightness.
      typename TableType::CalcPrecision_t max_dist = 0;

      for (; it.HasNext();) {
        typename TableType::Point_t point;
        int point_id;
        it.Next(&point, &point_id);
        max_dist = std::max(max_dist, bound.MidDistance(metric, point));
        if (bound.Contains(metric, point) == false) {
          fl::logger->Die() << "The ball bound is horribly wrong!";
        }
      }

      // Now check the max distance against the radius.
      if (fabs(bound.radius() - max_dist) > 1e-6) {
        fl::logger->Die() << "The ball bound is not tight enough!";
      }
    }
};

template<typename PointType>
class TestBoundIntegrity< fl::tree::BregmanBallBound<PointType> > {

  public:
    template<typename MetricType, typename TableType>
    TestBoundIntegrity(const MetricType &metric,
                       const TableType &table,
                       typename TableType::Tree_t *node) {

      // Get an iterator to the node.
      typename TableType::TreeIterator it = table.get_node_iterator(node);

      // Get the bound.
      const typename TableType::Tree_t::Bound_t &bound = table.get_node_bound(
            node);

      // The maximum distance to compare against the radius for
      // determining the tightness.
      typename TableType::CalcPrecision_t max_dist = 0;

      for (; it.HasNext();) {
        typename TableType::Point_t point;
        int point_id;
        it.Next(&point, &point_id);
        max_dist = std::max(max_dist, bound.MinDistance(metric, point));
        if (bound.Contains(metric, point) == false) {
          fl::logger->Die() << "The ball bound is horribly wrong!";
        }
      }

      // Now check the max distance against the radius.
      if (fabs(bound.radius() - max_dist) > 1e-6) {
        fl::logger->Die() << "The ball bound is not tight enough!";
      }
    }
};

template<typename StoragePrecision, typename CalcPrecision, int t_pow>
class TestBoundIntegrity< fl::tree::GenHrectBound<StoragePrecision, CalcPrecision, t_pow> > {
  public:

    template<typename MetricType, typename TableType>
    TestBoundIntegrity(const MetricType &metric,
                       const TableType &table,
                       typename TableType::Tree_t *node) {
      // Get an iterator to the node.
      typename TableType::TreeIterator it = table.get_node_iterator(node);

      // Get the bound.
      const typename TableType::Tree_t::Bound_t &bound = table.get_node_bound(
            node);

      // The empty bound which the previous existing bound will be
      // compared to.
      typename TableType::Tree_t::Bound_t test_bound;
      test_bound.Init((TableType &) table);

      // Loop through the points and test whether it truly belongs to
      // the bound or not.
      for (; it.HasNext();) {
        typename TableType::Point_t point;
        int point_id;
        it.Next(&point, &point_id);
        if (bound.Contains(point) == false) {
          fl::logger->Die() << "Something is horribly wrong with the kd-tree "
          "bound!";
        }

        test_bound |= point;
      }

      // Test the bound against the previous one.
      for (int i = 0; i < table.n_attributes(); i++) {
        const GenRange<StoragePrecision> &test_dim = test_bound.get(i);
        const GenRange<StoragePrecision> &prev_dim = bound.get(i);
        CalcPrecision low_diff = fabs(test_dim.lo - prev_dim.lo);
        CalcPrecision high_diff = fabs(test_dim.hi - prev_dim.hi);
        if (low_diff > 1e-6 || high_diff > 1e-6) {
          fl::logger->Die() << "The kdtree bound is not tight enough";
        }
      }
    }
};

template<typename BoundType1, typename PointType>
class BoundExtractor;

template< typename PointType1, typename PointType2>
class BoundExtractor<fl::tree::BallBound<PointType1>, PointType2> {
  public:
    typedef fl::tree::BallBound<PointType2> BoundType;
};

template<typename BoundType1, typename PointType>
class BoundExtractor {
  public:
    typedef BoundType1 BoundType;
};

template < typename StoragePrecision, typename CalcPrecision,
int t_pow, typename PointType >
class BoundExtractor<fl::tree::GenHrectBound<StoragePrecision, CalcPrecision, t_pow>, PointType> {

  public:
    typedef fl::tree::GenHrectBound<StoragePrecision, CalcPrecision, t_pow> BoundType;
};
};
};
};

#endif
