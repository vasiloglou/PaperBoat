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

#ifndef FL_LITE_MLPACK_KDE_DUALTREE_DFS_ITERATOR_DEV_H
#define FL_LITE_MLPACK_KDE_DUALTREE_DFS_ITERATOR_DEV_H

#include "mlpack/kde/dualtree_dfs.h"

namespace fl {
namespace ml {

template<typename ProblemType>
template<typename MetricType>
DualtreeDfs<ProblemType>::iterator<MetricType>::
IteratorArgType::IteratorArgType() {

  // Initialize the members.
  qnode_ = NULL;
  rnode_ = NULL;

  // Compute the range squared distances between the two nodes.
  squared_distance_range_.InitEmptySet();
}

template<typename ProblemType>
template<typename MetricType>
DualtreeDfs<ProblemType>::iterator<MetricType>::
IteratorArgType::IteratorArgType(const IteratorArgType &arg_in) {

  // Initialize the members.
  qnode_ = arg_in.qnode();
  rnode_ = arg_in.rnode();

  // Compute the range squared distances between the two nodes.
  squared_distance_range_ = arg_in.squared_distance_range();
}

template<typename ProblemType>
template<typename MetricType>
typename ProblemType::Table_t::Tree_t *DualtreeDfs<ProblemType>::iterator
<MetricType>::IteratorArgType::qnode() {
  return qnode_;
}

template<typename ProblemType>
template<typename MetricType>
typename ProblemType::Table_t::Tree_t *DualtreeDfs<ProblemType>::iterator
<MetricType>::IteratorArgType::qnode() const {
  return qnode_;
}

template<typename ProblemType>
template<typename MetricType>
typename ProblemType::Table_t::Tree_t *DualtreeDfs<ProblemType>::iterator
<MetricType>::IteratorArgType::rnode() {
  return rnode_;
}

template<typename ProblemType>
template<typename MetricType>
typename ProblemType::Table_t::Tree_t *DualtreeDfs<ProblemType>::iterator
<MetricType>::IteratorArgType::rnode() const {
  return rnode_;
}

template<typename ProblemType>
template<typename MetricType>
const GenRange<double> &DualtreeDfs<ProblemType>::iterator
<MetricType>::IteratorArgType::squared_distance_range() const {
  return squared_distance_range_;
}

template<typename ProblemType>
template<typename MetricType>
DualtreeDfs<ProblemType>::iterator<MetricType>::
IteratorArgType::IteratorArgType(
  const MetricType &metric_in,
  typename DualtreeDfs<ProblemType>::Table_t *query_table_in,
  typename DualtreeDfs<ProblemType>::Table_t::Tree_t *qnode_in,
  typename DualtreeDfs<ProblemType>::Table_t *reference_table_in,
  typename DualtreeDfs<ProblemType>::Table_t::Tree_t *rnode_in) {

  // Initialize the members.
  qnode_ = qnode_in;
  rnode_ = rnode_in;
  squared_distance_range_ =
    (query_table_in->get_node_bound(qnode_in)).RangeDistanceSq(
      metric_in,
      reference_table_in->get_node_bound(rnode_in));
}

template<typename ProblemType>
template<typename MetricType>
DualtreeDfs<ProblemType>::iterator<MetricType>::
IteratorArgType::IteratorArgType(
  const MetricType &metric_in,
  typename DualtreeDfs<ProblemType>::Table_t *query_table_in,
  typename DualtreeDfs<ProblemType>::Table_t::Tree_t *qnode_in,
  typename DualtreeDfs<ProblemType>::Table_t *reference_table_in,
  typename DualtreeDfs<ProblemType>::Table_t::Tree_t *rnode_in,
  const GenRange<double> &squared_distance_range_in) {

  // Initialize the members.
  qnode_ = qnode_in;
  rnode_ = rnode_in;
  squared_distance_range_ = squared_distance_range_in;
}

template<typename ProblemType>
template<typename MetricType>
DualtreeDfs<ProblemType>::iterator<MetricType>::iterator(
  const MetricType &metric_in, DualtreeDfs<ProblemType> &engine_in,
  typename ProblemType::Result_t *query_results_in): metric_(metric_in) {

  engine_ = &engine_in;
  query_table_ = engine_->query_table();
  reference_table_ = engine_->reference_table();
  query_results_ = query_results_in;

  // Initialize an empty trace for the computation and the query
  // root/reference root pair into the trace.
  trace_.Init();
  trace_.push_back(IteratorArgType(metric_in, query_table_,
                                   query_table_->get_tree(),
                                   reference_table_,
                                   reference_table_->get_tree()));
}

template<typename ProblemType>
template<typename MetricType>
void DualtreeDfs<ProblemType>::iterator<MetricType>::operator++() {

  // Push a blank argument to the trace for making the exit phase.
  trace_.push_front(IteratorArgType());

  // Pop the next item to visit in the list.
  IteratorArgType args = trace_.back();
  trace_.pop_back();

  while (trace_.empty() == false && args.rnode() != NULL) {

    // Get the arguments.
    Tree_t *qnode = args.qnode();
    Tree_t *rnode = args.rnode();
    const GenRange<double> &squared_distance_range =
      args.squared_distance_range();

    // Compute the delta change.
    typename ProblemType::Delta_t delta;
    delta.DeterministicCompute(metric_, engine_->problem_->global(),
                               qnode, rnode, squared_distance_range);
    bool prunable = engine_->CanSummarize_(qnode, rnode, delta,
                                           query_results_);

    if (prunable) {
      engine_->Summarize_(qnode, delta, query_results_);
    }
    else {

      // If the query node is leaf node,
      if (query_table_->node_is_leaf(qnode)) {

        // If the reference node is leaf node,
        if (reference_table_->node_is_leaf(rnode)) {
          engine_->DualtreeBase_(metric_, qnode, rnode, query_results_);
        }
        else {
          Tree_t *rnode_first;
          GenRange<double> squared_distance_range_first,
          squared_distance_range_second;
          Tree_t *rnode_second;
          engine_->Heuristic_(metric_, qnode, query_table_,
                              reference_table_->get_node_left_child(rnode),
                              reference_table_->get_node_right_child(rnode),
                              reference_table_,
                              &rnode_first, squared_distance_range_first,
                              &rnode_second, squared_distance_range_second);

          // Push the first prioritized reference node on the back
          // of the trace and the later one on the front of the
          // trace.
          trace_.push_back(IteratorArgType(
                             metric_, query_table_, qnode,
                             reference_table_, rnode_first,
                             squared_distance_range_first));
          trace_.push_front(IteratorArgType(
                              metric_, query_table_, qnode,
                              reference_table_, rnode_second,
                              squared_distance_range_second));
        }
      }

      // If the query node is a non-leaf node,
      else {

        // Here we split the query.
        Tree_t *qnode_left = query_table_->get_node_left_child(qnode);
        Tree_t *qnode_right = query_table_->get_node_right_child(qnode);

        // If the reference node is leaf node,
        if (reference_table_->node_is_leaf(rnode)) {

          // Push both combinations on the back of the trace.
          trace_.push_back(IteratorArgType(
                             metric_, query_table_, qnode_left,
                             reference_table_, rnode));
          trace_.push_back(IteratorArgType(
                             metric_, query_table_, qnode_right,
                             reference_table_, rnode));
        }

        // Otherwise, we split both the query and the reference.
        else {

          // Split the reference.
          Tree_t *rnode_left = reference_table_->get_node_left_child(rnode);
          Tree_t *rnode_right =
            reference_table_->get_node_right_child(rnode);

          // Prioritize on the left child of the query node.
          Tree_t *rnode_first = NULL, *rnode_second = NULL;
          GenRange<double> squared_distance_range_first;
          GenRange<double> squared_distance_range_second;
          engine_->Heuristic_(metric_, qnode_left, query_table_, rnode_left,
                              rnode_right, reference_table_,
                              &rnode_first, squared_distance_range_first,
                              &rnode_second, squared_distance_range_second);
          trace_.push_back(IteratorArgType(
                             metric_, query_table_, qnode_left,
                             reference_table_, rnode_first,
                             squared_distance_range_first));
          trace_.push_front(IteratorArgType(
                              metric_, query_table_, qnode_left,
                              reference_table_, rnode_second,
                              squared_distance_range_second));

          // Prioritize on the right child of the query node.
          engine_->Heuristic_(metric_, qnode_right, query_table_,
                              rnode_left, rnode_right, reference_table_,
                              &rnode_first, squared_distance_range_first,
                              &rnode_second, squared_distance_range_second);
          trace_.push_back(IteratorArgType(
                             metric_, query_table_, qnode_right,
                             reference_table_, rnode_first,
                             squared_distance_range_first));
          trace_.push_front(IteratorArgType(
                              metric_, query_table_, qnode_right,
                              reference_table_, rnode_second,
                              squared_distance_range_second));

        } // end of non-leaf query, non-leaf reference.
      } // end of non-leaf query.

    } // end of non-prunable case.

    // Pop the next item in the list.
    args = trace_.back();
    trace_.pop_back();

  } // end of the while loop.
}

template<typename ProblemType>
template<typename MetricType>
void DualtreeDfs<ProblemType>::iterator<MetricType>::Finalize() {
  return engine_->PostProcess_(
           metric_, query_table_->get_tree(), query_results_);
}

template<typename ProblemType>
template<typename MetricType>
typename ProblemType::Result_t &DualtreeDfs<ProblemType>::iterator
<MetricType>::operator*() {
  return *query_results_;
}

template<typename ProblemType>
template<typename MetricType>
const typename ProblemType::Result_t &DualtreeDfs<ProblemType>::iterator
<MetricType>::operator*() const {
  return *query_results_;
}

template<typename ProblemType>
template<typename MetricType>
typename DualtreeDfs<ProblemType>::template iterator<MetricType>
DualtreeDfs<ProblemType>::get_iterator(
  const MetricType &metric_in,
  typename ProblemType::Result_t *query_results_in) {

  // Allocate space for storing the final results.
  query_results_in->Init(query_table_->n_entries());

  return typename DualtreeDfs<ProblemType>::template
         iterator<MetricType>(metric_in, *this, query_results_in);
}
};
};

#endif
