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

#ifndef FL_LITE_MLPACK_KDE_DUALTREE_DFS_DEV_H
#define FL_LITE_MLPACK_KDE_DUALTREE_DFS_DEV_H

#include "mlpack/kde/dualtree_dfs.h"
#include "mlpack/kde/dualtree_dfs_iterator_dev.h"

extern index_t in_recursion_counter;

template<typename ProblemType>
ProblemType *fl::ml::DualtreeDfs<ProblemType>::problem() {
  return problem_;
}

template<typename ProblemType>
typename ProblemType::Table_t *fl::ml::DualtreeDfs<ProblemType>::query_table() {
  return query_table_;
}

template<typename ProblemType>
typename ProblemType::Table_t *
fl::ml::DualtreeDfs<ProblemType>::reference_table() {
  return reference_table_;
}

template<typename ProblemType>
void fl::ml::DualtreeDfs<ProblemType>::ResetStatistic() {
  query_statistics_.reset(new std::vector<
      typename ProblemType::Statistic_t>(query_table_->num_of_nodes()));
  reference_statistics_=query_statistics_;
  ResetStatisticRecursion_(query_table_->get_tree(), 
      query_table_, query_statistics_.get());
}

template<typename ProblemType>
void fl::ml::DualtreeDfs<ProblemType>::Init(ProblemType &problem_in) {
  problem_ = &problem_in;
  query_table_ = problem_->query_table();
  reference_table_ = problem_->reference_table();
  ResetStatistic();
// ************** This has to be fixed, it is redundant
  if (query_table_ != reference_table_) {
    reference_statistics_.reset( new
        std::vector<
            typename ProblemType::Statistic_t>(
               reference_table_->num_of_nodes()));
    ResetStatisticRecursion_(reference_table_->get_tree(), 
        reference_table_, reference_statistics_.get());
  }
}

template<typename ProblemType>
template<typename MetricType>
void fl::ml::DualtreeDfs<ProblemType>::Compute(
  const MetricType &metric,
  typename ProblemType::Result_t *query_results) {

  // Allocate space for storing the final results.
  query_results->Init(query_table_->n_entries());

  // Call the algorithm computation.
  GenRange<double> squared_distance_range =
    (query_table_->get_node_bound(query_table_->get_tree())).RangeDistanceSq(
      metric,
      reference_table_->get_node_bound
      (reference_table_->get_tree()));

  PreProcess_(query_table_->get_tree());
  PreProcessReferenceTree_(reference_table_->get_tree());
  problem_->global().reference_statistic()=reference_statistics_.get();
  DualtreeCanonical_(metric,
                     query_table_->get_tree(),
                     reference_table_->get_tree(),
                     1.0 - problem_->global().probability(),
                     squared_distance_range,
                     query_results);
  PostProcess_(metric, query_table_->get_tree(), query_results);
}

template<typename ProblemType>
void fl::ml::DualtreeDfs<ProblemType>::ResetStatisticRecursion_(
  typename ProblemType::Table_t::Tree_t *node,
  typename ProblemType::Table_t * table, 
  std::vector<typename ProblemType::Statistic_t> *statistics) {
  index_t node_id=table->get_node_id(node);
  statistics->at(node_id).SetZero();
  if (table->node_is_leaf(node) == false) {
    ResetStatisticRecursion_(table->get_node_left_child(node), table, statistics);
    ResetStatisticRecursion_(table->get_node_right_child(node), table, statistics);
  }
}

template<typename ProblemType>
void fl::ml::DualtreeDfs<ProblemType>::PreProcessReferenceTree_(
  typename ProblemType::Table_t::Tree_t *rnode) {

  index_t rnode_id = reference_table_->get_node_id(rnode);
  typename ProblemType::Table_t::TreeIterator rnode_it =
    reference_table_->get_node_iterator(rnode);
  typename ProblemType::Statistic_t &rnode_stat=
    reference_statistics_->at(rnode_id);
  if (reference_table_->node_is_leaf(rnode)) {
    rnode_stat.Init(rnode_it);
  }
  else {

    // Get the left and the right children.
    typename ProblemType::Table_t::Tree_t *rnode_left_child =
      reference_table_->get_node_left_child(rnode);
    typename ProblemType::Table_t::Tree_t *rnode_right_child =
      reference_table_->get_node_right_child(rnode);

    // Recurse to the left and the right.
    PreProcessReferenceTree_(rnode_left_child);
    PreProcessReferenceTree_(rnode_right_child);

    // Build the node stat by combining those owned by the children.
    typename ProblemType::Statistic_t &rnode_left_child_stat =
      reference_statistics_->at(reference_table_->get_node_id(rnode_left_child));
    typename ProblemType::Statistic_t &rnode_right_child_stat =
      reference_statistics_->at(reference_table_->get_node_id(rnode_right_child));

    rnode_stat.Init(rnode_it, rnode_left_child_stat,
                    rnode_right_child_stat);
  }
}

template<typename ProblemType>
void fl::ml::DualtreeDfs<ProblemType>::PreProcess_(
  typename ProblemType::Table_t::Tree_t *qnode) {

  typename ProblemType::Statistic_t &qnode_stat =
    query_statistics_->at(query_table_->get_node_id(qnode));
  qnode_stat.SetZero();

  if (!query_table_->node_is_leaf(qnode)) {
    PreProcess_(query_table_->get_node_left_child(qnode));
    PreProcess_(query_table_->get_node_right_child(qnode));
  }
}

template<typename ProblemType>
template<typename MetricType>
void fl::ml::DualtreeDfs<ProblemType>::DualtreeBase_(
  MetricType &metric,
  typename ProblemType::Table_t::Tree_t *qnode,
  typename ProblemType::Table_t::Tree_t *rnode,
  typename ProblemType::Result_t *query_results) {
  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  typename ProblemType::Statistic_t &qnode_stat =
    query_statistics_->at(query_table_->get_node_id(qnode));
  qnode_stat.summary.StartReaccumulate();

  // Postponed object to hold each query contribution.
  typename ProblemType::Postponed_t query_contribution;

  // Get the query node iterator and the reference node iterator.
  typename ProblemType::Table_t::TreeIterator qnode_iterator =
    query_table_->get_node_iterator(qnode);
  typename ProblemType::Table_t::TreeIterator rnode_iterator =
    reference_table_->get_node_iterator(rnode);

  // Compute unnormalized sum for each query point.
  while (qnode_iterator.HasNext()) {

    // Get the query point and its real index.
    typename ProblemType::Point_t q_col;
    index_t q_index;
    qnode_iterator.Next(&q_col, &q_index);

    // Reset the temporary variable for accumulating each
    // reference point contribution.
    query_contribution.Init(reference_table_->get_node_count(rnode));

    // Incorporate the postponed information.
    query_results->ApplyPostponed(q_index, qnode_stat.postponed);

    // Reset the reference node iterator.
    rnode_iterator.Reset();
    if (qnode==rnode) {
      while (rnode_iterator.HasNext()) {
        // Get the reference point and accumulate contribution.
        typename ProblemType::Point_t r_col;
        index_t r_col_id;
        rnode_iterator.Next(&r_col, &r_col_id);
        if (q_index==r_col_id) {
          continue;
        }
        query_contribution.ApplyContribution(problem_->global(),
                                           metric, q_col, r_col);
      } // end of iterating over each reference point.
    } else  {
      while (rnode_iterator.HasNext()) {
        // Get the reference point and accumulate contribution.
        typename ProblemType::Point_t r_col;
        index_t r_col_id;
        rnode_iterator.Next(&r_col, &r_col_id);
        query_contribution.ApplyContribution(problem_->global(),
                                           metric, q_col, r_col);
      } // end of iterating over each reference point.
    }
    // Each query point has taken care of all reference points.
    query_results->ApplyPostponed(q_index, query_contribution);

    // Refine min and max summary statistics.
    qnode_stat.summary.Accumulate(*query_results, q_index);

  } // end of looping over each query point.

  // Clear postponed information.
  qnode_stat.postponed.SetZero();
}

template<typename ProblemType>
template<typename MetricType>
bool fl::ml::DualtreeDfs<ProblemType>::CanProbabilisticSummarize_(
  const MetricType &metric,
  typename ProblemType::Table_t::Tree_t *qnode,
  typename ProblemType::Table_t::Tree_t *rnode,
  double failure_probability,
  typename ProblemType::Delta_t &delta,
  typename ProblemType::Result_t *query_results) {

  typename ProblemType::Statistic_t &qnode_stat =
    query_statistics_->at(query_table_->get_node_id(qnode));
  typename ProblemType::Summary_t new_summary(qnode_stat.summary);
  new_summary.ApplyPostponed(qnode_stat.postponed);
  new_summary.ApplyDelta(delta);

  return new_summary.CanProbabilisticSummarize(metric,
         problem_->global(), delta, qnode, rnode, failure_probability,
         query_results);
}

template<typename ProblemType>
bool fl::ml::DualtreeDfs<ProblemType>::CanSummarize_(
  typename ProblemType::Table_t::Tree_t *qnode,
  typename ProblemType::Table_t::Tree_t *rnode,
  const typename ProblemType::Delta_t &delta,
  typename ProblemType::Result_t *query_results) {

  typename ProblemType::Statistic_t &qnode_stat =
    query_statistics_->at(query_table_->get_node_id(qnode));
  typename ProblemType::Summary_t new_summary(qnode_stat.summary);
  new_summary.ApplyPostponed(qnode_stat.postponed);
  new_summary.ApplyDelta(delta);

  return new_summary.CanSummarize(problem_->global(), delta, qnode, rnode,
                                  query_results);
}

template<typename ProblemType>
template<typename GlobalType>
void fl::ml::DualtreeDfs<ProblemType>::ProbabilisticSummarize_(
  const GlobalType &global,
  typename ProblemType::Table_t::Tree_t *qnode,
  double failure_probability,
  const typename ProblemType::Delta_t &delta,
  typename ProblemType::Result_t *query_results) {

  query_results->ApplyProbabilisticDelta(global, qnode, failure_probability,
                                         delta);
}

template<typename ProblemType>
void fl::ml::DualtreeDfs<ProblemType>::Summarize_(
  typename ProblemType::Table_t::Tree_t *qnode,
  const typename ProblemType::Delta_t &delta,
  typename ProblemType::Result_t *query_results) {

  typename ProblemType::Statistic_t &qnode_stat =
    query_statistics_->at(query_table_->get_node_id(qnode));
  qnode_stat.postponed.ApplyDelta(delta, query_results);
}

template<typename ProblemType>
template<typename MetricType>
void fl::ml::DualtreeDfs<ProblemType>::Heuristic_(
  const MetricType &metric,
  typename ProblemType::Table_t::Tree_t *node,
  typename ProblemType::Table_t *node_table,
  typename ProblemType::Table_t::Tree_t *first_candidate,
  typename ProblemType::Table_t::Tree_t *second_candidate,
  typename ProblemType::Table_t *candidate_table,
  typename ProblemType::Table_t::Tree_t **first_partner,
  GenRange<double> &first_squared_distance_range,
  typename ProblemType::Table_t::Tree_t **second_partner,
  GenRange<double> &second_squared_distance_range) {

  GenRange<double> tmp_first_squared_distance_range =
    node_table->get_node_bound(node).RangeDistanceSq(
      metric,
      candidate_table->get_node_bound(first_candidate));
  GenRange<double> tmp_second_squared_distance_range =
    node_table->get_node_bound(node).RangeDistanceSq(
      metric,
      candidate_table->get_node_bound(second_candidate));

  if (tmp_first_squared_distance_range.lo <=
      tmp_second_squared_distance_range.lo) {
    *first_partner = first_candidate;
    first_squared_distance_range = tmp_first_squared_distance_range;
    *second_partner = second_candidate;
    second_squared_distance_range = tmp_second_squared_distance_range;
  }
  else {
    *first_partner = second_candidate;
    first_squared_distance_range = tmp_second_squared_distance_range;
    *second_partner = first_candidate;
    second_squared_distance_range = tmp_first_squared_distance_range;
  }
}

template<typename ProblemType>
template<typename MetricType>
bool fl::ml::DualtreeDfs<ProblemType>::DualtreeCanonical_(
  MetricType &metric,
  typename ProblemType::Table_t::Tree_t *qnode,
  typename ProblemType::Table_t::Tree_t *rnode,
  double failure_probability,
  const GenRange<double> &squared_distance_range,
  typename ProblemType::Result_t *query_results) {

  // Compute the delta change.
  typename ProblemType::Delta_t delta;
  delta.DeterministicCompute(metric, problem_->global(), qnode, rnode,
                             squared_distance_range);

  // If it is prunable, then summarize and return.
  if (CanSummarize_(qnode, rnode, delta, query_results)) {
    Summarize_(qnode, delta, query_results);
    return true;
  }
  else if (failure_probability > 1e-6) {

    // Try Monte Carlo.
    if (CanProbabilisticSummarize_(metric, qnode, rnode,
                                   failure_probability,
                                   delta, query_results)) {
      ProbabilisticSummarize_(problem_->global(), qnode,
                              failure_probability,
                              delta, query_results);
      return false;
    }
  }

  // If it is not prunable and the query node is a leaf,
  if (query_table_->node_is_leaf(qnode)) {

    bool exact_compute = true;
    if (reference_table_->node_is_leaf(rnode)) {
      DualtreeBase_(metric, qnode, rnode, query_results);
    } // qnode is leaf, rnode is leaf.
    else {
      typename ProblemType::Table_t::Tree_t *rnode_first;
      GenRange<double> squared_distance_range_first,
      squared_distance_range_second;
      typename ProblemType::Table_t::Tree_t *rnode_second;
      Heuristic_(metric, qnode, query_table_,
                 reference_table_->get_node_left_child(rnode),
                 reference_table_->get_node_right_child(rnode),
                 reference_table_,
                 &rnode_first, squared_distance_range_first,
                 &rnode_second, squared_distance_range_second);

      // Recurse.
      bool rnode_first_exact =
        DualtreeCanonical_(metric,
                           qnode,
                           rnode_first,
                           failure_probability / 2.0,
                           squared_distance_range_first,
                           query_results);

      bool rnode_second_exact =
        DualtreeCanonical_(metric,
                           qnode,
                           rnode_second,
                           (rnode_first_exact) ?
                           failure_probability : failure_probability / 2.0,
                           squared_distance_range_second,
                           query_results);
      exact_compute = rnode_first_exact && rnode_second_exact;
    } // qnode is leaf, rnode is not leaf.
    return exact_compute;
  } // end of query node being a leaf.

  // If we are here, we have to split the query.
  bool exact_compute_nonleaf_qnode = true;

  // Get the current query node statistic.
  typename ProblemType::Statistic_t &qnode_stat =
    query_statistics_->at(query_table_->get_node_id(qnode));

  // Left and right nodes of the query node and their statistic.
  typename ProblemType::Table_t::Tree_t *qnode_left =
    query_table_->get_node_left_child(qnode);
  typename ProblemType::Table_t::Tree_t *qnode_right =
    query_table_->get_node_right_child(qnode);
  typename ProblemType::Statistic_t &qnode_left_stat =
    query_statistics_->at(query_table_->get_node_id(qnode_left));
  typename ProblemType::Statistic_t &qnode_right_stat =
    query_statistics_->at(query_table_->get_node_id(qnode_right));

  // Push down postponed and clear.
  qnode_left_stat.postponed.ApplyPostponed(qnode_stat.postponed);
  qnode_right_stat.postponed.ApplyPostponed(qnode_stat.postponed);
  qnode_stat.postponed.SetZero();

  if (reference_table_->node_is_leaf(rnode)) {
    typename ProblemType::Table_t::Tree_t *qnode_first;
    GenRange<double>
    squared_distance_range_first, squared_distance_range_second;
    typename ProblemType::Table_t::Tree_t *qnode_second;
    Heuristic_(metric, rnode, reference_table_, qnode_left, qnode_right,
               query_table_, &qnode_first, squared_distance_range_first,
               &qnode_second, squared_distance_range_second);

    // Recurse.
    bool first_qnode_exact = DualtreeCanonical_(metric,
                             qnode_first,
                             rnode,
                             failure_probability,
                             squared_distance_range_first,
                             query_results);
    bool second_qnode_exact = DualtreeCanonical_(metric,
                              qnode_second,
                              rnode,
                              failure_probability,
                              squared_distance_range_second,
                              query_results);
    exact_compute_nonleaf_qnode = first_qnode_exact && second_qnode_exact;
  } // qnode is not leaf, rnode is leaf.

  else {
    typename ProblemType::Table_t::Tree_t *rnode_first;
    GenRange<double>
    squared_distance_range_first, squared_distance_range_second;
    typename ProblemType::Table_t::Tree_t *rnode_second;
    Heuristic_(metric,
               qnode_left,
               query_table_,
               reference_table_->get_node_left_child(rnode),
               reference_table_->get_node_right_child(rnode),
               reference_table_,
               &rnode_first,
               squared_distance_range_first,
               &rnode_second, squared_distance_range_second);

    // Recurse.
    bool qnode_left_rnode_first_exact = DualtreeCanonical_(
                                          metric,
                                          qnode_left,
                                          rnode_first,
                                          failure_probability / 2.0,
                                          squared_distance_range_first,
                                          query_results);
    bool qnode_left_rnode_second_exact = DualtreeCanonical_(
                                           metric,
                                           qnode_left,
                                           rnode_second,
                                           (qnode_left_rnode_first_exact) ?
                                           failure_probability : failure_probability / 2.0,
                                           squared_distance_range_second,
                                           query_results);

    Heuristic_(metric,
               qnode_right,
               query_table_,
               reference_table_->get_node_left_child(rnode),
               reference_table_->get_node_right_child(rnode),
               reference_table_,
               &rnode_first,
               squared_distance_range_first,
               &rnode_second,
               squared_distance_range_second);

    // Recurse.
    bool qnode_right_rnode_first_exact = DualtreeCanonical_(
                                           metric,
                                           qnode_right,
                                           rnode_first,
                                           failure_probability / 2.0,
                                           squared_distance_range_first,
                                           query_results);
    bool qnode_right_rnode_second_exact = DualtreeCanonical_(
                                            metric,
                                            qnode_right,
                                            rnode_second,
                                            (qnode_right_rnode_first_exact) ?
                                            failure_probability : failure_probability / 2.0,
                                            squared_distance_range_second,
                                            query_results);

    // Merge the boolean results.
    exact_compute_nonleaf_qnode = qnode_left_rnode_first_exact &&
                                  qnode_left_rnode_second_exact && qnode_right_rnode_first_exact &&
                                  qnode_right_rnode_second_exact;

  } // qnode is not leaf, rnode is not leaf.

  // Reset summary results of the current query node.
  qnode_stat.summary.StartReaccumulate();
  qnode_stat.summary.Accumulate(qnode_left_stat.summary,
                                qnode_left_stat.postponed);
  qnode_stat.summary.Accumulate(qnode_right_stat.summary,
                                qnode_right_stat.postponed);
  return exact_compute_nonleaf_qnode;
}

template<typename ProblemType>
template<typename MetricType>
void fl::ml::DualtreeDfs<ProblemType>::PostProcess_(
  const MetricType &metric,
  typename ProblemType::Table_t::Tree_t *qnode,
  typename ProblemType::Result_t *query_results) {

  typename ProblemType::Statistic_t &qnode_stat =
    query_statistics_->at(query_table_->get_node_id(qnode));

  if (query_table_->node_is_leaf(qnode)) {

    typename ProblemType::Table_t::TreeIterator qnode_iterator =
      query_table_->get_node_iterator(qnode);

    while (qnode_iterator.HasNext()) {
      typename ProblemType::Point_t q_col;
      index_t q_index;
      qnode_iterator.Next(&q_col, &q_index);
      query_results->ApplyPostponed(q_index, qnode_stat.postponed);
      query_results->PostProcess(metric, q_index, problem_->global(),
                                 problem_->is_monochromatic());
    }
    qnode_stat.postponed.SetZero();
  }
  else {
    typename ProblemType::Table_t::Tree_t *qnode_left =
      query_table_->get_node_left_child(qnode);
    typename ProblemType::Table_t::Tree_t *qnode_right =
      query_table_->get_node_right_child(qnode);
    typename ProblemType::Statistic_t &qnode_left_stat =
      query_statistics_->at(query_table_->get_node_id(qnode_left));
    typename ProblemType::Statistic_t &qnode_right_stat =
      query_statistics_->at(query_table_->get_node_id(qnode_right));

    qnode_left_stat.postponed.ApplyPostponed(qnode_stat.postponed);
    qnode_right_stat.postponed.ApplyPostponed(qnode_stat.postponed);
    qnode_stat.postponed.SetZero();

    PostProcess_(metric, qnode_left,  query_results);
    PostProcess_(metric, qnode_right, query_results);
  }
}

#endif
