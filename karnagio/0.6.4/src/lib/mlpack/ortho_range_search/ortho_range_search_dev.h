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
/** @file ortho_range_search.h
 *
 *  This file contains an implementation of a tree-based algorithm for
 *  orthogonal range search.
 *
 *  @author Dongryeol Lee (dongryel)
 */
#ifndef MLPACK_ORTHO_RANGE_SEARCH_ORTHO_RANGE_SEARCH_DEV_H
#define MLPACK_ORTHO_RANGE_SEARCH_ORTHO_RANGE_SEARCH_DEV_H

#include "fastlib/dense/matrix.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "boost/utility.hpp"
#include <time.h>
#include "mlpack/ortho_range_search/ortho_range_search.h"

namespace fl {
namespace ml {

template <typename OrthoArgs>
template<typename OutputTableType>
void OrthoRangeSearch<OrthoArgs>::OrthoSlowRangeSearch_(WindowTree_t *search_window_node,
    WindowTable_t &window_table,
    ReferenceTree_t *reference_node,
    ReferenceTable_t &reference_table,
    const index_t &start_dim,
    const index_t &end_dim,
    OutputTableType &candidate_points) {

  PruneStatus prune_flag;

  // Loop over each search window...
  typename WindowTable_t::TreeIterator window_iterator =
    window_table.get_node_iterator(search_window_node);
  index_t limit = reference_table.n_attributes();
  do {

    // Get the current search window.
    typename WindowTable_t::Point_t window;
    index_t window_id;
    window_iterator.Next(&window, &window_id);

    // Loop over each reference point...
    typename ReferenceTable_t::TreeIterator reference_iterator =
      reference_table.get_node_iterator(reference_node);
    do {

      // The default pruning status is SUBSUME.
      prune_flag = SUBSUME;

      // Get the current reference point.
      typename ReferenceTable_t::Point_t reference_pt;
      index_t reference_id;
      reference_iterator.Next(&reference_pt, &reference_id);

      // Loop over each dimension...
      for (index_t d = start_dim; d <= end_dim; d++) {

        // Determine which one of the two cases we have: EXCLUDE,
        // SUBSUME.

        // First the EXCLUDE case: when dist is above the upper
        // bound distance of this dimension, or dist is below
        // the lower bound distance of this dimension
        if (reference_pt[d] > window[d + limit] ||
            reference_pt[d] < window[d]) {
          prune_flag = EXCLUDE;
          break;
        }
      } // end of looping over dimensions...

      // Set each point result depending on the flag...
      typename OutputTableType::Point_t point;
      candidate_points.get(window_id, &point);
      point.set(reference_id,
                (prune_flag == SUBSUME));

    }
    while (reference_iterator.HasNext()); // end of iterating
    // over reference points...

  }
  while (window_iterator.HasNext()); // end of iterating over
  // search window...
}

template<typename OrthoArgs>
template<typename OutputTableType>
void OrthoRangeSearch<OrthoArgs>::OrthoRangeSearch_(WindowTree_t *search_window_node,
    WindowTable_t &window_table,
    ReferenceTree_t *reference_node,
    ReferenceTable_t &reference_table,
    index_t start_dim, index_t end_dim,
    OutputTableType &candidate_points) {

  PruneStatus prune_flag = SUBSUME;
  const typename WindowTree_t::Bound_t &search_window_node_bound =
    window_table.get_node_bound(search_window_node);

  // loop over each dimension to determine inclusion/exclusion by
  // determining the lower and the upper bound distance per each
  // dimension for the given reference node, kn
  for (index_t d = start_dim; d <= end_dim; d++) {

    const GenRange<double> &reference_node_dir_range =
      reference_table.get_node_bound(reference_node).get(d);

    // determine which one of the three cases we have: EXCLUDE,
    // SUBSUME, or INCONCLUSIVE.

    // First the EXCLUDE case: when min value is above the
    // upper bound distance of this dimension, or max value is
    // below the lower bound distance of this dimension
    index_t lo = d;
    index_t hi = d + reference_table.n_attributes();
    if (reference_node_dir_range.lo > search_window_node_bound[hi] ||
        reference_node_dir_range.hi < search_window_node_bound[lo]) {
      return;
    }
    // otherwise, check for SUBSUME case
    else
      if (search_window_node_bound[lo] <=
          reference_node_dir_range.lo &&
          reference_node_dir_range.hi <=
          search_window_node_bound[hi]) {
      }
    // if any dimension turns out to be inconclusive, then break.
      else {
        prune_flag = INCONCLUSIVE;
        break;
      }
  } // end of iterating over each dimension.

  // In case of subsume, then add all points owned by this node to
  // candidates - note that subsume prunes cannot be performed
  // always in batch query. This will be addressed very soon.
  if (window_table.get_node_count(search_window_node) == 1 &&
      prune_flag == SUBSUME) {
    typename WindowTable_t::TreeIterator window_iterator =
      window_table.get_node_iterator(search_window_node);
    do {
      typename WindowTable_t::Point_t window;
      index_t window_id;
      window_iterator.Next(&window, &window_id);

      typename ReferenceTable_t::TreeIterator reference_iterator =
        reference_table.get_node_iterator(reference_node);
      do {
        typename ReferenceTable_t::Point_t reference_pt;
        index_t reference_id;
        reference_iterator.Next(&reference_pt, &reference_id);
        typename OutputTableType::Point_t point;
        candidate_points.get(window_id, &point);
        point.set(reference_id, 1);
      }
      while (reference_iterator.HasNext());
    }
    while (window_iterator.HasNext());
    return;
  }
  else {
    if (window_table.node_is_leaf(search_window_node)) {

      // If both the search window and the reference nodes are
      // leaves, then compute exhaustively.
      if (reference_table.node_is_leaf(reference_node)) {
        OrthoSlowRangeSearch_
        (search_window_node, window_table, reference_node,
         reference_table, start_dim, end_dim, candidate_points);
      }
      // If the reference node can be expanded, then do so.
      else {
        OrthoRangeSearch_(search_window_node, window_table,
                          reference_table.get_node_left_child(reference_node),
                          reference_table, start_dim, end_dim, candidate_points);
        OrthoRangeSearch_(search_window_node, window_table,
                          reference_table.get_node_right_child(reference_node),
                          reference_table, start_dim, end_dim, candidate_points);
      }
    }
    else {

      // In this case, expand the query side.
      if (reference_table.node_is_leaf(reference_node)) {
        OrthoRangeSearch_(window_table.get_node_left_child(search_window_node),
                          window_table, reference_node, reference_table,
                          start_dim, end_dim, candidate_points);
        OrthoRangeSearch_(window_table.get_node_right_child(search_window_node),
                          window_table, reference_node, reference_table,
                          start_dim, end_dim, candidate_points);
      }

      // Otherwise, expand both query and the reference sides.
      else {
        OrthoRangeSearch_(window_table.get_node_left_child(search_window_node),
                          window_table, reference_table.get_node_left_child(reference_node),
                          reference_table, start_dim, end_dim, candidate_points);
        OrthoRangeSearch_(window_table.get_node_left_child(search_window_node),
                          window_table,
                          reference_table.get_node_right_child(reference_node),
                          reference_table, start_dim, end_dim, candidate_points);
        OrthoRangeSearch_(window_table.get_node_right_child(search_window_node),
                          window_table,
                          reference_table.get_node_left_child(reference_node),
                          reference_table, start_dim, end_dim, candidate_points);
        OrthoRangeSearch_(window_table.get_node_right_child(search_window_node),
                          window_table,
                          reference_table.get_node_right_child(reference_node),
                          reference_table, start_dim, end_dim, candidate_points);
      }
    }
  }
}

template<typename OrthoArgs>
template<typename OutputTableType>
void OrthoRangeSearch<OrthoArgs>::Compute(WindowTable_t &window_queries,
    const index_t &window_leaf_size,
    ReferenceTable_t &reference_points,
    const index_t &reference_leaf_size,
    OutputTableType *candidate_points) {

  // Build the tree of search windows.
  clock_t start = clock();
  typename WindowTable_t::template IndexArgs<fl::math::LMetric<2> > window_args;
  window_args.leaf_size = 1;
  window_queries.IndexData(window_args);

  // Build the tree for the reference points.
  typename ReferenceTable_t::template IndexArgs<fl::math::LMetric<2> > reference_window_args;
  reference_window_args.leaf_size = reference_leaf_size;
  reference_points.IndexData(reference_window_args);
  clock_t end = clock();
  printf("Took %g seconds to build.\n",
         ((double) end - start) / ((double) CLOCKS_PER_SEC));

  // Call the search algorithm.
  OrthoRangeSearch_(window_queries.get_tree(), window_queries,
                    reference_points.get_tree(), reference_points,
                    0, reference_points.n_attributes() - 1,
                    *candidate_points);
}
};
};

#endif
