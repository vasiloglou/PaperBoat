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
#ifndef FL_LITE_MLPACK_ALLKN_ALLKN_COMPUTATIONS_DEV_H_
#define FL_LITE_MLPACK_ALLKN_ALLKN_COMPUTATIONS_DEV_H_
#include "mlpack/allkn/allkn_computations.h"
#include <functional>
//#include <omp.h>
//#include <fastlib/omptl/omptl_algorithm>

namespace fl {
namespace ml {
template < typename NeighborContainerType,
typename DistanceContainerType,
typename IndexContainerType >
void ALLKNMPL::UpdateOperator1::type::InitTempNeighbors(
  NeighborContainerType *neighbors,
  index_t ind,
  index_t offset,
  DistanceContainerType &distances,
  IndexContainerType &indices) {
  neighbors->operator[](ind).first=distances[offset+ind],
                                   neighbors->operator[](ind).second=indices[offset+ind];

}

template < typename Container1Type,
typename Container2Type,
typename CalcPrecisionType
>
void ALLKNMPL::UpdateOperator1::type::Insert(Container1Type *indices,
    Container2Type *distances,
    index_t neighbor_position,
    index_t neighbor,
    index_t neighbor_rank,
    CalcPrecisionType distance) {
  indices->operator[](neighbor_position+neighbor_rank) = neighbor;
  distances->operator[](neighbor_position+neighbor_rank) = distance;
}

template < typename NeighborContainerType,
typename DistanceContainerType,
typename IndexContainerType >
void ALLKNMPL::UpdateOperator2::type::InitTempNeighbors(
  NeighborContainerType *neighbors,
  index_t ind,
  index_t offset,
  DistanceContainerType &distances,
  IndexContainerType &indices) {
}

template < typename Container1Type,
typename Container2Type,
typename CalcPrecisionType
>
void ALLKNMPL::UpdateOperator2::type::Insert(
  Container1Type *indices,
  Container2Type *distances,
  index_t p1,
  index_t p2,
  index_t neighbor_rank,
  CalcPrecisionType distance) {

    indices->push_back(std::make_pair(p1, p2));
    distances->push_back(distance);
}

template<typename TypeVector>
template<typename ArgsType>
void SingleThreadAllKNTraversal<TypeVector>::
QueryBinaryReferenceBinary::ComputeDualNeighborsRecursion(ArgsType *args) {
  static const bool IS_RANGE_NEIGHBORS = ArgsType::IS_RANGE_NEIGHBORS;
  DEBUG_ASSERT(args->query_node() != NULL);
  DEBUG_ASSERT_MSG(args->reference_node() != NULL, "reference node is null");
  // Make sure the bounding information is correct
  // This is commented out for 2 reasons. a) Precision problems causes
  // this to fail. b) This is an expensive debug as it requires computation.
  //DEBUG_ASSERT(lower_bound_distance == MinNodeDistSq_(query_node,
  //    reference_node));
  bool prune_or_not;
  CalcPrecision_t dist_so_far=
    args->stat()->at(args->query_table()->get_node_id(args->query_node())).dist_so_far();

  if (mode == NearestNeighborAllKN) {
    prune_or_not = args->bound_distance() > dist_so_far;
  }
  if (mode == FurthestNeighborAllKN) {
    prune_or_not =
      args->bound_distance() < dist_so_far;
  }

  if (prune_or_not) {
    // Pruned by distance
    if (args->info() != NULL) {
      args->info()->num_of_prunes()++;
    }
  }
  else {
    if (args->query_table()->node_is_leaf(args->query_node()) &&
        args->reference_table()->node_is_leaf(args->reference_node())) {
      // Base Case
      if (IsMonochromatic && args->reference_node() == args->query_node()) {
        SingleThreadDualTreeAllKNBaseTraitMono_t::ComputeBaseCase(args);
      }
      else {
        SingleThreadDualTreeAllKNBaseTraitBi_t::ComputeBaseCase(args);
      }
    }
    else {
      if (args->query_table()->node_is_leaf(args->query_node())) {
        // Only query is a leaf
        // We'll order the computation by distance
        ReferenceTree_t *left = args->
                                reference_table()->get_node_left_child(
                                  args->reference_node());

        ReferenceTree_t *right = args->
                                 reference_table()->get_node_right_child(
                                   args->reference_node());

        CalcPrecision_t left_distance;
        CalcPrecision_t right_distance;
        bool go_left=false;
        if (mode == NearestNeighborAllKN) {
          left_distance =
            args->query_table()->get_node_bound(args->query_node()).
            MinDistanceSq(*args->metric(),
                          args->reference_table()->get_node_bound(left));

          right_distance = args->query_table()->
                           get_node_bound(args->query_node()).
                           MinDistanceSq(*args->metric(),
                                         args->reference_table()->get_node_bound(right));
          go_left = left_distance < right_distance;
        }

        if (mode == FurthestNeighborAllKN) {
          left_distance =
            args->query_table()->get_node_bound(args->query_node()).
            MaxDistanceSq(*args->metric(),
                          args->reference_table()->get_node_bound(left));

          right_distance = args->query_table()->
                           get_node_bound(args->query_node()).
                           MaxDistanceSq(*args->metric(),
                                         args->reference_table()->get_node_bound(right));
          go_left = left_distance > right_distance;
        }

        if (go_left) {
          args->bound_distance() = left_distance;
          args->reference_node() = left;
          ComputeDualNeighborsRecursion(args);
          args->reference_node() = right;
          args->bound_distance() = right_distance;
          ComputeDualNeighborsRecursion(args);
        }
        else {
          args->reference_node() = right;
          args->bound_distance() = right_distance;
          ComputeDualNeighborsRecursion(args);
          args->reference_node() = left;
          args->bound_distance() = left_distance;
          ComputeDualNeighborsRecursion(args);
        }
      }
      else {
        if (args->reference_table()->node_is_leaf(args->reference_node())) {
          // Only reference is a leaf
          QueryTree_t *query_node = args->query_node();
          QueryTree_t *left = args->query_table()->get_node_left_child(
                                args->query_node());
          QueryTree_t *right = args->query_table()->get_node_right_child(
                                 args->query_node());
          CalcPrecision_t left_distance;
          CalcPrecision_t right_distance;

          if (mode == NearestNeighborAllKN) {
            left_distance =
              args->query_table()->get_node_bound(left).
              MinDistanceSq(*args->metric(),
                            args->reference_table()->get_node_bound(
                              args->reference_node()));

            right_distance =
              args->query_table()->get_node_bound(right).
              MinDistanceSq(*args->metric(),
                            args->reference_table()->get_node_bound(
                              args->reference_node()));
          }
          if (mode == FurthestNeighborAllKN) {
            left_distance =
              args->query_table()->get_node_bound(left).
              MaxDistanceSq(*args->metric(),
                            args->reference_table()->get_node_bound(
                              args->reference_node()));

            right_distance =
              args->query_table()->get_node_bound(right).
              MaxDistanceSq(*args->metric(),
                            args->reference_table()->get_node_bound(
                              args->reference_node()));
          }

          args->query_node() = left;
          args->bound_distance() = left_distance;
          ComputeDualNeighborsRecursion(args);

          args->query_node() = right;
          args->bound_distance() = right_distance;
          ComputeDualNeighborsRecursion(args);

          // We need to update the upper bound based on the new upper bounds of
          // the children
          
          if (mode == NearestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
             args->stat()->at(
                args->query_table()->get_node_id(query_node)).set_dist_so_far(
                  std::max(
                    args->stat()->at(
                      args->query_table()->get_node_id(left)).dist_so_far(),
                    args->stat()->at(
                      args->query_table()->get_node_id(right)).dist_so_far()));
          }
          if (mode == FurthestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
              args->stat()->at(
                args->query_table()->get_node_id(query_node)).set_dist_so_far(
                  std::min(
                    args->stat()->at(
                      args->query_table()->get_node_id(left)).dist_so_far(),
                    args->stat()->at(
                      args->query_table()->get_node_id(right)).dist_so_far()));
          }

        }
        else {
          // Recurse on both as above
          QueryTree_t *query_node = args->query_node();
          QueryTree_t *query_left = args->query_table()->get_node_left_child(
                                      args->query_node());
          QueryTree_t *query_right = args->query_table()->get_node_right_child(
                                       args->query_node());
          ReferenceTree_t *reference_left = args->reference_table()
                                            ->get_node_left_child(args->reference_node());
          ReferenceTree_t *reference_right = args->reference_table()
                                             ->get_node_right_child(args->reference_node());
          CalcPrecision_t left_distance;
          CalcPrecision_t right_distance;
          bool go_left;
          if (mode == NearestNeighborAllKN) {
            left_distance =
              args->query_table()->get_node_bound(query_left).
              MinDistanceSq(*args->metric(),
                            args->reference_table()->get_node_bound(reference_left));

            right_distance =
              args->query_table()->get_node_bound(query_left).
              MinDistanceSq(*args->metric(),
                            args->reference_table()->get_node_bound(reference_right));
            go_left = left_distance < right_distance;
          }
          if (mode == FurthestNeighborAllKN) {
            left_distance =
              args->query_table()->get_node_bound(query_left).
              MaxDistanceSq(*args->metric(),
                            args->reference_table()->get_node_bound(reference_left));

            right_distance =
              args->query_table()->get_node_bound(query_left).
              MaxDistanceSq(*args->metric(),
                            args->reference_table()->get_node_bound(reference_right));
            go_left = left_distance > right_distance;
          }


          if (go_left) {
            args->query_node() = query_left;
            args->reference_node() = reference_left;
            args->bound_distance() = left_distance;
            ComputeDualNeighborsRecursion(args);

            args->query_node() = query_left;
            args->reference_node() = reference_right;
            args->bound_distance() = right_distance;
            ComputeDualNeighborsRecursion(args);
          }
          else {
            args->query_node() = query_left;
            args->reference_node() = reference_right;
            args->bound_distance() = right_distance;
            ComputeDualNeighborsRecursion(args);
            args->query_node() = query_left;
            args->reference_node() = reference_left;
            args->bound_distance() = left_distance;
            ComputeDualNeighborsRecursion(args);
          }
          if (mode == NearestNeighborAllKN) {
            left_distance =
              args->query_table()->get_node_bound(query_right).
              MinDistanceSq(*args->metric(),
                            args->reference_table()->
                            get_node_bound(reference_left));
            right_distance =
              args->query_table()->get_node_bound(query_right).
              MinDistanceSq(*args->metric(),
                            args->reference_table()->get_node_bound(
                              reference_right));
            go_left = left_distance < right_distance;
          }

          if (mode == FurthestNeighborAllKN) {
            left_distance =
              args->query_table()->get_node_bound(query_right).
              MaxDistanceSq(*args->metric(),
                            args->reference_table()->
                            get_node_bound(reference_left));
            right_distance =
              args->query_table()->get_node_bound(query_right).
              MaxDistanceSq(*args->metric(),
                            args->reference_table()->get_node_bound(
                              reference_right));
            go_left = left_distance > right_distance;
          }


          if (go_left) {
            args->query_node() = query_right;
            args->reference_node() = reference_left;
            args->bound_distance() = left_distance;
            ComputeDualNeighborsRecursion(args);
            args->query_node() = query_right;
            args->reference_node() = reference_right;
            args->bound_distance() = right_distance;
            ComputeDualNeighborsRecursion(args);
          }
          else {
            args->query_node() = query_right;
            args->reference_node() = reference_right;
            args->bound_distance() = right_distance;
            ComputeDualNeighborsRecursion(args);
            args->query_node() = query_right;
            args->reference_node() = reference_left;
            args->bound_distance() = left_distance;
            ComputeDualNeighborsRecursion(args);
          }

          // Update the bound as above
          if (mode == NearestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
             args->stat()->at(
                args->query_table()->get_node_id(query_node)).set_dist_so_far(
                  std::max(
                    args->stat()->at(
                      args->query_table()->get_node_id(query_left)).dist_so_far(),
                    args->stat()->at(
                      args->query_table()->get_node_id(query_right)).dist_so_far()));
          }
          if (mode == FurthestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
             args->stat()->at(
                args->query_table()->get_node_id(query_node)).set_dist_so_far(
                  std::min(
                    args->stat()->at(
                      args->query_table()->get_node_id(query_left)).dist_so_far(),
                    args->stat()->at(
                      args->query_table()->get_node_id(query_right)).dist_so_far()));
          }
        }
      }
    }
  }
}

template<typename TypeVector>
template<typename ArgsType>
void SingleThreadAllKNTraversal<TypeVector>::
QueryBinaryReferenceBinary::ComputeNaive(ArgsType *args) {
  DEBUG_ASSERT(args->query_node() != NULL);
  DEBUG_ASSERT_MSG(args->reference_node() != NULL, "reference node is null");
  // Base Case
  if (IsMonochromatic == true) {
    SingleThreadDualTreeAllKNBaseTraitMono_t::ComputeBaseCase(args);
  }
  else {
    SingleThreadDualTreeAllKNBaseTraitBi_t::ComputeBaseCase(args);
  }
}

template<typename TypeVector>
template<typename ArgsType>
void SingleThreadAllKNTraversal<TypeVector>::
QueryBinaryReferenceBinary::ComputeAllSingleNeighborsRecursion(ArgsType *args) {
  static const bool IS_RANGE_NEIGHBORS = ArgsType::IS_RANGE_NEIGHBORS;
  if (args->query_table()->get_tree() != NULL) {
    typename QueryTable_t::TreeIterator it(*(args->query_table()),
                                           args->query_table()->get_tree());
    while (it.HasNext()) {
      if (mode == NearestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
        args->dist_so_far() = std::numeric_limits<CalcPrecision_t>::max();
      }
      if (mode == FurthestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
        args->dist_so_far() = -std::numeric_limits<CalcPrecision_t>::max();
      }
      args->reference_node() = args->reference_table()->get_tree();
      QueryPoint_t query_point;
      index_t query_point_id;
      it.Next(&query_point, &query_point_id);
      args->query_point() = &query_point;
      args->query_point_id() = query_point_id;
      ComputeSingleNeighborsRecursion(args);
    }
  }
  else {
    index_t num_of_entries = args->query_table()->n_entries();
    const QueryTable_t *table = args->query_table();
    for (index_t i = 0; i <= num_of_entries; i++) {
      if (mode == NearestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
        args->dist_so_far() = std::numeric_limits<CalcPrecision_t>::max();
      }
      if (mode == FurthestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
        args->dist_so_far() = -std::numeric_limits<CalcPrecision_t>::max();
      }

      args->reference_node() = args->reference_table()->get_tree();
      QueryPoint_t query_point;
      table->get(i, &query_point);
      args->query_point() = &query_point;
      args->query_point_id() = i;
      ComputeSingleNeighborsRecursion(args);
    }
  }
}


template<typename TypeVector>
template<typename ArgsType>
void SingleThreadAllKNTraversal<TypeVector>::
QueryBinaryReferenceBinary::ComputeSingleNeighborsRecursion(ArgsType *args) {
  static const bool IS_RANGE_NEIGHBORS = ArgsType::IS_RANGE_NEIGHBORS;
  DEBUG_ASSERT_MSG(args->reference_node() != NULL, "reference node is null");
  // Make sure the bounding information is correct

  // node->is_leaf() works as one would expect
  if (args->reference_table()->node_is_leaf(args->reference_node())) {
    // Base Case
    index_t nearest_size;
    if (Is1N == true) {
      nearest_size = 1;
    }
    else {
      nearest_size = static_cast<index_t>(args->kns());
    }
    typename ReferenceTable_t::TreeIterator reference_it(
      *args->reference_table(),
      args->reference_node());
    ReferencePoint_t reference_point;
    index_t reference_index;

    std::vector<std::pair<CalcPrecision_t, index_t> > neighbors;
    index_t ind;
    if (!IS_RANGE_NEIGHBORS) {
      if (Is1N == false) {
        neighbors.resize(nearest_size + reference_it.count());
        if (mode == NearestNeighborAllKN) {
          std::fill(neighbors.begin(), neighbors.end(),
                    std::make_pair(std::numeric_limits<CalcPrecision_t>::max(), -1));
        }
        else {
          if (mode == FurthestNeighborAllKN) {
            std::fill(neighbors.begin(), neighbors.end(),
                      std::make_pair(-std::numeric_limits<CalcPrecision_t>::max(), -1));
          }
        }
        ind = args->query_point_id() * static_cast<index_t>(args->kns());
      }
      else {
        ind = args->query_point_id();
      }
      if (Is1N == false && !IS_RANGE_NEIGHBORS) {
        for (index_t i = 0; i < static_cast<index_t>(args->kns()); i++) {
          boost::mpl::eval_if <
          boost::mpl::bool_<IS_RANGE_NEIGHBORS>,
          ALLKNMPL::UpdateOperator2,
          ALLKNMPL::UpdateOperator1
          >::type::InitTempNeighbors(
            &neighbors,
            i,
            ind,
            *(args->neighbor_distances()),
            *(args->neighbor_indices()));
        }
      }
    }
    //#pragma omp parallel for firstprivate(args)
    for (index_t i = 0; i < reference_it.count(); i++) {
      reference_it.get(i, &reference_point);
      reference_it.get_id(i, &reference_index);
      // Confirm that points do not identify themselves as neighbors
      // in the monochromatic case
      if (!(IsMonochromatic &&
            unlikely(reference_index == args->query_point_id()))) {
        // do not change the order of arguments it will fail in assymetric divergences
        CalcPrecision_t distance =
          args->metric()->DistanceSq(
            reference_point,
            *(args->query_point()));

        if (Is1N == false) {
          // we'll update the candidate
          bool push_back_or_not;
          if (mode == NearestNeighborAllKN) {
            push_back_or_not =
              distance < args->dist_so_far();
          }
          if (mode == FurthestNeighborAllKN) {
            push_back_or_not =
              distance > args->dist_so_far();
          }
          if (push_back_or_not) {
            neighbors[nearest_size+i].first = distance;
            neighbors[nearest_size+i].second = reference_index;
          }
        }
        else {
          // unfortunatelly for the nearest neighbor we need a synchronization
          // point
          // #pragma omp critical (one_nearest_neighbor_insert)
          {
            bool push_back_or_not;
            if (mode == NearestNeighborAllKN) {
              push_back_or_not =
                distance < args->dist_so_far();
            }
            if (mode == FurthestNeighborAllKN) {
              push_back_or_not =
                distance > args->dist_so_far();
            }

            if (push_back_or_not) {
              index_t neighbor_id;
              if (IS_RANGE_NEIGHBORS) {
                neighbor_id = args->query_point_id();
              }
              else {
                neighbor_id = ind;
              }
              boost::mpl::eval_if <
              boost::mpl::bool_<IS_RANGE_NEIGHBORS>,
              ALLKNMPL::UpdateOperator2,
              ALLKNMPL::UpdateOperator1
              >::type::Insert(args->neighbor_indices(),
                              args->neighbor_distances(),
                              neighbor_id,
                              reference_index,
                              0,
                              distance);
              if (!IS_RANGE_NEIGHBORS) {
                args->dist_so_far() = (*(args->neighbor_distances()))[ind];
              }
            }
          }
        }
      }
    } // for reference_index
    if (Is1N == false && !IS_RANGE_NEIGHBORS) {
      if (mode == NearestNeighborAllKN) {
        std::sort(neighbors.begin(), neighbors.end());
        /* std::nth_element(neighbors.begin(),
                         neighbors.begin()+static_cast<int>(args->kns())-1,
                         neighbors.end());
         */

      }
      if (mode == FurthestNeighborAllKN) {
        std::sort(neighbors.begin(), neighbors.end(),
                    std::greater<std::pair<CalcPrecision_t, index_t> >());

        /* std::nth_element(neighbors.begin(),
                         neighbors.begin()+static_cast<int>(args->kns())-1,
                         neighbors.end(),
                         std::greater<std::pair<CalcPrecision_t, index_t> >());
        */
      }
      //#pragma omp parallel if (!IS_RANGE_NEIGHBORS) firstprivate(args)
      {
        //#pragma omp for
        for (index_t i = 0; i < static_cast<index_t>(args->kns()); i++) {
          index_t neighbor_id;
          neighbor_id = ind;
          boost::mpl::eval_if <
          boost::mpl::bool_<IS_RANGE_NEIGHBORS>,
          ALLKNMPL::UpdateOperator2,
          ALLKNMPL::UpdateOperator1
          >::type::Insert(args->neighbor_indices(),
                          args->neighbor_distances(),
                          neighbor_id,
                          neighbors[i].second,
                          i,
                          neighbors[i].first);
        }
      }
      args->dist_so_far() = (*(args->neighbor_distances()))
                            [ind+static_cast<index_t>(args->kns())-1];
    }
    else {
    }
  } else {
    // We'll order the computation by distance
    ReferenceTree_t *left_node = args->reference_table()->get_node_left_child(
                                   args->reference_node());
    ReferenceTree_t *right_node = args->reference_table()->get_node_right_child(
                                    args->reference_node());
    CalcPrecision_t left_distance = 0;
    CalcPrecision_t right_distance = 0;
    bool go_left = false;
    if (IS_RANGE_NEIGHBORS) {
      go_left = true;
    }
    if (!IS_RANGE_NEIGHBORS) {
      if (mode == NearestNeighborAllKN) {
        left_distance =
          args->reference_table()->get_node_bound(left_node).MinDistanceSq(
            *args->metric(),
            *args->query_point());
        right_distance = args->reference_table()->get_node_bound(
                           right_node).MinDistanceSq(*args->metric(),
                                                     *args->query_point());
        go_left = left_distance < right_distance;
      }
      if (mode == FurthestNeighborAllKN) {
        left_distance =
          args->reference_table()->get_node_bound(left_node).MaxDistanceSq(
            *args->metric(),
            *args->query_point());
        right_distance = args->reference_table()->get_node_bound(
                           right_node).MaxDistanceSq(*args->metric(),
                                                     *args->query_point());
        go_left = left_distance > right_distance;
      }
    }
    if (go_left) {
      args->reference_node() = left_node;
      ComputeSingleNeighborsRecursion(args);
      bool prune_or_not = false;
      if (mode == NearestNeighborAllKN) {
        prune_or_not = args->dist_so_far() < right_distance;
      }
      if (mode == FurthestNeighborAllKN) {
        prune_or_not = args->dist_so_far() > right_distance;
      }

      if (prune_or_not) {
        args->info()->num_of_prunes()++;
        return;
      }
      args->reference_node() = right_node;
      ComputeSingleNeighborsRecursion(args);
    }
    else {
      args->reference_node() = right_node;
      ComputeSingleNeighborsRecursion(args);
      bool prune_or_not = false;
      if (mode == NearestNeighborAllKN) {
        prune_or_not = args->dist_so_far() < left_distance;
      }
      if (mode == FurthestNeighborAllKN) {
        prune_or_not = args->dist_so_far() > left_distance;
      }

      if (prune_or_not) {
        args->info()->num_of_prunes()++;
        return;
      }
      args->reference_node() = left_node;
      ComputeSingleNeighborsRecursion(args);
    }
  }
}

template<typename TypeVector>
template<typename ArgsType>
void SingleThreadAllKNTraversal<TypeVector>::
QueryBinaryReferenceBinary::ComputeDualNeighborsProgressive(
  std::deque<ArgsType> *trace,
  index_t max_trace_size,
  bool *done,
  index_t *points_finished) {
  static const bool IS_RANGE_NEIGHBORS = ArgsType::IS_RANGE_NEIGHBORS;
  fl::logger->Debug()<<"dual tree trace size = "<<trace->size();
  // We introduce a blank ArgsType to indicate that we need to
  // exit because this phase of search is done
  trace->push_front(ArgsType());

  // Pop the next item to visit in the list.
  ArgsType args = trace->back();
  trace->pop_back();

  while (trace->empty() == false && args.reference_table() != NULL) {
    bool prune_or_not;
    if (mode == NearestNeighborAllKN) {
      prune_or_not =
        args.bound_distance() >
           args.stat()->at(args.query_table()->get_node_id(
             args.query_node())).dist_so_far();
    }
    if (mode == FurthestNeighborAllKN) {
      prune_or_not =
        args.bound_distance() <
          args.stat()->at(args.query_table()->get_node_id(
             args.query_node())).dist_so_far();

    }

    // If it is prunable,
    if (prune_or_not) {

      if (args.info() != NULL) {
        args.info()->num_of_prunes()++;
      }
    }

    // If it is not prunable,
    else {

      // If both the query and the reference are leaves,
      if (args.query_table()->node_is_leaf(args.query_node()) &&
          args.reference_table()->node_is_leaf(args.reference_node())) {

        // Decide whether to call the mono/bichromatic base case.
        if (IsMonochromatic && args.reference_node() == args.query_node()) {
          SingleThreadDualTreeAllKNBaseTraitMono_t::ComputeBaseCase(&args);
        }
        else {
          SingleThreadDualTreeAllKNBaseTraitBi_t::ComputeBaseCase(&args);
        }
      }

      // If one of the nodes is not a leaf node,
      else {

        // If only the query side is a leaf node,
        if (args.query_table()->node_is_leaf(args.query_node())) {

          // Prioritize on the reference side.
          ReferenceTree_t *left =
            args.reference_table()->get_node_left_child(
              args.reference_node());

          ReferenceTree_t *right =
            args.reference_table()->get_node_right_child(
              args.reference_node());

          CalcPrecision_t left_distance;
          CalcPrecision_t right_distance;
          bool go_left;

          if (mode == NearestNeighborAllKN) {
            left_distance =
              args.query_table()->get_node_bound(args.query_node()).
              MinDistanceSq(*args.metric(),
                            args.reference_table()->get_node_bound(left));
            right_distance =
              args.query_table()->get_node_bound(args.query_node()).
              MinDistanceSq(*args.metric(),
                            args.reference_table()->get_node_bound(right));
            go_left = left_distance < right_distance;
          }

          if (mode == FurthestNeighborAllKN) {
            left_distance =
              args.query_table()->get_node_bound(args.query_node()).
              MaxDistanceSq(*args.metric(),
                            args.reference_table()->get_node_bound(left));
            right_distance =
              args.query_table()->get_node_bound(args.query_node()).
              MaxDistanceSq(*args.metric(),
                            args.reference_table()->get_node_bound(right));
            go_left = left_distance > right_distance;
          }

          if (go_left) {
            args.bound_distance() = left_distance;
            args.reference_node() = left;
            if (trace->size()<max_trace_size) {
              trace->push_back(args);
            } else {
              fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
            }
            args.reference_node() = right;
            args.bound_distance() = right_distance;
            if (trace->size()<max_trace_size) {
              trace->push_front(args);
            } else {
              fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
            }
          }
          else {
            args.reference_node() = right;
            args.bound_distance() = right_distance;
            if (trace->size()<max_trace_size) {
              trace->push_back(args);
            } else {
              fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
            }
            args.reference_node() = left;
            args.bound_distance() = left_distance;
            if (trace->size()<max_trace_size) {
              trace->push_front(args);
            } else {
              fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";

            }
          }
        } // end of qnode only being the leaf node,

        // If the query node is not a leaf node,
        else {

          // If the reference node is a leaf,
          if (args.reference_table()->node_is_leaf(args.reference_node())) {

            QueryTree_t *query_node = args.query_node();
            QueryTree_t *left = args.query_table()->get_node_left_child(
                                  args.query_node());
            QueryTree_t *right = args.query_table()->get_node_right_child(
                                   args.query_node());
            CalcPrecision_t left_distance;
            CalcPrecision_t right_distance;

            if (mode == NearestNeighborAllKN) {
              left_distance =
                args.query_table()->get_node_bound(left).
                MinDistanceSq(*args.metric(),
                              args.reference_table()->get_node_bound(
                                args.reference_node()));

              right_distance =
                args.query_table()->get_node_bound(right).
                MinDistanceSq(*args.metric(),
                              args.reference_table()->get_node_bound(
                                args.reference_node()));
            }
            if (mode == FurthestNeighborAllKN) {
              left_distance =
                args.query_table()->get_node_bound(left).
                MaxDistanceSq(*args.metric(),
                              args.reference_table()->get_node_bound(
                                args.reference_node()));

              right_distance =
                args.query_table()->get_node_bound(right).
                MaxDistanceSq(*args.metric(),
                              args.reference_table()->get_node_bound(
                                args.reference_node()));
            }

            args.query_node() = left;
            args.bound_distance() = left_distance;
            if (trace->size()<max_trace_size) {
              trace->push_back(args);
            } else {
              fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
            }

            args.query_node() = right;
            args.bound_distance() = right_distance;
            if (trace->size()<max_trace_size) {
              trace->push_back(args);
            } else {
              fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
            }

            // We need to update the upper bound based on the new
            // upper bounds of the children.
            if (mode == NearestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
                args.stat()->at(args.query_table()->
                get_node_id(query_node)).set_dist_so_far(
                  std::max(
                      args.stat()->at(args.query_table()->
                      get_node_id(left)).dist_so_far(),
                      args.stat()->at(args.query_table()->
                      get_node_id(right)).dist_so_far()));
            }
            if (mode == FurthestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
                args.stat()->at(args.query_table()->get_node_id(
                  query_node)).set_dist_so_far(
                    std::min(
                        args.stat()->at(args.query_table()->
                        get_node_id(left)).dist_so_far(),
                        args.stat()->at(args.query_table()->
                        get_node_id(right)).dist_so_far()));
            }

          } // end of qnode = non-leaf, rnode = leaf,

          else { // If the reference node is also not a leaf node,

            QueryTree_t *query_node = args.query_node();
            QueryTree_t *query_left =
              args.query_table()->get_node_left_child(args.query_node());
            QueryTree_t *query_right =
              args.query_table()->get_node_right_child(args.query_node());
            ReferenceTree_t *reference_left =
              args.reference_table()->get_node_left_child(
                args.reference_node());
            ReferenceTree_t *reference_right =
              args.reference_table()->get_node_right_child(
                args.reference_node());
            CalcPrecision_t left_distance;
            CalcPrecision_t right_distance;
            bool go_left;
            if (mode == NearestNeighborAllKN) {
              left_distance =
                args.query_table()->get_node_bound(query_left).
                MinDistanceSq(
                  *args.metric(),
                  args.reference_table()->get_node_bound(reference_left));

              right_distance =
                args.query_table()->get_node_bound(query_left).
                MinDistanceSq(
                  *args.metric(),
                  args.reference_table()->get_node_bound(reference_right));
              go_left = left_distance < right_distance;
            }
            if (mode == FurthestNeighborAllKN) {
              left_distance =
                args.query_table()->get_node_bound(query_left).
                MaxDistanceSq(
                  *args.metric(),
                  args.reference_table()->get_node_bound(reference_left));

              right_distance =
                args.query_table()->get_node_bound(query_left).
                MaxDistanceSq(
                  *args.metric(),
                  args.reference_table()->get_node_bound(reference_right));
              go_left = left_distance > right_distance;
            }

            if (go_left) {
              args.query_node() = query_left;
              args.reference_node() = reference_left;
              args.bound_distance() = left_distance;
              if (trace->size()<max_trace_size) {
                trace->push_back(args);
              } else {
                fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
              }

              args.query_node() = query_left;
              args.reference_node() = reference_right;
              args.bound_distance() = right_distance;
              if (trace->size()<max_trace_size) {
                trace->push_front(args);
              } else {
                fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
              }
            }
            else {
              args.query_node() = query_left;
              args.reference_node() = reference_right;
              args.bound_distance() = right_distance;
              if (trace->size()<max_trace_size) {
                trace->push_back(args);
              } else {
                fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
              }
              args.query_node() = query_left;
              args.reference_node() = reference_left;
              args.bound_distance() = left_distance;
              if (trace->size()<max_trace_size) {
                trace->push_front(args);
              } else {
                fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
              }
            }
            if (mode == NearestNeighborAllKN) {
              left_distance =
                args.query_table()->get_node_bound(query_right).
                MinDistanceSq(*args.metric(),
                              args.reference_table()->
                              get_node_bound(reference_left));
              right_distance =
                args.query_table()->get_node_bound(query_right).
                MinDistanceSq(*args.metric(),
                              args.reference_table()->get_node_bound(
                                reference_right));
              go_left = left_distance < right_distance;
            }

            if (mode == FurthestNeighborAllKN) {
              left_distance =
                args.query_table()->get_node_bound(query_right).
                MaxDistanceSq(*args.metric(),
                              args.reference_table()->
                              get_node_bound(reference_left));
              right_distance =
                args.query_table()->get_node_bound(query_right).
                MaxDistanceSq(*args.metric(),
                              args.reference_table()->get_node_bound(
                                reference_right));
              go_left = left_distance > right_distance;
            }

            if (go_left) {
              args.query_node() = query_right;
              args.reference_node() = reference_left;
              args.bound_distance() = left_distance;
              if (trace->size()<max_trace_size) {
                trace->push_back(args);
              } else {
                fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
              }
              args.query_node() = query_right;
              args.reference_node() = reference_right;
              args.bound_distance() = right_distance;
              if (trace->size()<max_trace_size) {
                trace->push_front(args);
              } else {
                fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
              }
            }
            else {
              args.query_node() = query_right;
              args.reference_node() = reference_right;
              args.bound_distance() = right_distance;
              if (trace->size()<max_trace_size) {
                trace->push_back(args);
              } else {
                fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
              }
              args.query_node() = query_right;
              args.reference_node() = reference_left;
              args.bound_distance() = left_distance;
              if (trace->size()<max_trace_size) {
                trace->push_front(args);
              } else {
                fl::logger->Warning()<<"stack iterator reached maximum size ("
                <<max_trace_size<<")";
              }
            }

            // Update the bound as above
            if (mode == NearestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
                args.stat()->at(args.query_table()->
                get_node_id(query_node)).set_dist_so_far(
                  std::max(
                      args.stat()->at(args.query_table()->
                      get_node_id(query_left)).dist_so_far(),
                      args.stat()->at(args.query_table()->
                      get_node_id(query_right)).dist_so_far()));
            }
            if (mode == FurthestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
                args.stat()->at(args.query_table()->get_node_id(
                  query_node)).set_dist_so_far(
                    std::min(
                        args.stat()->at(args.query_table()->
                        get_node_id(query_left)).dist_so_far(),
                        args.stat()->at(args.query_table()->
                        get_node_id(query_right)).dist_so_far()));
            }

          } // end of qnode = non-leaf, rnode = non-leaf
        }
      } // end of not ( qnode = leaf, rnode = leaf )
    } // end of non-prunable case.

    // Pop the next item in the list.
    args = trace->back();
    trace->pop_back();

  } // end of the iterator loop.
}

template<typename TypeVector>
template<typename ArgsType>
void SingleThreadAllKNTraversal<TypeVector>::
QueryBinaryReferenceBinary::InitSingleNeighborsProgressive(
  std::vector<ArgsType> *trace) {
  static const bool IS_RANGE_NEIGHBORS = ArgsType::IS_RANGE_NEIGHBORS;
  ArgsType args;
  if (mode == NearestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
    args.dist_so_far() = std::numeric_limits<CalcPrecision_t>::max();
  }
  if (mode == FurthestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
    args.dist_so_far() = -std::numeric_limits<CalcPrecision_t>::max();
  }
  args.reference_node() = args.reference_table()->get_tree();
  index_t num_of_entries = args.query_table()->n_entries();
  const QueryTable_t *table = args.query_table();
  for (index_t i = 0; i <= num_of_entries; i++) {
    args.reference_node() = args.reference_table()->get_tree();
    args.query_point() = new QueryPoint_t();
    table->get(i, args.query_point());
    args.query_point_id() = i;
  }
}

template<typename TypeVector>
template<typename ArgsType>
void SingleThreadAllKNTraversal<TypeVector>::
QueryBinaryReferenceBinary::ComputeAllSingleNeighborsProgressive(
  std::vector<std::vector<ArgsType> > *trace,
  bool *done,
  index_t *points_finished) {

  for (unsigned int i = 0; i < trace->size(); i++) {
    ComputeSingleNeighborsProgressive(&(trace->at(i)));
    if (trace->empty() == true) {
      ++(*points_finished);
      if (*points_finished == (index_t) trace->size()) {
        *done = true;
        return;
      }
    }
  }
}

template<typename TypeVector>
template<typename ArgsType>
void SingleThreadAllKNTraversal<TypeVector>::
QueryBinaryReferenceBinary::ComputeSingleNeighborsProgressive(
  std::vector<ArgsType> *trace) {
  static const bool IS_RANGE_NEIGHBORS = ArgsType::IS_RANGE_NEIGHBORS;
  // Make sure the bounding information is correct
  DEBUG_ASSERT_MSG(trace->empty() == false, "Empty function trace");
  ArgsType args = trace->back();
  trace->pop_back();
  DEBUG_ASSERT_MSG(args.reference_node() != NULL, "reference node is null");
  while (args.reference_table()->node_is_leaf(args.reference_node()) == false) {
    // We'll order the computation by distance
    ReferenceTree_t *left_node = args.reference_table()->get_node_left_child(
                                   args.reference_node());
    ReferenceTree_t *right_node = args.reference_table()->get_node_right_child(
                                    args.reference_node());
    CalcPrecision_t left_distance = 0;
    CalcPrecision_t right_distance = 0;
    bool go_left;
    if (IS_RANGE_NEIGHBORS) {
      go_left = true;
    }
    if (!IS_RANGE_NEIGHBORS) {
      if (mode == NearestNeighborAllKN) {
        left_distance =
          args.reference_table()->get_node_bound(left_node).MinDistanceSq(
            *args.metric(),
            *args.query_point());
        right_distance = args.reference_table()->get_node_bound(
                           right_node).MinDistanceSq(*args.metric(),
                                                     *args.query_point());
        go_left = left_distance < right_distance;
      }
      if (mode == FurthestNeighborAllKN) {
        left_distance =
          args.reference_table()->get_node_bound(left_node).MaxDistanceSq(
            *args.metric(),
            *args.query_point());
        right_distance = args.reference_table()->get_node_bound(
                           right_node).MaxDistanceSq(*args.metric(),
                                                     *args.query_point());
        go_left = left_distance > right_distance;
      }
    }
    if (go_left) {
      args.reference_node() = left_node;
      //ComputeSingleNeighborsRecursion(args);
      bool prune_or_not;
      if (mode == NearestNeighborAllKN) {
        prune_or_not = (args.dist_so_far() < right_distance);
      }
      if (mode == FurthestNeighborAllKN) {
        prune_or_not = (args.dist_so_far() > right_distance);
      }

      if (prune_or_not) {
        args.info()->num_of_prunes()++;
        if (trace->empty()) {
          delete args.query_point();
          return;
        }
        args = trace->back();
        trace->pop_back();
      }
      else {
        //  ComputeSingleNeighborsRecursion(args);
        trace->push_back(args);
        trace->back().reference_node() = right_node;
      }
    }
    else {
      args.reference_node() = right_node;
      //  ComputeSingleNeighborsRecursion(args);
      bool prune_or_not;
      if (mode == NearestNeighborAllKN) {
        prune_or_not = args.dist_so_far() < left_distance;
      }
      if (mode == FurthestNeighborAllKN) {
        prune_or_not = args.dist_so_far() > left_distance;
      }

      if (prune_or_not) {
        args.info()->num_of_prunes()++;
        if (trace->empty()) {
          delete args.query_point();
          return;
        }
        args = trace->back();
        trace->pop_back();
      }
      else {
        //ComputeSingleNeighborsRecursion(args);
        trace->push_back(args);
        trace->back().reference_node() = left_node;
      }
    }
  }
  // node->is_leaf() works as one would expect
  // Base Case
  index_t nearest_size;
  if (Is1N == true) {
    nearest_size = 1;
  }
  else {
    nearest_size = static_cast<index_t>(args.kns());
  }
  std::vector<std::pair<CalcPrecision_t, index_t> > neighbors;
  index_t ind;
  typename ReferenceTable_t::TreeIterator reference_it(
    *args.reference_table(),
    args.reference_node());
  ReferencePoint_t reference_point;
  index_t reference_index;

  if (!IS_RANGE_NEIGHBORS) {
    if (Is1N == false) {
      neighbors.resize(nearest_size + reference_it.count());
      ind = args.query_point_id() * static_cast<index_t>(args.kns());
    }
    else {
      ind = args.query_point_id();
    }
    if (Is1N == false && !IS_RANGE_NEIGHBORS) {
      for (index_t i = 0; i < static_cast<index_t>(args.kns()); i++) {
        boost::mpl::eval_if <
        boost::mpl::bool_<IS_RANGE_NEIGHBORS>,
        ALLKNMPL::UpdateOperator2,
        ALLKNMPL::UpdateOperator1
        >::type::InitTempNeighbors(
          &neighbors,
          i,
          ind,
          *(args.neighbor_distances()),
          *(args.neighbor_indices()));
      }
    }
  }
  // #pragma omp parallel for  shared(args)
  for (index_t i = 0; i < reference_it.count(); ++i) {
    reference_it.get(i, &reference_point);
    reference_it.get_id(i, &reference_index);
    // Confirm that points do not identify themselves as neighbors
    // in the monochromatic case
    if (!(IsMonochromatic &&
          unlikely(reference_index == args.query_point_id()))) {
      // Do not change the order of arguments, because it will fail in assymetric divergences
      CalcPrecision_t distance =
        args.metric()->DistanceSq(
          reference_point,
          *(args.query_point()));
      if (Is1N == false) {
        // we'll update the candidate
        bool push_back_or_not;
        if (mode == NearestNeighborAllKN) {
          push_back_or_not =
            distance < args.dist_so_far();
        }
        if (mode == FurthestNeighborAllKN) {
          push_back_or_not =
            distance > args.dist_so_far();
        }

        if (push_back_or_not) {
          neighbors[nearest_size+i].first = distance;
          neighbors[nearest_size+i].second = reference_index;
        }
      }
      else {
// #pragma omp critical (update_one_nn_progressive)
        {
          // we'll update the candidate
          bool push_back_or_not;
          if (mode == NearestNeighborAllKN) {
            push_back_or_not =
              distance < args.dist_so_far();
          }
          if (mode == FurthestNeighborAllKN) {
            push_back_or_not =
              distance > args.dist_so_far();
          }
          if (push_back_or_not) {
            index_t neighbor_id;
            if (IS_RANGE_NEIGHBORS) {
              neighbor_id = args.query_point_id();
            }
            else {
              neighbor_id = ind;
            }
            boost::mpl::eval_if <
            boost::mpl::bool_<IS_RANGE_NEIGHBORS>,
            ALLKNMPL::UpdateOperator2,
            ALLKNMPL::UpdateOperator1
            >::type::Insert(args.neighbor_indices(),
                            args.neighbor_distances(),
                            neighbor_id,
                            reference_index,
                            0,
                            distance);
            if (!IS_RANGE_NEIGHBORS) {
              args.dist_so_far() = (*(args.neighbor_distances()))[ind];
            }
          }
        } // omp critical
      }
    }
  } // for reference_index
  if (Is1N == false && !IS_RANGE_NEIGHBORS) {
    if (mode == NearestNeighborAllKN) {
      std::sort(neighbors.begin(), neighbors.end());
      /* std::nth_element(neighbors.begin(),
                       neighbors.begin()+static_cast<int>(args.kns())-1,
                       neighbors.end());
       */
    }
    if (mode == FurthestNeighborAllKN) {
      /* std::nth_element(neighbors.begin(),
                        neighbors.begin()+static_cast<int>(args.kns())-1,
                        neighbors.end(),
                        std::greater<std::pair<CalcPrecision_t, index_t> >());
       */
      std::sort(neighbors.begin(), neighbors.end(),
                  std::greater<std::pair<CalcPrecision_t, index_t> >());
    }
    for (index_t i = 0; i < static_cast<index_t>(args.kns()); i++) {
      index_t neighbor_id;
      neighbor_id = ind;
      boost::mpl::eval_if <
      boost::mpl::bool_<IS_RANGE_NEIGHBORS>,
      ALLKNMPL::UpdateOperator2,
      ALLKNMPL::UpdateOperator1
      >::type::Insert(args.neighbor_indices(),
                      args.neighbor_distances(),
                      neighbor_id,
                      neighbors[i].second,
                      i,
                      neighbors[i].first);
    }
    args.dist_so_far() = (*(args.neighbor_distances()))
                         [ind+static_cast<index_t>(args.kns())-1];
  }
  else {
  }
  if (trace->empty() == true) {
    delete args.query_point();
  }
}

template < typename AllKNArgs,
bool IsMonochromatic,
bool Is1N,
AllKNMode mode >
template<typename ArgsType>
void SingleThreadDualTreeAllKNBaseTrait<AllKNArgs, IsMonochromatic, Is1N, mode>::
ComputeBaseCase(ArgsType *args) {
  static const bool IS_RANGE_NEIGHBORS = ArgsType::IS_RANGE_NEIGHBORS;
  // Check that the pointers are not NULL
  DEBUG_ASSERT(args->query_node() != NULL);
  DEBUG_ASSERT(args->reference_node() != NULL);

  // Used to find the query node's new upper bound
  CalcPrecision_t query_bound_neighbor_distance;
  if (mode == NearestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
    query_bound_neighbor_distance = -std::numeric_limits<CalcPrecision_t>::max();
  }
  if (mode == FurthestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
    query_bound_neighbor_distance = std::numeric_limits<CalcPrecision_t>::max();
  }
  if (IS_RANGE_NEIGHBORS) {
    query_bound_neighbor_distance = static_cast<CalcPrecision_t>(args->kns());
  }

  index_t neighbors_size = static_cast<index_t>(args->kns());

  // node->begin() is the index of the first point in the node,
  // node->end is one past the last index

  typename QueryTable_t::TreeIterator query_it(*(args->query_table()),
      args->query_node());
  typename ReferenceTable_t::TreeIterator reference_it(
    *(args->reference_table()),
    args->reference_node());
  query_it.Reset();


  std::vector<std::pair<CalcPrecision_t, index_t> > neighbors;
  if (!IS_RANGE_NEIGHBORS) {
    neighbors.resize(neighbors_size);
  }
 
  ReferencePoint_t reference_point;
  QueryPoint_t query_point; 
  for (index_t k = 0; k < query_it.count(); ++k) {
    index_t  query_id;
    index_t  ref_id;
    // Get the query point from the matrix
    query_it.get(k, &query_point);
    query_it.get_id(k, &query_id);

    if (!Is1N && !IS_RANGE_NEIGHBORS) {
      index_t ind = static_cast<index_t>(query_id * args->kns());
      //index_t ind = 0;
      for (index_t i = 0; i < static_cast<index_t>(args->kns()); i++) {
        boost::mpl::eval_if <
        boost::mpl::bool_<IS_RANGE_NEIGHBORS>,
        ALLKNMPL::UpdateOperator2,
        ALLKNMPL::UpdateOperator1
        >::type::InitTempNeighbors(
          &neighbors,
          i,
          ind,
          *args->neighbor_distances(),
          *args->neighbor_indices());
      }

      CalcPrecision_t query_to_node_distance;
      if (mode == NearestNeighborAllKN) {
        query_to_node_distance = args->reference_table()->get_node_bound(args->reference_node()).
                                 MinDistanceSq(*args->metric(), query_point);
      }
      if (mode == FurthestNeighborAllKN) {
        query_to_node_distance = args->reference_table()->get_node_bound(args->reference_node()).
                                 MaxDistanceSq(*args->metric(), query_point);
      }
      bool search_or_not;
      if (mode == NearestNeighborAllKN) {
        search_or_not =
          query_to_node_distance <
          (*(args->neighbor_distances()))[ind
                                          +static_cast<index_t>(args->kns())-1];
      }
      if (mode == FurthestNeighborAllKN) {
        search_or_not =
          query_to_node_distance >
          (*(args->neighbor_distances()))[ind
                                          +static_cast<index_t>(args->kns())-1];
      }
      if (search_or_not == true) {
        // We'll do the same for the references
        for (int j = 0; j < reference_it.count(); ++j) {
          // Confirm that points do not identify themselves as neighbors
          // in the monochromatic case

          reference_it.get(j, &reference_point);
          reference_it.get_id(j, &ref_id);
          if (!IsMonochromatic ||  query_id != ref_id) {
            CalcPrecision_t distance = args->metric()->DistanceSq(
                                         query_point,
                                         reference_point);
            // If the reference point is closer than the current candidate,
            // we'll update the candidate
            bool push_back_or_not;
            if (mode == NearestNeighborAllKN) {
              push_back_or_not =
                distance < (*(args->neighbor_distances()))[ind+
                    static_cast<index_t>(args->kns())-1];
            }
            if (mode == FurthestNeighborAllKN) {
              push_back_or_not =
                distance > (*(args->neighbor_distances()))[ind+
                    static_cast<index_t>(args->kns())-1];
            }
            if (push_back_or_not) {
              neighbors.push_back(std::make_pair(distance, ref_id));
            }
          }
        }// for reference_index

        if (mode == NearestNeighborAllKN) {
          std::sort(neighbors.begin(), neighbors.end());
        }

        if (mode == FurthestNeighborAllKN) {
          std::sort(neighbors.begin(), neighbors.end(),
                      std::greater<std::pair<CalcPrecision_t, index_t> >());
        }
        for (index_t i = 0; i < static_cast<index_t>(args->kns()); i++) {
          index_t neighbor_id;
          neighbor_id = ind;
          boost::mpl::eval_if <
          boost::mpl::bool_<IS_RANGE_NEIGHBORS>,
          ALLKNMPL::UpdateOperator2,
          ALLKNMPL::UpdateOperator1
          >::type::Insert(args->neighbor_indices(),
                          args->neighbor_distances(),
                          neighbor_id,
                          neighbors[i].second,
                          i,
                          neighbors[i].first);
        }
        neighbors.resize(static_cast<index_t>(args->kns()));
      }
      // We need to find the upper bound distance for this query node
      bool update_upper_bound_or_not;
      if (mode == NearestNeighborAllKN) {
        update_upper_bound_or_not =
          (*(args->neighbor_distances()))[ind
                                          +static_cast<index_t>(args->kns())-1] >
          query_bound_neighbor_distance;
      }
      if (mode == FurthestNeighborAllKN) {
        update_upper_bound_or_not =
          (*(args->neighbor_distances()))[ind
                                          +static_cast<index_t>(args->kns())-1] <
          query_bound_neighbor_distance;
      }

      if (update_upper_bound_or_not) {
        query_bound_neighbor_distance = (*(args->neighbor_distances()))
                                         [ind+static_cast<index_t>(args->kns())-1];
      }

    }
    else {
      index_t ind = query_id;
      CalcPrecision_t query_to_node_distance ;
      if (mode == NearestNeighborAllKN) {
        query_to_node_distance =
          args->reference_table()->get_node_bound(args->reference_node()).
          MinDistanceSq(*args->metric(), query_point);
      }
      if (mode == FurthestNeighborAllKN) {
        query_to_node_distance =
          args->reference_table()->get_node_bound(args->reference_node()).
          MaxDistanceSq(*args->metric(), query_point);
      }
      bool search_or_not;
      if (mode == NearestNeighborAllKN) {
        if (!IS_RANGE_NEIGHBORS) {
          search_or_not =
            query_to_node_distance < (*(args->neighbor_distances()))[ind];
        }
        if (IS_RANGE_NEIGHBORS) {
          search_or_not =
            query_to_node_distance < static_cast<CalcPrecision_t>(args->kns());
        }
      }
      if (mode == FurthestNeighborAllKN) {
        if (!IS_RANGE_NEIGHBORS) {
          search_or_not =
            query_to_node_distance > (*(args->neighbor_distances()))[ind];
        }
        if (IS_RANGE_NEIGHBORS) {
          search_or_not =
            query_to_node_distance > static_cast<CalcPrecision_t>(args->kns());
        }
      }

      if (search_or_not) {
        // We'll do the same for the references
        reference_it.Reset();
        for (index_t j = 0; j < reference_it.count(); ++j) {
          //if(query
          // Confirm that points do not identify themselves as neighbors
          // in the monochromatic case
          reference_it.get(j, &reference_point);
          reference_it.get_id(j, &ref_id);
          if (!IsMonochromatic ||  query_id != ref_id) {
            // Do not change the order of arguments, it will fail for non symetric divergences
            CalcPrecision_t distance = args->metric()->DistanceSq(
                                         reference_point,
                                         query_point);
            // If the reference point is closer than the current candidate,
            // we'll update the candidate
            bool push_or_not;
            if (mode == NearestNeighborAllKN) {
              if (!IS_RANGE_NEIGHBORS) {
                push_or_not = distance < (*(args->neighbor_distances()))[ind];
              }
              if (IS_RANGE_NEIGHBORS) {
                push_or_not =
                  distance < static_cast<CalcPrecision_t>(args->kns());
              }
            }
            if (mode == FurthestNeighborAllKN) {
              if (!IS_RANGE_NEIGHBORS) {
                push_or_not = distance > (*(args->neighbor_distances()))[ind];
              }
              if (IS_RANGE_NEIGHBORS) {
                push_or_not =
                  distance > static_cast<CalcPrecision_t>(args->kns());
              }
            }
            if (push_or_not) {
              index_t neighbor_id = ind;
              boost::mpl::eval_if <
              boost::mpl::bool_<IS_RANGE_NEIGHBORS>,
              ALLKNMPL::UpdateOperator2,
              ALLKNMPL::UpdateOperator1
              >::type::Insert(args->neighbor_indices(),
                              args->neighbor_distances(),
                              neighbor_id,
                              ref_id,
                              0,
                              distance);
            }
          }
        } // for reference_index
      }
      // We need to find the upper bound distance for this query node
      bool update_bound_or_not;
      if (mode == NearestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
        update_bound_or_not =
          (*(args->neighbor_distances()))[ind] > query_bound_neighbor_distance;
      }
      if (mode == FurthestNeighborAllKN && !IS_RANGE_NEIGHBORS) {
        update_bound_or_not =
          (*(args->neighbor_distances()))[ind] < query_bound_neighbor_distance;
      }
      if (IS_RANGE_NEIGHBORS) {
        update_bound_or_not = false;
      }
      if (update_bound_or_not) {
        query_bound_neighbor_distance = (*(args->neighbor_distances()))[ind];
      }
    }
  } // for query_index

  if (!IS_RANGE_NEIGHBORS) {
      args->stat()->at(args->query_table()->get_node_id(args->query_node())).
    set_dist_so_far(query_bound_neighbor_distance);
  }
}
}
}

#endif
