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
#ifndef FL_LITE_MLPACK_ALLKN_ALLKN_DEV_H_
#define FL_LITE_MLPACK_ALLKN_ALLKN_DEV_H_

#include "mlpack/allkn/allkn_computations_dev.h"
#include "mlpack/allkn/allkn.h"
#include "mlpack/allkn/allkn_defs.h"

namespace fl {
namespace ml {
/**
 *  Function definitions
 */
template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::TreeArguments() :
    query_point_(NULL),
    query_table_(NULL),
    query_point_id_(-1),
    reference_node_(NULL),
    reference_table_(NULL),
    neighbor_distances_(NULL),
    neighbor_indices_(NULL),
    stat_(NULL),
    kns_(0),
    bound_distance_(new CalcPrecision_t()),
    dist_so_far_(new CalcPrecision_t()),
    metric_(NULL),
    info_(NULL) {
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::TreeArguments(QueryTree_t* query_node1,
    QueryTable_t* query_table1,
    ReferenceTree_t* reference_node1,
    ReferenceTable_t* reference_table1,
    ContainerDistType* neighbor_distances1,
    ContainerIndType*  neighbor_indices1,
    std::vector<NeighborStatistic<CalcPrecision_t> > *stat1,
    const NeighborMethodType knns1,
    const CalcPrecision_t lower_bound_distance1,
    const MetricType *metric,
    DynamicArguments_t *info)  :
    query_node_(query_node1),
    query_table_(query_table1),
    query_point_id_(-1),
    reference_node_(reference_node1),
    reference_table_(reference_table1),
    neighbor_distances_(neighbor_distances1),
    neighbor_indices_(neighbor_indices1),
    stat_(stat1),
    kns_(knns1),
    bound_distance_(new CalcPrecision_t()),
    metric_(metric),
    info_(info)  {
  *bound_distance_ = lower_bound_distance1;
  dist_so_far_ = bound_distance_;
};

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::TreeArguments(QueryPoint_t &query_point1,
    index_t query_point_id1,
    ReferenceTree_t* reference_node1,
    const ReferenceTable_t* reference_table1,
    ContainerDistType* neighbor_distances1,
    ContainerIndType*  neighbor_indices1,
    const NeighborMethodType knns1,
    const CalcPrecision_t dist_so_far,
    const MetricType *metric,
    DynamicArguments_t *info) :
    query_point_(query_point1),
    query_point_id_(query_point_id1),
    query_table_(NULL),
    reference_node_(reference_node1),
    reference_table_(reference_table1),
    neighbor_distances_(neighbor_distances1),
    neighbor_indices_(neighbor_indices1),
    kns_(knns1),
    dist_so_far_(new CalcPrecision_t()),
    metric_(metric),
    info_(info) {

  *dist_so_far_ = dist_so_far;
  bound_distance_ = dist_so_far_;
};

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::TreeArguments(const TreeArguments &other) :
    query_point_(other.query_point_),
    query_table_(other.query_table_),
    query_point_id_(other.query_point_id_),
    reference_node_(other.reference_node_),
    reference_table_(other.reference_table_),
    neighbor_distances_(other.neighbor_distances_),
    neighbor_indices_(other.neighbor_indices_),
    stat_(other.stat_),
    kns_(other.kns_),
    metric_(other.metric_),
    info_(other.info_) {
  bound_distance_.reset(new CalcPrecision_t());
  *bound_distance_ = *(other.bound_distance_);
  dist_so_far_ = bound_distance_;
};


template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename AllKN<ArgMap>::template TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>
&AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::operator=(
  const TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType> &other) {
  query_point_ = other.query_point_;
  query_point_id_ = other.query_point_id_;
  query_table_ = other.query_table_;
  reference_node_ = other.reference_node_;
  reference_table_ = other.reference_table_;
  neighbor_distances_ = other.neighbor_distances_;
  neighbor_indices_ = other.neighbor_indices_;
  stat_=other.stat_;
  kns_ = other.kns_;
  metric_ = other.metric_;
  info_ = other.info_;
  if (other.bound_distance_.get() != NULL) {
    DEBUG_ASSERT(bound_distance_.get() != NULL);
    *bound_distance_ = *(other.bound_distance_);
  }
  if (other.dist_so_far_.get() != NULL) {
    DEBUG_ASSERT(dist_so_far_.get() != NULL);
    *dist_so_far_ = *(other.dist_so_far_);
  }
  return *this;
}
template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
void AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::CopyNonSharedOnly(
  const TreeArguments &other) {
  query_point_ = other.query_point_;
  query_point_id_ = other.query_point_id_;
  query_table_ = other.query_table_;
  reference_node_ = other.reference_node_;
  reference_table_ = other.reference_table_;
  neighbor_distances_ = other.neighbor_distances_;
  neighbor_indices_ = other.neighbor_indices_;
  stat_=other.stat_;
  kns_ = other.kns_;
  metric_ = other.metric_;
  info_ = other.info_;
  // min_dist_so_far
  // lower_bound_distance
  // are all shared in the multithread computation
  dist_so_far_ = other.dist_so_far_;
  bound_distance_ = other.bound_distance_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename AllKN<ArgMap>::QueryTree_t* &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::query_node() {
  return query_node_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename AllKN<ArgMap>::QueryPoint_t* &AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::query_point() {
  return query_point_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
const typename AllKN<ArgMap>::QueryTable_t* &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::query_table() {
  return query_table_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
index_t &AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::query_point_id() {
  return query_point_id_;

}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename AllKN<ArgMap>::ReferenceTree_t* &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::reference_node() {
  return reference_node_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
const typename AllKN<ArgMap>::ReferenceTable_t* &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::reference_table() {
  return reference_table_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
ContainerDistType* &
AllKN<ArgMap>::TreeArguments < MetricType, NeighborMethodType,
ContainerDistType, ContainerIndType >::neighbor_distances() {
  return neighbor_distances_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
ContainerIndType* &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::neighbor_indices() {
  return neighbor_indices_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
std::vector<NeighborStatistic<typename AllKN<ArgMap>::CalcPrecision_t> >* &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::stat() {
  return stat_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
NeighborMethodType &AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::kns() {
  return kns_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename AllKN<ArgMap>::CalcPrecision_t &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::lower_bound_distance() {
  return *bound_distance_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename AllKN<ArgMap>::CalcPrecision_t &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::bound_distance() {
  return *bound_distance_;
}


template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename AllKN<ArgMap>::CalcPrecision_t &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::min_dist_so_far() {
  return *dist_so_far_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename AllKN<ArgMap>::CalcPrecision_t &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::dist_so_far() {
  return *dist_so_far_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
const MetricType* &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::metric() {
  return metric_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename AllKN<ArgMap>::DynamicArguments_t* &
AllKN<ArgMap>::TreeArguments<MetricType, NeighborMethodType, ContainerDistType, ContainerIndType>::info() {
  return info_;
}

/**
 * definitions of the iterator class
 */
template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
AllKN<ArgMap>::iterator < MetricType, NeighborMethodType,
ContainerDistType, ContainerIndType >::iterator() :
    allkn_(NULL),
    single_tree_stacks_(NULL),
    is_dual_tree_(false),
    kns_(-1),
    done_(false),
    dist_(NULL),
    ind_(NULL),
    metric_(NULL),
    stage_(0) {
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
AllKN<ArgMap>::iterator < MetricType, NeighborMethodType, ContainerDistType, ContainerIndType >
::iterator(const iterator &other) {

  allkn_ = other.allkn_;
  single_tree_stacks_ = other.single_tree_stacks_;
  is_dual_tree_ = other.is_dual_tree_;
  kns_ = other.kns_;
  done_ = other.done_;
  dist_ = other.dist_;
  ind_ = other.ind_;
  metric_ = other.metric_;
  stage_ = other.stage_;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename AllKN<ArgMap>::template iterator < MetricType, NeighborMethodType, ContainerDistType, ContainerIndType > &
AllKN<ArgMap>::iterator < MetricType, NeighborMethodType,
ContainerDistType, ContainerIndType >::operator=(const iterator &other) {

  allkn_ = other.allkn_;
  max_trace_size_=other.max_trace_size_;
  single_tree_stacks_ = other.single_tree_stacks_;
  is_dual_tree_ = other.is_dual_tree_;
  kns_ = other.kns_;
  done_ = other.done_;
  dist_ = other.dist_;
  ind_ = other.ind_;
  metric_ = other.metric_;
  stage_ = other.stage_;
  return *this;
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
AllKN<ArgMap>::iterator < MetricType, NeighborMethodType,
ContainerDistType, ContainerIndType >::~iterator() {
  if (is_dual_tree_) {
    delete dual_tree_stack_;
  }
  else {
    delete single_tree_stacks_;
  }
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
bool AllKN<ArgMap>::iterator < MetricType, NeighborMethodType,
ContainerDistType, ContainerIndType >::operator==(const iterator &other) const {
  return (done_ == other.done_ && stage_ == other.stage_ && allkn_ == other.allkn_);
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
bool AllKN<ArgMap>::iterator < MetricType, NeighborMethodType,
ContainerDistType, ContainerIndType >::operator!=(const iterator &other) const {
  return !operator==(other);
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
void AllKN<ArgMap>::iterator < MetricType, NeighborMethodType,
ContainerDistType, ContainerIndType >::Init(
  AllKN<ArgMap> *allkn,
  index_t max_trace_size,
  const std::string &algorithm,
  const MetricType &metric,
  NeighborMethodType kns,
  ContainerDistType *dist,
  ContainerIndType *ind) {
  allkn_ = allkn;
  max_trace_size_=max_trace_size;
  if (algorithm == "dual") {
    is_dual_tree_ = true;
  }
  else {
    if (algorithm == "single") {
      is_dual_tree_ = false;
    }
    else {
      fl::logger->Die() << "Wrong algorithm argument";
    }
  }
  kns_ = kns;
  dist_ = dist;
  ind_ = ind;
  metric_ = &metric;
  stage_ = 0;
  static const bool IS_RANGE_NEIGHBORS =
    !boost::is_integral<NeighborMethodType>::value;

  boost::mpl::eval_if <
  IsSTLPairTrait<typename ContainerIndType::value_type>,
  InitContainerWithPairs,
  InitContainerWithoutPairs
  >::type::Init(ind, kns * allkn_->query_table()->n_entries());

  boost::mpl::eval_if <
  IsSTLPairTrait<typename ContainerIndType::value_type>,
  InitContainerWithPairs,
  InitContainerWithoutPairs
  >::type::Init(dist, kns * allkn_->query_table()->n_entries());

  CalcPrecision_t dist_so_far_init_value;
  if (KNmode == NearestNeighborAllKN) {
    std::fill(dist_->begin(),
              dist_->end(),
              std::numeric_limits<CalcPrecision_t>::max());
    if (IS_RANGE_NEIGHBORS == true) {
      dist_so_far_init_value = static_cast<CalcPrecision_t>(kns);
    }
    else {
      dist_so_far_init_value = std::numeric_limits<CalcPrecision_t>::max();
    }
    allkn_->ResetStatistics(dist_so_far_init_value);
  }

  if (KNmode == FurthestNeighborAllKN) {
    std::fill(dist_->begin(),
              dist_->end(),
              -std::numeric_limits<CalcPrecision_t>::max());
    if (IS_RANGE_NEIGHBORS == true) {
      dist_so_far_init_value = static_cast<CalcPrecision_t>(kns);
    }
    else {
      dist_so_far_init_value = -std::numeric_limits<CalcPrecision_t>::max();
    }
    allkn_->ResetStatistics(dist_so_far_init_value);
  }

  DynamicArguments_t *info = NULL;
  Args_t args(allkn_->query_table_->get_tree(),
              allkn_->query_table_,
              allkn_->reference_table_->get_tree(),
              allkn->reference_table_,
              dist,
              ind,
              &(allkn->stat_),
              kns_,
              dist_so_far_init_value,
              metric_,
              info);

  if (!is_dual_tree_) {
    single_tree_stacks_ = new std::vector<std::vector<Args_t> >();
    index_t num_of_entries = args.query_table()->n_entries();
    single_tree_stacks_->resize(num_of_entries);
    const QueryTable_t *table = args.query_table();
    for (index_t i = 0; i < num_of_entries; i++) {
      // don't worry about this allocation
      // once the neighbor is found it will be freed
      args.query_point() = new QueryPoint_t();
      table->get(i, args.query_point());
      args.query_point_id() = i;
      single_tree_stacks_->at(i).push_back(args);
    }
  }
  else {
    dual_tree_stack_ = new std::deque<Args_t>();
    dual_tree_stack_->push_back(args);
  }
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename ContainerDistType::value_type AllKN<ArgMap>::iterator < MetricType, NeighborMethodType,
ContainerDistType, ContainerIndType >::operator*() {
  if (is_dual_tree_) {
    return dual_tree_stack_->size();
  }
  else {
    return single_tree_stacks_->size();
  }
}

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
typename AllKN<ArgMap>::template iterator < MetricType, NeighborMethodType,
ContainerDistType, ContainerIndType > &
AllKN<ArgMap>::iterator < MetricType, NeighborMethodType,
ContainerDistType, ContainerIndType >::operator++() {
  static const bool IS_RANGE_NEIGHBORS =
    !boost::is_integral<NeighborMethodType>::value;
  if (is_dual_tree_) {
    if (allkn_->monochromatic_flag_ == true) {
      if (IS_RANGE_NEIGHBORS == true ||  kns_ == 1) {
        AllKNTraversal:: template type <
        boost::mpl::vector6 <
        ArgMap,
        boost::mpl::bool_<QueryTree_t::IsBinary>,
        boost::mpl::bool_<ReferenceTree_t::IsBinary>,
        boost::mpl::bool_<true>,
        boost::mpl::bool_<true>,
        boost::mpl::int_<KNmode>
        >
        >::ComputeDualNeighborsProgressive(dual_tree_stack_, max_trace_size_, &done_, &points_finished_);
      }
      else {
        AllKNTraversal:: template type <
        boost::mpl::vector6 <
        ArgMap, boost::mpl::bool_<QueryTree_t::IsBinary>,
        boost::mpl::bool_<ReferenceTree_t::IsBinary>,
        boost::mpl::bool_<false>, boost::mpl::bool_<true>,
        boost::mpl::int_<KNmode>
        >
        >::
        ComputeDualNeighborsProgressive(dual_tree_stack_, max_trace_size_, &done_, &points_finished_);
      }
    }
    else {
      if (IS_RANGE_NEIGHBORS || kns_ == 1) {
        AllKNTraversal:: template type <
        boost::mpl::vector6 <
        ArgMap,
        boost::mpl::bool_<QueryTree_t::IsBinary>,
        boost::mpl::bool_<ReferenceTree_t::IsBinary>,
        boost::mpl::bool_<true>, boost::mpl::bool_<false>,
        boost::mpl::int_<KNmode>
        >
        >::ComputeDualNeighborsProgressive(dual_tree_stack_, max_trace_size_, &done_, &points_finished_);
      }
      else {
        AllKNTraversal:: template type <
        boost::mpl::vector6 <
        ArgMap,
        boost::mpl::bool_<QueryTree_t::IsBinary>,
        boost::mpl::bool_<ReferenceTree_t::IsBinary>,
        boost::mpl::bool_<false>, boost::mpl::bool_<false>,
        boost::mpl::int_<KNmode>
        >
        >::ComputeDualNeighborsProgressive(dual_tree_stack_, max_trace_size_, &done_, &points_finished_);
      }
    }
  }
  else {
    if (!is_dual_tree_) {
      if (allkn_->monochromatic_flag_ == true) {
        if (IS_RANGE_NEIGHBORS || kns_ == 1) {
          AllKNTraversal:: template type <
          boost::mpl::vector6 <
          ArgMap, boost::mpl::bool_<QueryTree_t::IsBinary>,
          boost::mpl::bool_<ReferenceTree_t::IsBinary>,
          boost::mpl::bool_<true>,
          boost::mpl::bool_<true>,
          boost::mpl::int_<KNmode>
          >
          >::ComputeAllSingleNeighborsProgressive(single_tree_stacks_,
                                                  &done_, &points_finished_);
        }
        else {
          AllKNTraversal:: template type <
          boost::mpl::vector6 <
          ArgMap,
          boost::mpl::bool_<QueryTree_t::IsBinary>,
          boost::mpl::bool_<ReferenceTree_t::IsBinary>,
          boost::mpl::bool_<false>, boost::mpl::bool_<true>,
          boost::mpl::int_<KNmode>
          >
          >::ComputeAllSingleNeighborsProgressive(single_tree_stacks_,
                                                  &done_, &points_finished_);
        }
      }
      else {
        if (IS_RANGE_NEIGHBORS || kns_ == 1) {
          AllKNTraversal:: template type <
          boost::mpl::vector6 <
          ArgMap,
          boost::mpl::bool_<QueryTree_t::IsBinary>,
          boost::mpl::bool_<ReferenceTree_t::IsBinary>,
          boost::mpl::bool_<true>,
          boost::mpl::bool_<false>,
          boost::mpl::int_<KNmode>
          >
          >::ComputeAllSingleNeighborsProgressive(single_tree_stacks_,
                                                  &done_, &points_finished_);
        }
        else {
          AllKNTraversal:: template type <
          boost::mpl::vector6 <
          ArgMap,
          boost::mpl::bool_<QueryTree_t::IsBinary>,
          boost::mpl::bool_<ReferenceTree_t::IsBinary>,
          boost::mpl::bool_<false>, boost::mpl::bool_<false>,
          boost::mpl::int_<KNmode>
          >
          >::ComputeAllSingleNeighborsProgressive(single_tree_stacks_,
                                                  &done_, &points_finished_);
        }
      }
    }
  }
  ++stage_;
  return *this;
}
/**
 * Constructors are generally very simple in fl-lite; most of the work is done by Init().  This is only
 * responsible for ensuring that the object is ready to be destroyed safely.
 */
template<typename ArgMap>
AllKN<ArgMap>::AllKN() {
  query_table_ = NULL;
  reference_table_ = NULL;
}

template<typename ArgMap>
AllKN<ArgMap>::~AllKN() {
}


/**
 * Computes the minimum squared distance between the bounding boxes
 * of two nodes
 */
template<typename ArgMap>
template<typename MetricType>
typename AllKN<ArgMap>::CalcPrecision_t
AllKN<ArgMap>::MinNodeDistSq(const MetricType &metric, QueryTree_t *query_node,
                             ReferenceTree_t* reference_node) {
  // node->bound() gives us the DHrectBound class for the node
  // It has a function MinDistanceSq which takes another DHrectBound
  return query_table_->get_node_bound(query_node).MinDistanceSq(
           metric, reference_table_->get_node_bound(reference_node));
}
/**
 * Computes the maximum squared distance between the bounding boxes
 * of two nodes
 */
template<typename ArgMap>
template<typename MetricType>
typename AllKN<ArgMap>::CalcPrecision_t
AllKN<ArgMap>::MaxNodeDistSq(const MetricType &metric, QueryTree_t *query_node,
                             ReferenceTree_t* reference_node) {
  // node->bound() gives us the DHrectBound class for the node
  // It has a function MaxDistanceSq which takes another DHrectBound
  return query_table_->get_node_bound(query_node).MaxDistanceSq(
           metric, reference_table_->get_node_bound(reference_node));
}

/**
 * Computes the minimum squared distances between a point and a
 * node's bounding box
 */
template<typename ArgMap>
template<typename MetricType>
typename AllKN<ArgMap>::CalcPrecision_t
AllKN<ArgMap>::MinPointNodeDistSq(const MetricType &metric,
                                  const QueryPoint_t& query_point,
                                  ReferenceTree_t* reference_node) {

  // node->bound() gives us the DHrectBound class for the node
  // It has a function MinDistanceSq which takes another DHrectBound
  return reference_table_->get_node_bound(reference_node).MinDistanceSq(
           metric, query_point);
}

/**
  * Computes the maximum squared distances between a point and a
  * node's bounding box
  */
template<typename ArgMap>
template<typename MetricType>
typename AllKN<ArgMap>::CalcPrecision_t
AllKN<ArgMap>::MaxPointNodeDistSq(const MetricType &metric,
                                  const QueryPoint_t& query_point,
                                  ReferenceTree_t* reference_node) {

  // node->bound() gives us the DHrectBound class for the node
  // It has a function MaxDistanceSq which takes another DHrectBound
  return reference_table_->get_node_bound(reference_node).MaxDistanceSq(
           metric, query_point);
}


template<typename ArgMap>
void  AllKN<ArgMap>::Init(ReferenceTable_t *const reference_table, QueryTable_t *const query_table) {
  DEBUG_ASSERT(reference_table != NULL);
  number_of_prunes_ = 0;
  if (query_table == NULL) {
    monochromatic_flag_ = true;
    //reference_table_ = query_table
    query_table_ = reference_table;
    reference_table_ = reference_table;
  }
  else {
    DEBUG_ASSERT(query_table != reference_table);
    monochromatic_flag_ = false;
    query_table_ = query_table;
    reference_table_ = reference_table;
  }
}

template<typename ArgMap>
template<typename PrecisionType>
void AllKN<ArgMap>::ResetStatistics(PrecisionType value) {
  stat_.clear();
  stat_.resize(query_table_->num_of_nodes());
  ResetStatisticsRecursion_(query_table_->get_tree(), query_table_, value);
}



/**
 * Computes the nearest neighbors and stores them in *results
 */
// Containers must be properly initialized
template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
void  AllKN<ArgMap>::ComputeNeighbors(const std::string &traversal_mode,
                                      const MetricType &metric,
                                      NeighborMethodType kns,
                                      ContainerDistType* neighbor_distances,
                                      ContainerIndType* neighbor_indices) {

  DynamicArguments_t info;
  ComputeNeighbors(traversal_mode,
                   metric,
                   kns,
                   &info,
                   neighbor_distances,
                   neighbor_indices);
}

/**
 * Computes the nearest neighbors and stores them in *results
 * In this version you get the chance to pass more parameters online
 */

template<typename ArgMap>
template < typename MetricType,
typename NeighborMethodType,
typename ContainerDistType,
typename ContainerIndType >
void  AllKN<ArgMap>::ComputeNeighbors(const std::string &traversal_mode,
                                      const MetricType &metric,
                                      NeighborMethodType kns,
                                      DynamicArguments_t *info,
                                      ContainerDistType* neighbor_distances,
                                      ContainerIndType* neighbor_indices) {

  static const bool IS_RANGE_NEIGHBORS =
    !boost::is_integral<NeighborMethodType>::value;

  boost::mpl::eval_if <
  IsSTLPairTrait<typename ContainerIndType::value_type>,
  InitContainerWithPairs,
  InitContainerWithoutPairs
  >::type::Init(neighbor_indices, kns * query_table_->n_entries());

  boost::mpl::eval_if <
  IsSTLPairTrait<typename ContainerIndType::value_type>,
  InitContainerWithPairs,
  InitContainerWithoutPairs
  >::type::Init(neighbor_distances, kns * query_table_->n_entries());

  CalcPrecision_t dist_so_far_init_value;
  if (KNmode == NearestNeighborAllKN) {
    std::fill(neighbor_distances->begin(),
              neighbor_distances->end(),
              std::numeric_limits<CalcPrecision_t>::max());
    if (IS_RANGE_NEIGHBORS == true) {
      dist_so_far_init_value = static_cast<CalcPrecision_t>(kns);
    }
    else {
      dist_so_far_init_value = std::numeric_limits<CalcPrecision_t>::max();
    }
    ResetStatistics(dist_so_far_init_value);
  }

  if (KNmode == FurthestNeighborAllKN) {
    std::fill(neighbor_distances->begin(),
              neighbor_distances->end(),
              -std::numeric_limits<CalcPrecision_t>::max());
    if (IS_RANGE_NEIGHBORS == true) {
      dist_so_far_init_value = static_cast<CalcPrecision_t>(kns);
    }
    else {
      dist_so_far_init_value = -std::numeric_limits<CalcPrecision_t>::max();
    }
    ResetStatistics(dist_so_far_init_value);
  }

  TreeArguments < MetricType,
  NeighborMethodType,
  ContainerDistType,
  ContainerIndType >  args(query_table_->get_tree(),
                           query_table_,
                           reference_table_->get_tree(),
                           reference_table_,
                           neighbor_distances,
                           neighbor_indices,
                           &stat_,
                           kns,
                           dist_so_far_init_value,
                           &metric,
                           info);
  if (traversal_mode == "dual") {
    if (monochromatic_flag_ == true) {
      if (IS_RANGE_NEIGHBORS == true ||  kns == 1) {
        AllKNTraversal:: template type <
        boost::mpl::vector6 <
        ArgMap,
        boost::mpl::bool_<QueryTree_t::IsBinary>,
        boost::mpl::bool_<ReferenceTree_t::IsBinary>,
        boost::mpl::bool_<true>,
        boost::mpl::bool_<true>,
        boost::mpl::int_<KNmode>
        >
        >::ComputeDualNeighborsRecursion(&args);
      }
      else {
        AllKNTraversal:: template type <
        boost::mpl::vector6 <
        ArgMap, boost::mpl::bool_<QueryTree_t::IsBinary>,
        boost::mpl::bool_<ReferenceTree_t::IsBinary>,
        boost::mpl::bool_<false>, boost::mpl::bool_<true>,
        boost::mpl::int_<KNmode>
        >
        >::
        ComputeDualNeighborsRecursion(&args);
      }
    }
    else {
      if (IS_RANGE_NEIGHBORS || kns == 1) {
        AllKNTraversal:: template type <
        boost::mpl::vector6 <
        ArgMap,
        boost::mpl::bool_<QueryTree_t::IsBinary>,
        boost::mpl::bool_<ReferenceTree_t::IsBinary>,
        boost::mpl::bool_<true>, boost::mpl::bool_<false>,
        boost::mpl::int_<KNmode>
        >
        >::ComputeDualNeighborsRecursion(&args);
      }
      else {
        AllKNTraversal:: template type <
        boost::mpl::vector6 <
        ArgMap,
        boost::mpl::bool_<QueryTree_t::IsBinary>,
        boost::mpl::bool_<ReferenceTree_t::IsBinary>,
        boost::mpl::bool_<false>, boost::mpl::bool_<false>,
        boost::mpl::int_<KNmode>
        >
        >::ComputeDualNeighborsRecursion(&args);
      }
    }
  }
  else {
    if (traversal_mode == "single") {
      if (monochromatic_flag_ == true) {
        if (IS_RANGE_NEIGHBORS || kns == 1) {
          AllKNTraversal:: template type <
          boost::mpl::vector6 <
          ArgMap, boost::mpl::bool_<QueryTree_t::IsBinary>,
          boost::mpl::bool_<ReferenceTree_t::IsBinary>,
          boost::mpl::bool_<true>,
          boost::mpl::bool_<true>,
          boost::mpl::int_<KNmode>
          >
          >::ComputeAllSingleNeighborsRecursion(&args);
        }
        else {
          AllKNTraversal:: template type <
          boost::mpl::vector6 <
          ArgMap,
          boost::mpl::bool_<QueryTree_t::IsBinary>,
          boost::mpl::bool_<ReferenceTree_t::IsBinary>,
          boost::mpl::bool_<false>, boost::mpl::bool_<true>,
          boost::mpl::int_<KNmode>
          >
          >::ComputeAllSingleNeighborsRecursion(&args);
        }
      }
      else {
        if (IS_RANGE_NEIGHBORS || kns == 1) {
          AllKNTraversal:: template type <
          boost::mpl::vector6 <
          ArgMap,
          boost::mpl::bool_<QueryTree_t::IsBinary>,
          boost::mpl::bool_<ReferenceTree_t::IsBinary>,
          boost::mpl::bool_<true>,
          boost::mpl::bool_<false>,
          boost::mpl::int_<KNmode>
          >
          >::ComputeAllSingleNeighborsRecursion(&args);
        }
        else {
          AllKNTraversal:: template type <
          boost::mpl::vector6 <
          ArgMap,
          boost::mpl::bool_<QueryTree_t::IsBinary>,
          boost::mpl::bool_<ReferenceTree_t::IsBinary>,
          boost::mpl::bool_<false>, boost::mpl::bool_<false>,
          boost::mpl::int_<KNmode>
          >
          >::ComputeAllSingleNeighborsRecursion(&args);
        }
      }
    }
    else {
      if (traversal_mode == "naive") {
        if (monochromatic_flag_ == true) {
          if (IS_RANGE_NEIGHBORS || kns == 1) {
            AllKNTraversal:: template type <
            boost::mpl::vector6 <
            ArgMap,
            boost::mpl::bool_<QueryTree_t::IsBinary>,
            boost::mpl::bool_<ReferenceTree_t::IsBinary>,
            boost::mpl::bool_<true>,
            boost::mpl::bool_<true>,
            boost::mpl::int_<KNmode>
            >
            >::ComputeNaive(&args);
          }
          else {
            AllKNTraversal:: template type <
            boost::mpl::vector6 <
            ArgMap,
            boost::mpl::bool_<QueryTree_t::IsBinary>,
            boost::mpl::bool_<ReferenceTree_t::IsBinary>,
            boost::mpl::bool_<false>, boost::mpl::bool_<true>,
            boost::mpl::int_<KNmode>
            >
            >::ComputeNaive(&args);
          }
        }
        else {
          if (IS_RANGE_NEIGHBORS || kns == 1) {
            AllKNTraversal:: template type <
            boost::mpl::vector6 <
            ArgMap,
            boost::mpl::bool_<QueryTree_t::IsBinary>,
            boost::mpl::bool_<ReferenceTree_t::IsBinary>,
            boost::mpl::bool_<true>, boost::mpl::bool_<false>,
            boost::mpl::int_<KNmode>
            >
            >::ComputeNaive(&args);
          }
          else {
            AllKNTraversal:: template type <
            boost::mpl::vector6 <
            ArgMap,
            boost::mpl::bool_<QueryTree_t::IsBinary>,
            boost::mpl::bool_<ReferenceTree_t::IsBinary>,
            boost::mpl::bool_<false>, boost::mpl::bool_<false>,
            boost::mpl::int_<KNmode>
            >
            >::ComputeNaive(&args);
          }
        }
      }
      else {
        fl::logger->Die() << "This choice " << traversal_mode
        << " is not supported";
      }
    }
  }
  number_of_prunes_ = info->num_of_prunes();
} // ComputeNeighbors


template<typename ArgMap>
template<typename TreeType, typename TableType, typename PrecisionType>
void AllKN<ArgMap>::ResetStatisticsRecursion_(TreeType *node,
    TableType *table, PrecisionType value) {

  stat_[table->get_node_id(node)].set_dist_so_far(value);
  if (!table->node_is_leaf(node)) {
    ResetStatisticsRecursion_(table->get_node_left_child(node), table, value);
    ResetStatisticsRecursion_(table->get_node_right_child(node), table, value);
  }
}


} // namespace ml
}   // namespace fl


#endif
