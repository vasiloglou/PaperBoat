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
#ifndef FL_LITE_MLPACK_CLUSTERING_KMEANS_DEV_H
#define FL_LITE_MLPACK_CLUSTERING_KMEANS_DEV_H
#include <set>
#include "mlpack/clustering/kmeans.h"
#include "mlpack/clustering/kmeans_defs.h"
#include "fastlib/tree/bounds.h"

namespace fl {
namespace ml {


template<typename KMeansMap>
KMeans<KMeansMap>::KMeans() {
  curr_centroids_ = NULL;
  metric_ = NULL;
  new_centroids_ = NULL;
  k_ = -1;
  final_distortion_ = -1;
  max_iterations_= -1;
  minimum_cluster_movement_threshold_ = 0;
}


template<typename KMeansMap>
KMeans<KMeansMap>::~KMeans() {
  delete[] curr_centroids_;
  delete metric_;
  delete[] new_centroids_;
}

/**
* If centroids are passed in they must already have been initialized.
*/
template<typename KMeansMap>
void KMeans<KMeansMap>::Init(const int k_in,
                             typename KMeans<KMeansMap>::Table_t* table_in,
                             const CentroidPoint_t* centroids_in,
                             const typename KMeans<KMeansMap>::Metric_t* metric_in) {
  DEBUG_ASSERT(k_in > 1);
  Destroy(); // if previously initilized then free memory etc.
  CentroidPoint_t* centroids = new CentroidPoint_t[k_in];
  for (int i = 0; i < k_in; i++) {
    centroids[i].Copy(centroids_in[i]);
  }
  PrivateInit(k_in, table_in, centroids, new Metric_t(*metric_in));
}

template<typename KMeansMap>
void KMeans<KMeansMap>::Init(const int k_in,
                             typename KMeans<KMeansMap>::Table_t* table_in,
                             const typename KMeans<KMeansMap>::Metric_t* metric_in) {
  DEBUG_ASSERT(k_in > 1);
  Destroy(); // if previously initilized then free memory etc.
  CentroidPoint_t* centroids = new CentroidPoint_t[k_in];
  AssignInitialCentroids(k_in, centroids, table_in);
  PrivateInit(k_in, table_in, centroids, new Metric_t(*metric_in));
}

/**
* If centroids are passed in they must already have been initialized.
*/
template<typename KMeansMap>
void KMeans<KMeansMap>::Init(const int k_in,
                             typename KMeans<KMeansMap>::Table_t* table_in,
                             const CentroidPoint_t* centroids_in) {
  DEBUG_ASSERT(k_in > 1);
  Destroy(); // if previously initilized then free memory etc.
  CentroidPoint_t* centroids = new CentroidPoint_t[k_in];
  for (int i = 0; i < k_in; i++) {
    centroids[i].Copy(centroids_in[i]);
  }
  PrivateInit(k_in, table_in, centroids, new Metric_t());
}

template<typename KMeansMap>
void KMeans<KMeansMap>::Init(const int k_in,
                             typename KMeans<KMeansMap>::Table_t* table_in) {
  DEBUG_ASSERT(k_in > 1);
  Destroy(); // if previously initilized then free memory etc.
  CentroidPoint_t* centroids = new CentroidPoint_t[k_in];
  AssignInitialCentroids(k_in, centroids, table_in);
  PrivateInit(k_in, table_in, centroids, new Metric_t());
}

/**
* This function keep's calling the assignment/update function
* till the centroids do not change. Note: if by chance the randomly
* initialized centroids happen to be the final centroids (extremely rare)
* then instead of finishing in one iteration, it will take 2 iterations.
* This is because 'point_cluster_assignments' is initially set to all 0.
* And any change in this is used to indicated if 'something_changed_'. So
* in the first iteration it will change even though the centroids got the
* same value. In the next iteration it will terminate.
*/
template<typename KMeansMap>
index_t KMeans<KMeansMap>::RunKMeans(const std::string traversal_mode) {
  // assign it to zero'th cluster
  point_centroid_assignments_.assign(table_->n_entries(), 0);
  if (traversal_mode == "tree") {
    return TreeBasedKMeans();
  }
  else if (traversal_mode == "naive") {
    return NaiveKMeans();
  }
  else {
    fl::logger->Die() << "Unrecognized traversal mode " << traversal_mode;
    return 0;
  }
}

/**
 * Copies the centroids into the passed table.
 * Assumes the table is already initialized with K entries
 * of type CentroidPoint
 */
template<typename KMeansMap>
void KMeans<KMeansMap>::GetCentroids(CentroidTable_t* centroids) {
  for (int i = 0; i < k_; i++) {
	CentroidPoint_t point;
    centroids->get(i, &point);
    point.template dense_point<typename CentroidPoint_t::CalcPrecision_t>().CopyValues(curr_centroids_[i]);
  }

}

/**
 * Copies memberships into the passed in vector.
 * if the mode is 0 then it initializes the container
 * if the mode is 1 then assumes space has been preallocated
 */
template<typename KMeansMap>
template<typename ContainerType>
void KMeans<KMeansMap>::GetMemberships(ContainerType* memberships_out) {
  index_t size = point_centroid_assignments_.size();
  typename ContainerType::Point_t point;
  for(index_t i=0; i < size; ++i) {
    memberships_out->get(i, &point);
    point.set(0, point_centroid_assignments_[i]);
  }
}

/**
 * Copies memberships into the passed in vector.
 * if the mode is 0 then it initializes the container
 * if the mode is 1 then assumes space has been preallocated
 */
template<typename KMeansMap>
template<typename ContainerType>
void KMeans<KMeansMap>::GetMembershipCounts(ContainerType* counts_out) {
  index_t size = centroid_point_counts_.size();
  for(index_t i=0; i < size; ++i) {
    counts_out->set(i, centroid_point_counts_[i]);
  }
}


template<typename KMeansMap>
typename KMeans<KMeansMap>::CalcPrecision_t KMeans<KMeansMap>::GetDistortion() {
    return final_distortion_;
}

template<typename KMeansMap>
index_t KMeans<KMeansMap>::NaiveKMeans() {
  index_t iterations = 0;
  CalcPrecision_t distortion = 0;
  do {
    something_changed_ = false;
    centroid_point_counts_.assign(k_, 0);
    for (int i = 0; i < k_; i++) {
      new_centroids_[i].SetAll(0);
    }
	  distortion = 0;
    for (int i = 0; i < table_->n_entries(); i++) {
      Point_t point;
      table_->get(i, &point);
	  CalcPrecision_t distance_square_out;
      int closest_centroid = GetClosestCentroid(&point, distance_square_out);
      AssignPointToCentroid(&point, i, closest_centroid);
	  distortion += distance_square_out;
    }
    //scale
    for (int i = 0; i < k_; i++) {
      if(centroid_point_counts_[i] > 0) {    // if centroid has any points
        fl::la::SelfScale((CalcPrecision_t)1.0 / centroid_point_counts_[i], &new_centroids_[i]);
      } else {    // if no points ensure centroid remains unmoved
        new_centroids_[i].CopyValues(curr_centroids_[i]);
      }
    }
    // make new centroids current
    CentroidPoint_t* temp = curr_centroids_;
    curr_centroids_ = new_centroids_;
    new_centroids_ = temp;
    fl::logger->Debug() << "naive iteration="<<iterations
              <<", distortion="<<distortion/table_->n_entries();
    iterations++;
    if(BreakOnMinimumClusterMovement()) {
      break;
    }
  }
  while (something_changed_ && (max_iterations_ == -1 || iterations <= max_iterations_));
  fl::logger->Message() << "Total iterations: " << iterations;
  final_distortion_ = (distortion/table_->n_entries());
  return iterations;
}

template<typename KMeansMap>
bool KMeans<KMeansMap>::BreakOnMinimumClusterMovement() {
    if(minimum_cluster_movement_threshold_ > 0) {
      // ok let's see how much each cluster moved
      CalcPrecision_t max_cluster_movement = (-std::numeric_limits<CalcPrecision_t>::max());
      for (int i = 0; i < k_; i++) {
        CalcPrecision_t distance = metric_->DistanceSq(new_centroids_[i], curr_centroids_[i]);
        max_cluster_movement = (distance > max_cluster_movement? distance : max_cluster_movement);
      }
      if(max_cluster_movement < minimum_cluster_movement_threshold_) {
         fl::logger->Message() << "The maximum cluster movement is extremely low at " << max_cluster_movement;
         fl::logger->Message() << "Finishing up.";
        return true;
      }
    }
    return false;
}

template<typename KMeansMap>
index_t KMeans<KMeansMap>::TreeBasedKMeans() {
  Tree_t* root = table_->get_tree();
  index_t iterations = 0;
  std::list<index_t> blacklisted_orig;
  for(index_t i = 0; i < k_; i++) {
    blacklisted_orig.push_back(i);
  }
  do {
    something_changed_ = false;
    centroid_point_counts_.assign(k_, 0);
    for (int i = 0; i < k_; i++) {
      new_centroids_[i].SetAll(0);
    }
    std::list<index_t> blacklisted(blacklisted_orig);
    AssignUpdateStepRecursive(root, blacklisted);
    //scale
    for (int i = 0; i < k_; i++) {
      if(centroid_point_counts_[i] > 0) {    // if centroid has any points
        fl::la::SelfScale((CalcPrecision_t)1.0 / centroid_point_counts_[i], &new_centroids_[i]);
      } else {    // if no points ensure centroid remains unmoved
        new_centroids_[i].CopyValues(curr_centroids_[i]);
      }
    }
    // make new centroids current
    CentroidPoint_t* temp = curr_centroids_;
    curr_centroids_ = new_centroids_;
    new_centroids_ = temp;
    iterations++;
    if(BreakOnMinimumClusterMovement()) {
      break;
    }
  }  while (something_changed_ &&  (max_iterations_ == -1 || iterations <= max_iterations_));
  fl::logger->Message() << "Total iterations: " << iterations;
  typename Table_t::TreeIterator point_it(*table_, table_->get_tree());
  point_it.Reset();
  CalcPrecision_t distortion = 0;
  while (point_it.HasNext()) {
	Point_t point;
	index_t point_id;
    point_it.Next(&point, &point_id);
    distortion += metric_->DistanceSq(point, 
        curr_centroids_[point_centroid_assignments_[point_id]]. 
        template dense_point<typename CentroidPoint_t::CalcPrecision_t>());
  }
  final_distortion_ = (distortion/table_->n_entries());
  return iterations;
}


template<typename KMeansMap>
void KMeans<KMeansMap>::KMeansPlusPlus(const int k,
        double probability, 
        Metric_t &metric,
        Table_t &table,
        CentroidTable_t* centroid_table) {
  DEBUG_ASSERT(k >= 1);
  // choose the first point
  index_t rand_index=fl::math::Random(index_t(0), table.n_entries()-1);
  typename Table_t::Point_t point;
  table.get(rand_index, &point);
  typename CentroidTable_t::Point_t cent;
  centroid_table->get(0, &cent);
  for(typename Table_t::Point_t::iterator it=point.begin();
        it!=point.end(); ++it) {
    cent.set(it.attribute(), it.value());
  }

 // std::cout<<std::endl;
 // cent.Print(std::cout,",");
 //  std::cout<<std::endl;
  for(index_t k=1; k<centroid_table->n_entries(); ++k) {
    std::vector<std::pair<CalcPrecision_t, index_t> > distances(table.n_entries());
    for(index_t i=0; i<table.n_entries(); ++i) {
      table.get(i, &point);
      double best_distance=std::numeric_limits<CalcPrecision_t>::max();
      distances[i].second=i;
      for(index_t l=0; l<k; ++l) {
        centroid_table->get(l, &cent);
        CalcPrecision_t distance=metric.DistanceSq(cent.template dense_point<
            CalcPrecision_t>(), point);
        if (distance<best_distance) {
          best_distance=distance;
        }
      }
      distances[i].first=best_distance;
    }
    index_t lo=0;
    index_t hi=table.n_entries()-1;
    do {
      index_t mid=(hi+lo)/2;
      std::nth_element(distances.begin()+lo, distances.begin()+mid, distances.begin()+hi);
      if (fl::math::Random(0.0, 1.0)<probability) {
        lo=mid;
      } else {
        hi=mid;
      }
    }
    while(hi-lo<10);
    
    table.get(distances[(hi+lo)/2].second, &point);
    centroid_table->get(k, &cent);
    for(typename Table_t::Point_t::iterator it=point.begin();
        it!=point.end(); ++it) {
      cent.set(it.attribute(), it.value());
    }
    // cent.Print(std::cout, ",");
    // std::cout<<std::endl;
  }
}

/*
* Initializes the centroids to coincide with random points in
* the dataset. Points passed are initilized in the function.
*/
template<typename KMeansMap>
void KMeans<KMeansMap>::AssignInitialCentroids(const int k,
    CentroidPoint_t* points,
    typename KMeans<KMeansMap>::Table_t* table) {
  DEBUG_ASSERT(k >= 1);
  int i=0;
  std::set<index_t> unique_ids;
  while (i<k) {
    index_t rand_index = fl::math::Random(index_t(0), table->n_entries()-1);
    if (unique_ids.find(rand_index)!=unique_ids.end()) {
      continue;
    }
    Point_t random_point;
    table->get(rand_index, &random_point);
    points[i].Copy(random_point);
    i++;
    unique_ids.insert(rand_index);
  } // for each centroid
}

template<typename KMeansMap>
void KMeans<KMeansMap>::AssignInitialCentroids(const int k, CentroidTable_t* centroid_table, Table_t &table) {
  DEBUG_ASSERT(k >= 1);
  typename CentroidTable_t::Point_t point1;
  typename Table_t::Point_t point2;
  int i=0;
  std::set<index_t> unique_ids;
  while(i<k) {
    int rand_index = fl::math::Random(index_t(0), table.n_entries()-1);
    if (unique_ids.find(rand_index)!=unique_ids.end()) {
      continue;
    }
    centroid_table->get(i, &point1);
    table.get(rand_index, &point2);
    for(typename Table_t::Point_t::iterator it=point2.begin();
        it!=point2.end(); ++it) {
      point1.set(it.attribute(), it.value());
    }
    ++i;
    unique_ids.insert(rand_index);
  }

}

// returns the index of the closest centroid
// if there is a tie returns the one with the lower index
template<typename KMeansMap>
int KMeans<KMeansMap>::GetClosestCentroid(
	const typename KMeans<KMeansMap>::Point_t *point,
	CalcPrecision_t& distance_square_out) {
  CalcPrecision_t min_distance = metric_->DistanceSq(
      curr_centroids_[0].template dense_point<
          typename CentroidPoint_t::CalcPrecision_t>(), *point);
  int centroid_idx = 0;
  for (int i = 1; i < k_; i++) {
    // get the min distance from node to centroid
    CalcPrecision_t distance = metric_->DistanceSq(
        curr_centroids_[i]. template dense_point<
        typename CentroidPoint_t::CalcPrecision_t>(), *point);
    if (distance < min_distance) {
      min_distance = distance;
      centroid_idx = i;
    }
  }
  distance_square_out = min_distance;
  return centroid_idx;
}

// returns the index of the closest centroid
// if there is a tie returns the one with the lower index
template<typename KMeansMap>
int KMeans<KMeansMap>::GetClosestCentroid(const typename KMeans<KMeansMap>::Point_t *point, std::list<index_t>& blacklisted) {
  CalcPrecision_t min_distance = std::numeric_limits<CalcPrecision_t>::max();
  int centroid_idx = -1;
  for(std::list<index_t>::iterator it = blacklisted.begin(); it != blacklisted.end(); it++) {
    index_t i = *it;
    // get the min distance from node to centroid
    CalcPrecision_t distance = metric_->DistanceSq(
        curr_centroids_[i].template 
        dense_point<typename CentroidPoint_t::CalcPrecision_t>(), *point);
    if (distance < min_distance) {
      min_distance = distance;
      centroid_idx = i;
    }
  }
  return centroid_idx;
}

/**
* Returns true if 1 and only 1 centroid is closest to the node.
* It puts the index of the first of the closest centroids into
* centroid_idx i.e. even if the function returns false indicating
* that there are 2 centroids which share smallest minimum distances
* with the node, centroid_idx will contain one of these centroids
* and particularly the one with the lower index;
*/
template<typename KMeansMap>
bool KMeans<KMeansMap>::GetClosestCentroid(typename KMeans<KMeansMap>::Tree_t* node,
    int& centroid_idx,
    std::list<index_t> &blacklisted) {
  CalcPrecision_t min_distance = std::numeric_limits<CalcPrecision_t>::max();	//table_->get_node_bound(node).MinDistanceSq(*metric_, curr_centroids_[0]);
  bool shortest_dist_center_exists = true; // if 1 and only 1 centroid is closest
  centroid_idx = -1;
  //for (index_t p = 0; p < blacklisted.size(); p++) {
  for(std::list<index_t>::iterator it = blacklisted.begin(); it != blacklisted.end(); it++ ) {
    index_t i = *it;
    // get the min distance from node to centroid
    CalcPrecision_t distance =
      table_->get_node_bound(node).MinDistanceSq(*metric_, 
          curr_centroids_[i].
          template dense_point<typename CentroidPoint_t::CalcPrecision_t>());
    if (distance < min_distance) {
      shortest_dist_center_exists = true;
      min_distance = distance;
      centroid_idx = i;
    }
    else if (distance == min_distance) {
      shortest_dist_center_exists = false;
    }
  }
  return shortest_dist_center_exists;
}

/**
* This function returns true if centroid_idx dominates all other
* non-blacklisted centroids with repspect to this node. It also
* blacklists all centroids which were previously not blacklisted
* but are dominated by this centroid (centroid_idx).
*/
template<typename KMeansMap>
template<typename TableType>
bool KMeans<KMeansMap>::BallKdNullaryMetaFunction1::Do(TableType *table,
    Tree_t *node,
    const Metric_t *metric,
    CentroidPoint_t *curr_centroids,
    index_t k,
    const int centroid_idx,
    std::list<index_t> &blacklisted) {
  bool dominates = true;
  for(std::list<index_t>::iterator it = blacklisted.begin(); it != blacklisted.end();) {
    bool incerement_iterator = true;
    index_t i = *it;
    if (likely(i != centroid_idx)) {
      // find corner closest to centroid i
      CentroidPoint_t p_extreme;
      p_extreme.template dense_point<typename CentroidPoint_t::CalcPrecision_t>().
        Copy(table->get_node_bound(node).center());
      CalcPrecision_t radius_of_ball = table->get_node_bound(node).radius(); // ball info
      CentroidPoint_t centroid_i_minus_centroid_idx;
      centroid_i_minus_centroid_idx.Copy(curr_centroids[i]);
      fl::la::SubFrom(curr_centroids[centroid_idx], &centroid_i_minus_centroid_idx);
      CalcPrecision_t norm = fl::la::LengthEuclidean(centroid_i_minus_centroid_idx);
      fl::la::SelfScale((radius_of_ball / norm), &centroid_i_minus_centroid_idx);
      fl::la::AddTo(centroid_i_minus_centroid_idx, &p_extreme);
      // if this corner is closer to centroid i return false
      // centroid_idx does not dominate
      if (metric->DistanceSq(curr_centroids[i], p_extreme)
          <= metric->DistanceSq(curr_centroids[centroid_idx], p_extreme)) {
        dominates = false;
      }
      else { // this centroid is dominated and thus blacklisted
        it = blacklisted.erase(it);
		incerement_iterator = false;
      }
    }
	if(incerement_iterator) {
	  it++;
	}
  } // for loop
  return dominates;
}

template<typename KMeansMap>
template<typename TableType>
bool KMeans<KMeansMap>::BallKdNullaryMetaFunction2::Do(TableType *table,
    Tree_t *node,
    const Metric_t *metric,
    CentroidPoint_t *curr_centroids,
    index_t k,
    const int centroid_idx,
    std::list<index_t> &blacklisted) {
  bool dominates = true;
  Point_t p_extreme;
  Point_t dummy_point;
  table->get(0, &dummy_point);
  for(std::list<index_t>::iterator it = blacklisted.begin(); it != blacklisted.end();) {
    bool incerement_iterator = true;
    index_t i = *it;
    if (likely(i != centroid_idx)) {
      // find corner closest to centroid i
      p_extreme.Copy(dummy_point);
      for (int j = 0; j < table->get_node_bound(node).dim(); j++) {
        CalcPrecision_t hi = table->get_node_bound(node).get(j).hi;
        CalcPrecision_t lo = table->get_node_bound(node).get(j).lo;
        p_extreme.set(j, curr_centroids[i][j] > curr_centroids[centroid_idx][j] ? hi : lo);
      }
      // if this corner is closer to centroid i return false
      // centroid_idx does not dominate
      if (metric->DistanceSq(curr_centroids[i].template 
            dense_point<typename CentroidPoint_t::CalcPrecision_t>(), p_extreme)
          <= metric->DistanceSq(curr_centroids[centroid_idx], p_extreme)) {
        dominates = false;
      }
      else { // this centroid is dominated and thus blacklisted
        it = blacklisted.erase(it);
		incerement_iterator = false;
      }
    } // if not the centroid in question
	if(incerement_iterator) {
		it++;
	}
  }// for loop
  return dominates;
}

template<typename KMeansMap>
bool KMeans<KMeansMap>::Dominates(const int centroid_idx,
                                  typename KMeans<KMeansMap>::Tree_t* node,
                                  std::list<index_t> &blacklisted) {
  return boost::mpl::if_ <
         boost::is_same <
         typename Tree_t::TreeSpec_t,
         typename fl::tree::MetricTree
         > ,
         BallKdNullaryMetaFunction1,
         BallKdNullaryMetaFunction2
         >::type::Do(table_, node, metric_, curr_centroids_, k_, centroid_idx, blacklisted);
} // Dominates

template<typename KMeansMap>
void KMeans<KMeansMap>::KMeansBaseCase(typename KMeans<KMeansMap>::Tree_t* node, std::list<index_t>& blacklisted) {
  typename Table_t::TreeIterator point_it(*table_, node);
  point_it.Reset();
  AssignPoints(point_it, blacklisted);
}

template<typename KMeansMap>
void KMeans<KMeansMap>::AssignPoints(typename KMeans<KMeansMap>::Table_t::TreeIterator &point_it, std::list<index_t>& blacklisted) {
  Point_t point;
  index_t  point_id;
  while (point_it.HasNext()) {
    // Get the query point from the matrix
    point_it.Next(&point, &point_id);
    int closest_centroid = GetClosestCentroid(&point, blacklisted);
    AssignPointToCentroid(&point, point_id, closest_centroid);
  }
}

/**
* Assigns a point to a centroid.
*/
template<typename KMeansMap>
inline void KMeans<KMeansMap>::AssignPointToCentroid(const typename KMeans<KMeansMap>::Point_t* point,
    const int point_id, const int centroid_idx) {
  something_changed_ = !something_changed_ ?
                       point_centroid_assignments_[point_id] != centroid_idx : true;
  point_centroid_assignments_[point_id] = centroid_idx; // assign
  centroid_point_counts_[centroid_idx] = centroid_point_counts_[centroid_idx] + 1; // update count
  //for(index_t i = 0 ; i < table_->n_attributes(); i++) {
  fl::la::AddTo(*point, &(new_centroids_[centroid_idx].
        template dense_point<typename CentroidPoint_t::CalcPrecision_t>()));
    //new_centroids_[centroid_idx][i] += (*point)[i];
  //}
}

template<typename KMeansMap>
void KMeans<KMeansMap>::AssignAllPointsToCentroid(
  typename KMeans<KMeansMap>::Tree_t* node,
  const int centroid_idx) {
  Point_t point;
  index_t  point_id;
  typename Table_t::TreeIterator point_it(*table_, node);
  point_it.Reset();
  while (point_it.HasNext()) {
    point_it.Next(&point, &point_id);
    AssignPointToCentroid(&point, point_id, centroid_idx);
  }
}

template<typename KMeansMap>
void KMeans<KMeansMap>::AssignUpdateStepRecursive(typename KMeans<KMeansMap>::Tree_t* node,
    std::list<index_t>& blacklisted) {
  if (table_->node_is_leaf(node)) {
    KMeansBaseCase(node, blacklisted); // base case
  }
  else { // check if any center is dominant
    int closest_centroid_idx = 0; // valid only of shortest_dist_center_exists is true
    bool shortest_dist_center_exists = GetClosestCentroid(node, closest_centroid_idx, blacklisted);
    // This is done anyway to eliminate redundant centroids.
    bool dominates = Dominates(closest_centroid_idx, node, blacklisted);
    if (shortest_dist_center_exists && dominates) {
      AssignAllPointsToCentroid(node, closest_centroid_idx);
    }
    else {
      Tree_t *query_left = table_->get_node_left_child(node);
      Tree_t *query_right = table_->get_node_right_child(node);
      std::list<index_t> left_blacklisted(blacklisted);
      std::list<index_t> right_blacklisted(blacklisted);
      AssignUpdateStepRecursive(query_left, left_blacklisted);
      AssignUpdateStepRecursive(query_right, right_blacklisted);
    }
  }
}

/**
* Avoids issues if Init(...) is called more than once
* on the same object.
*/
template<typename KMeansMap>
void KMeans<KMeansMap>::Destroy() {
  delete[] new_centroids_;
  delete[] curr_centroids_;
  delete metric_;
  new_centroids_ = NULL;
  curr_centroids_ = NULL;
  metric_ = NULL;
  table_ = NULL;
  centroid_point_counts_.clear();
  point_centroid_assignments_.clear();
  k_ = -1;
}

template<typename KMeansMap>
void KMeans<KMeansMap>::PrivateInit(
  const int k_in,
  typename KMeans<KMeansMap>::Table_t* table_in,
  CentroidPoint_t* centroids_in,
  const typename KMeans<KMeansMap>::Metric_t* metric_in) {
  k_ = k_in;
  table_ = table_in;
  metric_ = metric_in;
  curr_centroids_ = centroids_in;
  new_centroids_ = new CentroidPoint_t[k_in];
  Point_t dummy_point;
  table_in->get(0, &dummy_point);
  std::vector<index_t> dims(1,dummy_point.size());
  for (int i = 0; i < k_in; i++) {
    new_centroids_[i].Init(dims);
    new_centroids_[i].template 
      dense_point<typename CentroidPoint_t::CalcPrecision_t>().CopyValues(dummy_point);
  }
}
} // ml
} // fl

#endif

