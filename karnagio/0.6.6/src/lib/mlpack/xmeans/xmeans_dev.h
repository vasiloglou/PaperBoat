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
#ifndef FL_LITE_MLPACK_XMEANS_XMEANS_DEV_H
#define FL_LITE_MLPACK_XMEANS_XMEANS_DEV_H

#include "mlpack/xmeans/xmeans.h"
#include "fastlib/tree/bounds.h"

namespace fl {
namespace ml {

	template<typename XMeansMap>
	XMeans<XMeansMap>::KMeans::KMeans() {
		new_centroids_ = NULL;
	}


	template<typename XMeansMap>
	XMeans<XMeansMap>::KMeans::~KMeans() {
		delete[] new_centroids_;
	}

		template<typename XMeansMap>
		void XMeans<XMeansMap>::KMeans::Init(
			Table_t* table_in, 
			CentroidPoint_t* parent_centroids_in, 
			CentroidPoint_t* children_centroids_in, 
			index_t* memberships_out,
			index_t* centroid_counts_out,
			fl::table::TableVector<index_t>& parent_memberships_in,
			const int k_parents_in, 
			const Metric_t* metric_in) {
				k_parents_ = k_parents_in;
				curr_centroids_ = parent_centroids_in;
				parent_centroids_ = parent_centroids_in;
				children_centroids_ = children_centroids_in;
				new_centroids_ = new CentroidPoint_t[k_parents_in * 2];
				centroid_counts_out_ = centroid_counts_out;
				memberships_out_ = memberships_out;
				parent_memberships_in_ = &parent_memberships_in;
				table_ = table_in;
				metric_ = metric_in;
				for (int i = 0; i < k_parents_in * 2; i++) {
					new_centroids_[i].Copy(parent_centroids_in[0]);
				}
				// assign all to invalid cluster
				for(int i = 0; i < table_->n_entries(); i++) {
					memberships_out_[i] = -1;
				}
		}

		template<typename XMeansMap>
		void XMeans<XMeansMap>::KMeans::RunKMeans() {
			do {
				something_changed_ = false;
				for (int i = 0; i < k_parents_ * 2; i++) {
					new_centroids_[i].SetAll(0);
					centroid_counts_out_[i] = 0;
				}
				std::vector<bool> blacklisted;
				blacklisted.assign(k_parents_, false);
				AssignUpdateStepRecursive(table_->get_tree(), blacklisted, -1);
				//scale
				for (int i = 0; i < k_parents_ * 2; i++) {
					fl::la::SelfScale((CalcPrecision_t)1.0 / centroid_counts_out_[i], &new_centroids_[i]);
				}
				// make new centroids current
				CentroidPoint_t* temp = children_centroids_;
				for (int i = 0; i < k_parents_ * 2; i++) {
					temp[i].CopyValues(new_centroids_[i]);
				}
			} while (something_changed_);
		}
	
		template<typename XMeansMap>
		void XMeans<XMeansMap>::KMeans::AssignUpdateStepRecursive(
			Tree_t* node,
			std::vector<bool>& blacklisted,
			const index_t parent_idx) {
				if (table_->node_is_leaf(node)) {
					if(parent_idx != -1) {
						KMeansBaseCase(node, parent_idx); // base case
					} else {
						ChildrenKMeansBaseCase(node);
					}
				} else { // check if any center is dominant
					int closest_centroid_idx = 0; // valid only of shortest_dist_center_exists is true
					bool shortest_dist_center_exists = 
						GetClosestCentroid(node, closest_centroid_idx, parent_idx == -1? k_parents_: 2);
					// This is done anyway to eliminate redundant centroids.
					bool dominates = Dominates(closest_centroid_idx, node, blacklisted, parent_idx == -1? k_parents_: 2);
					if (shortest_dist_center_exists && dominates) {
						if(parent_idx != -1) {
							AssignAllPointsToCentroid(node, closest_centroid_idx, parent_idx);
						} else {
							// now change the values
							curr_centroids_ = children_centroids_ + (2 * closest_centroid_idx);
							new_centroids_ = new_centroids_ + (2 * closest_centroid_idx);
							centroid_counts_out_ = centroid_counts_out_ + (2 * closest_centroid_idx);
							// run kmeans
							std::vector<bool> blacklisted;
							blacklisted.assign(2, false);
							AssignUpdateStepRecursive(node, blacklisted, closest_centroid_idx);
							// restore values before continuing
							curr_centroids_ = parent_centroids_;
							new_centroids_ = new_centroids_ - (2 * closest_centroid_idx);
							centroid_counts_out_ = centroid_counts_out_ - (2 * closest_centroid_idx);
						}
					} else {
						Tree_t *query_left = table_->get_node_left_child(node);
						Tree_t *query_right = table_->get_node_right_child(node);
						std::vector<bool> left_blacklisted(blacklisted);
						std::vector<bool> right_blacklisted(blacklisted);
						AssignUpdateStepRecursive(query_left, left_blacklisted, parent_idx);
						AssignUpdateStepRecursive(query_right, right_blacklisted, parent_idx);
					}
				}
		}

		// returns the index of the closest centroid
		// if there is a tie returns the one with the lower index
		template<typename XMeansMap>
		int XMeans<XMeansMap>::KMeans::GetClosestCentroid(const Point_t *point, index_t k) {
			CalcPrecision_t min_distance = metric_->DistanceSq(
          curr_centroids_[0].template dense_point<typename CentroidPoint_t::CalcPrecision_t>(), *point);
			int centroid_idx = 0;
			for (int i = 1; i < k; i++) {
				// get the min distance from node to centroid
				CalcPrecision_t distance = metric_->DistanceSq(
            curr_centroids_[i].template dense_point<typename CentroidPoint_t::CalcPrecision_t>(), *point);
				if (distance < min_distance) {
					min_distance = distance;
					centroid_idx = i;
				}
			}
			return centroid_idx;
		}

		template<typename XMeansMap>
		bool XMeans<XMeansMap>::KMeans::GetClosestCentroid(
			Tree_t* node,
			int& centroid_idx,
			index_t k) {
				CalcPrecision_t min_distance =
					table_->get_node_bound(node).MinDistanceSq(*metric_, 
              curr_centroids_[0].template dense_point<typename CentroidPoint_t::CalcPrecision_t>());
				bool shortest_dist_center_exists = true; // if 1 and only 1 centroid is closest
				centroid_idx = 0;
				for (int i = 1; i < k; i++) {
					// get the min distance from node to centroid
					CalcPrecision_t distance =
						table_->get_node_bound(node).MinDistanceSq(*metric_, 
                curr_centroids_[i].template dense_point<typename CentroidPoint_t::CalcPrecision_t>());
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

		template<typename XMeansMap>
		template<typename TableType>
		bool XMeans<XMeansMap>::KMeans::BallKdNullaryMetaFunction1::Do(TableType *table,
			Tree_t *node,
			const Metric_t *metric,
			CentroidPoint_t *curr_centroids,
			index_t k,
			const int centroid_idx,
			std::vector<bool> &blacklisted) {
				bool dominates = true;
				for (int i = 0; i < k; i++) {
					if (i != centroid_idx && !blacklisted[i]) {
						// find corner closest to centroid i
						CentroidPoint_t p_extreme;
						p_extreme.template dense_point<typename CentroidPoint_t::CalcPrecision_t>().Copy(table->get_node_bound(node).center());
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
							blacklisted[i] = true;
						}
					}
				}
				return dominates;
		}

		template<typename XMeansMap>
		template<typename TableType>
		bool XMeans<XMeansMap>::KMeans::BallKdNullaryMetaFunction2::Do(TableType *table,
			Tree_t *node,
			const Metric_t *metric,
			CentroidPoint_t *curr_centroids,
			index_t k,
			const int centroid_idx,
			std::vector<bool> &blacklisted) {
				bool dominates = true;
				Point_t p_extreme;
				Point_t dummy_point;
				table->get(0, &dummy_point);
				for (int i = 0; i < k; i++) {
					if (i != centroid_idx && !blacklisted[i]) {
						// find corner closest to centroid i
						p_extreme.Copy(dummy_point);
						for (int j = 0; j < table->get_node_bound(node).dim(); j++) {
							CalcPrecision_t hi = table->get_node_bound(node).get(j).hi;
							CalcPrecision_t lo = table->get_node_bound(node).get(j).lo;
							p_extreme.set(j, curr_centroids[i][j] > curr_centroids[centroid_idx][j] ? hi : lo);
						}
						// if this corner is closer to centroid i return false
						// centroid_idx does not dominate
						if (metric->DistanceSq(curr_centroids[i], p_extreme)
							<= metric->DistanceSq(curr_centroids[centroid_idx], p_extreme)) {
								dominates = false;
						} else { // this centroid is dominated and thus blacklisted
							blacklisted[i] = true;
						}
					}
				}
				return dominates;
		}

		template<typename XMeansMap>
		bool XMeans<XMeansMap>::KMeans::Dominates(
			const int centroid_idx,
			Tree_t* node,
			std::vector<bool> &blacklisted,
			index_t k) {
				return boost::mpl::if_ <
					boost::is_same <
					typename Tree_t::TreeSpec_t,
					typename fl::tree::MetricTree
					> ,
					BallKdNullaryMetaFunction1,
					BallKdNullaryMetaFunction2
				>::type::Do(table_, node, metric_, curr_centroids_, k, centroid_idx, blacklisted);
		} // Dominates


		template<typename XMeansMap>
		void XMeans<XMeansMap>::KMeans::KMeansBaseCase(Tree_t* node, 
				const index_t parent_idx) {
			typename Table_t::TreeIterator point_it(*table_, node);
			point_it.Reset();
			Point_t point;
			index_t point_id;
			for(index_t i = 0; i < point_it.count(); i++) {
				point_it.get(i, &point);
				point_it.get_id(i, &point_id);
				int closest_centroid = GetClosestCentroid(&point, 2);
				AssignPointToCentroid(&point, point_id, closest_centroid, parent_idx);
			}
		}

		template<typename XMeansMap>
		void XMeans<XMeansMap>::KMeans::ChildrenKMeansBaseCase(
			Tree_t* node) {
			typename Table_t::TreeIterator point_it(*table_, node);
			Point_t point;
			index_t point_id;
			CentroidPoint_t* cache_new_centroids = new_centroids_;
			index_t* cache_centroid_counts = centroid_counts_out_;
			for(index_t i = 0; i < point_it.count(); i++) {
				point_it.get(i, &point);
				point_it.get_id(i, &point_id);
				int closest_centroid = (*parent_memberships_in_)[point_id];	// get the parent centroid
				// now find closest child centroid
				curr_centroids_ = children_centroids_ + (2 * closest_centroid);
				centroid_counts_out_ = cache_centroid_counts + (2 * closest_centroid);
				int closest_child = GetClosestCentroid(&point, 2);
				new_centroids_ = cache_new_centroids + (2 * closest_centroid);
				AssignPointToCentroid(&point, point_id, closest_child, closest_centroid);
			}
			curr_centroids_ = parent_centroids_;
			new_centroids_ = cache_new_centroids;
			centroid_counts_out_ = cache_centroid_counts;
		}

		/**
		* Assigns a point to a centroid.
		*/
		template<typename XMeansMap>
		inline void XMeans<XMeansMap>::KMeans::AssignPointToCentroid(
			const Point_t* point,
			const int point_offset, 
			const int centroid_idx,
			const int parent_idx) {
				// TODO: it is probably a quick and easy optimization
				// to have a something_changed array for each parent node
				// so that local kmeans computation can be saved.
				something_changed_ = !something_changed_ ?
					memberships_out_[point_offset] != ((2*parent_idx)+centroid_idx): true;
				memberships_out_[point_offset] = ((2*parent_idx)+centroid_idx); // assign
				DEBUG_ASSERT(centroid_idx == 0 || centroid_idx == 1);
				DEBUG_ASSERT((*parent_memberships_in_)[point_offset] == parent_idx);
				centroid_counts_out_[centroid_idx] = centroid_counts_out_[centroid_idx] + 1; // update count
				fl::la::AddTo(*point, &(new_centroids_[centroid_idx].
                  template dense_point<typename CentroidPoint_t::CalcPrecision_t>()));
		}

		template<typename XMeansMap>
		void XMeans<XMeansMap>::KMeans::AssignAllPointsToCentroid(
			Tree_t* node,
			const int centroid_idx,
			const int parent_idx) {
				Point_t point;
				index_t point_id;
				typename Table_t::TreeIterator point_it(*table_, node);
				for(index_t i = 0; i < point_it.count(); i++) {
					point_it.get(i, &point);
					point_it.get_id(i, &point_id);
					AssignPointToCentroid(&point, point_id, centroid_idx, parent_idx);
				}
		}


	} // ml
} // fl

#endif
