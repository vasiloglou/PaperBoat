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
#ifndef FL_LITE_MLPACK_XMEANS_H
#define FL_LITE_MLPACK_XMEANS_H

#include "fastlib/base/base.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/math/fl_math.h"
#include "fastlib/math/gen_range.h"
#include "fastlib/tree/metric_tree.h"
#include "fastlib/table/table_vector.h"
#include "mlpack/clustering/kmeans_dev.h"
#include <map>
#include "boost/program_options.hpp"
#include "boost/mpl/if.hpp"

namespace fl {
  namespace ml {

    template <typename KMeansType>
    class XMeans {

    public:
      typedef typename KMeansType::Table_t Table_t;
      typedef typename Table_t::CalcPrecision_t CalcPrecision_t;
      typedef typename Table_t::Point_t Point_t;
      typedef typename KMeansType::CentroidTable_t CentroidTable_t;
      typedef typename KMeansType::Metric_t Metric_t;
      typedef typename Table_t::Tree_t Tree_t;
      typedef typename CentroidTable_t::Point_t CentroidPoint_t;

      /**
      * There are parent centroids and there are 2 children for each parent centroid.
      * Say there are "k" parent centroids and thus 2 "k" children centroids. Let us
      * also say that our data set has already been clustered to find "k" centroids
      * and these are the parent centroids. Thus each point in our dataset belongs to
      * one of the "k" parent centroids. Our goal in this "localized" kmeans is to do
      * the following:
      * for (each parent centroid):
      *    Let X represent all the points in the dataset 
      *    belonging to this parent centroid.
      *    Cluster all points in X between the 2 children centroids
      *    of this parent.
      *
      */ 
      class KMeans {

      public:

        KMeans();
        ~KMeans();

        void Init(
          Table_t* table_in, 
          CentroidPoint_t* parent_centroids_in, 
          CentroidPoint_t* children_centroids_in, 
          index_t* memberships_out,
          index_t* centroid_counts_out,
          fl::table::TableVector<index_t>& parent_memberships_in,
          const int k_parents_in, 
          const Metric_t* metric_in);

        void RunKMeans();

        /**
        * Copies the centroids into the passed points.
        * Memory should already be initialized.
        */
        void GetCentroids(CentroidTable_t* centroids) ;

        /**
        * Memory should already be initialized
        */
        void GetMemberships(std::vector<index_t> memberships_out);


      private:

        // returns the index of the closest centroid
        // if there is a tie returns the one with the lower index
        int GetClosestCentroid(const Point_t *point, index_t k);

        /**
        * Returns true if 1 and only 1 centroid is closest to the node.
        * It puts the index of the first of the closest centroids into
        * centroid_idx i.e. even if the function returns false indicating
        * that there are 2 centroids which share smallest minimum distances
        * with the node, centroid_idx will contain one of these centroids
        * and particularly the one with the lower index;
        */
        bool GetClosestCentroid(Tree_t* node, int& centroid_idx, index_t k);

        /**
        * This function returns true if centroid_idx dominates all other
        * non-blacklisted centroids with repspect to this node. It also
        * blacklists all centroids which were previously not blacklisted
        * but are dominated by this centroid (centroid_idx).
        */
        bool Dominates(const int centroid_idx, Tree_t* node, 
          std::vector<bool> &blacklisted, index_t k);
        /**
        * The following structs are necessary for Dominates
        * They do some sort of metaprogramming so that we can differentiate
        * between ball tree and kdtree
        */
        struct BallKdNullaryMetaFunction1 {
          // We need to templatize over TableType, so that we can get lazy evaluation
          template<typename TableType>
          static bool Do(TableType *table,
            Tree_t *node,
            const Metric_t *metric,
            CentroidPoint_t *curr_centroids,
            index_t k,
            const int centroid_idx,
            std::vector<bool> &blacklisted);
        };

        struct BallKdNullaryMetaFunction2 {
          // We need to templatize over TableType, so that we can get lazy evaluation
          template<typename TableType>
          static bool Do(TableType *table,
            Tree_t *node,
            const Metric_t *metric,
            CentroidPoint_t *curr_centroids,
            index_t k,
            const int centroid_idx,
            std::vector<bool> &blacklisted);
        };

        void KMeansBaseCase(Tree_t* node, const index_t parent_idx);

        void ChildrenKMeansBaseCase(Tree_t* node);

        // Assigns a point to a centroid.
        inline void AssignPointToCentroid(const Point_t* point,
          const int point_id, const int centroid_idx, const int parent_idx);

        void AssignAllPointsToCentroid(Tree_t* node, 
          const int centroid_idx, const int parent_idx);

        bool CheckToSeeLocalResultsAreCorrect(
          CentroidPoint_t* centroids, 
          index_t k, Table_t* table, 
          CentroidPoint_t* compare_centroids);

        void AssignUpdateStepRecursive(
          Tree_t* node, 
          std::vector<bool>& blacklisted, const index_t parent_idx);

        int k_parents_;     // the number of clusters
        CentroidPoint_t* parent_centroids_;
        CentroidPoint_t* children_centroids_;	// if not null we do improve-params step
        index_t* centroid_counts_out_;
        index_t* memberships_out_;
        fl::table::TableVector<index_t>* parent_memberships_in_;

        // internal use
        CentroidPoint_t* curr_centroids_;	// current centroids (internal)
        CentroidPoint_t* new_centroids_;	// internal use
        bool something_changed_;

        // invariant
        const Metric_t* metric_;
        Table_t *table_;
      };

      // this will return a pointer to an array of k_min_
      // centroids. Freeing the memory for these is responsibility of
      // the calling function
      CentroidPoint_t* GetInitialCentroids() {
        return initial_centroids_;
      }

      // Sometimes in kmeans some centers have no points. This function
      // will remove such centroids, update memberships with the new id's
      // and return the new value of 'k' the number of centroids which have
      // at least one point. 
      // Note: A reallocation of the centroid points is done and the old
      // centroids are deleted
      static index_t RemoveEmptyCentroids(CentroidPoint_t** centroids,
        fl::table::TableVector<index_t>* centroid_memberships,
        index_t k, index_t** centroid_counts) {
          std::vector<index_t> true_position(k,-1);
          index_t counter = 0;
          for(index_t i = 0; i < k ; i++) {
            if((*centroid_counts)[i] != 0) {
              true_position[i] = counter;
              counter++;
            }
          }
          if(counter == k) {
            // no empty clusters found
            return k;
          }
          for(index_t i = 0; i < centroid_memberships->size(); i++) {
            centroid_memberships->set(i, true_position[(*centroid_memberships)[i]]);
          }
          CentroidPoint_t* temp = *centroids;
          *centroids = new CentroidPoint_t[counter];
          index_t* temp_counts = *centroid_counts;
          *centroid_counts = new index_t[counter];
          for(index_t i = 0; i < k; i++) {
            if(true_position[i] != -1) {
              (*centroids)[true_position[i]].Copy(temp[i]);
              (*centroid_counts)[true_position[i]] = temp_counts[i];
            }
          }
          delete[] temp;
          delete[] temp_counts;
          return counter;
      }

      void Run() {
        // these will hold the final values to choose from
        std::vector<CentroidPoint_t*> models;
        std::vector<CalcPrecision_t> bic_scores;
        std::vector<index_t> k_values;
        // local values per iteration
        fl::table::TableVector<index_t> parent_memberships_out;
        parent_memberships_out.Init(table_->n_entries());
        index_t* children_memberships_out = new index_t[table_->n_entries()];
        index_t k = k_min_;
        CentroidPoint_t* centroids = GetInitialCentroids();	// these need to be initialized
        index_t macro_iterations = 0;
        do {
          macro_iterations++;
          // first - run kmeans on the centroids
          fl::logger->Message() << "K Clusters = " << k;
          KMeansType kmeans;
          kmeans.Init(k, table_, centroids);
          kmeans.RunKMeans("tree");

          // get the centroids
          std::vector<index_t> dense_dim(1, table_->n_attributes());
          std::vector<index_t> sparse_dim;
          CentroidTable_t centroid_table;
          centroid_table.Init(dense_dim, sparse_dim, k);
          kmeans.GetCentroids(&centroid_table);
          for(index_t i = 0; i < k ; i++) {
            CentroidPoint_t centroid;
            centroid_table.get(i, &centroid);
            centroids[i].CopyValues(centroid);
          }
          // get memberships
          kmeans.GetMemberships(&parent_memberships_out);
          // get counts
          fl::table::TableVector<index_t> parent_memberships_counts_out;
          parent_memberships_counts_out.Init(k);
          kmeans.GetMembershipCounts(&parent_memberships_counts_out);
          index_t* centroid_counts = new index_t[k];
          for(index_t i = 0; i < k; i++) {
            centroid_counts[i] = parent_memberships_counts_out[i];
          }
          // got final centroids, their counts and point assignments
          // remove empty centroids if any
          k = RemoveEmptyCentroids(&centroids, &parent_memberships_out, k, &centroid_counts);

          // calcluate scores for the model
          CalcPrecision_t* distortions = new CalcPrecision_t[k];
          CalcPrecision_t* centroid_bic = new CalcPrecision_t[k];
          CalcPrecision_t bic_score = GetScore(table_, metric_, centroids, k,
            parent_memberships_out, centroid_counts,
            distortions, centroid_bic);
          models.push_back(centroids);	// push the old centroids in there
          bic_scores.push_back(bic_score);
          k_values.push_back(k);
          fl::logger->Debug() << "Iteration: " << macro_iterations
            << ", BIC Score: " << bic_score
            << ", K: " << k
            << ", Total Distortion: " << kmeans.GetDistortion();

          // This is the greedy case from the previous iteration
          if(k == k_max_) {
            delete[] centroid_counts;
            delete[] distortions;
            delete[] centroid_bic;
            break;
          }
          // need to split the centroids
          std::vector<CalcPrecision_t> centroid_variances;
          for(int i = 0 ; i < k; i++) {
            centroid_variances.push_back(distortions[i]/centroid_counts[i]);
          }
          // second - split centroids - run local kmeans for each centroid
          CentroidPoint_t* split_centroids = new CentroidPoint_t[2 * k];
          SplitCentroids(k, centroids, centroid_variances, split_centroids); 
          // get localized kmeans results for children
          KMeans kmeans_children;
          index_t* children_centroid_counts = new index_t[k * 2];
          kmeans_children.Init(table_, centroids, split_centroids, children_memberships_out, 
            children_centroid_counts, parent_memberships_out, k, metric_);
          kmeans_children.RunKMeans();
          // get the localized kmeans model scores
          CalcPrecision_t* model_bic_scores = new CalcPrecision_t[k];
          GetPairwiseScore(
            split_centroids,
            k * 2,
            children_memberships_out, 
            children_centroid_counts,
            model_bic_scores);
          // check to see which parents/children to keep
          CentroidPoint_t *temp = centroids;
          index_t new_k = 0;
          for(int i = 0; i < k; i++) {
            new_k += (centroid_bic[i] < model_bic_scores[i]? 2 : 1);
          }

          // We get greedy now. We can't keep all the children. We want to 
          // keep those children which increase the score the maximum. One
          // way to do this is sort based on the difference between parent and
          // children score and keep those children with max difference.
          if(new_k > k_max_) {
            fl::logger->Debug() << "Exceeded maximum. Going for greedy approach.";
            std::multimap<CalcPrecision_t, index_t> children_parent_score_diffs;
            for(index_t i = 0; i < k; i++) {
              children_parent_score_diffs.insert(
                std::pair<CalcPrecision_t, index_t>(
                centroid_bic[i] - model_bic_scores[i], i));
            }	
            // we keep the first (max_k_ - k_) values
            typename std::multimap<CalcPrecision_t, index_t>::iterator it = children_parent_score_diffs.begin(); 
            for (index_t i = 0; i < k_max_ - k; i++) {
              it++;
            }
            // for the rest of children which are better than parents make their score -INF
            for(; it != children_parent_score_diffs.end() && (*it).first <= 0; it++) {
              model_bic_scores[(*it).second] = (-std::numeric_limits<CalcPrecision_t>::max());
            }
            new_k = k_max_;
          }
          // Check if nothing changed
          // TODO: At this point pelleg takes half the worst scoring parents and splits them
          // You may want to implement that.
          if(new_k == k) { 
            delete[] centroid_counts;
            delete[] distortions;
            delete[] centroid_bic;
            delete[] split_centroids;
            delete[] children_centroid_counts;
            delete[] model_bic_scores;
            break;
          }
          centroids = new CentroidPoint_t[new_k];
          int counter = 0;
          for(int i = 0; i < k; i++) {
            if(centroid_bic[i] < model_bic_scores[i]) {
              centroids[counter].Copy(split_centroids[2*i]);
              counter++;
              centroids[counter].Copy(split_centroids[2*i+1]);
              counter++;
            } else {
              centroids[counter].Copy(temp[i]);
              counter++;
            }
          }
          k = new_k;
          delete[] centroid_counts;
          delete[] centroid_bic;
          delete[] model_bic_scores;		// TODO, remove these deletes
          delete[] children_centroid_counts;
          delete[] split_centroids;
          delete[] distortions;
        } while(true);
        fl::logger->Debug() << "Finished. Total Macro Iterations (Outer): " << macro_iterations;
        index_t max_index = -1;
        CalcPrecision_t best_score = (-std::numeric_limits<CalcPrecision_t>::infinity());
        for(index_t i = 0; i < models.size(); i++) {
          if(bic_scores[i] > best_score) {
            max_index = i;
            best_score = bic_scores[i];
          }
        }
        for(index_t i = 0; i < models.size(); i++) {
          if(i != max_index) {
            delete[] models[i];
          } else {
            final_centroids_= models[i];
            final_k_ = k_values[i];
            final_score_ = bic_scores[i];
          }
        }
        fl::logger->Debug() << "Final K is " << final_k_ << " with score " << final_score_ << ".";
        delete[] children_memberships_out;
      }

    public:

      index_t GetFinalK(){
        return final_k_;
      }

      void GetCentroids(CentroidTable_t* centroids) {
        for (int i = 0; i < final_k_; i++) {
          CentroidPoint_t point;
          centroids->get(i, &point);
          point.CopyValues(final_centroids_[i]);
        }
      }


      // For each of the centroids create 2 children centroid. This is
      // done by taking the radius of a centroid, choosing a random vector
      // and creating 2 points at a distance of radius/2 each from the 
      // centroid along this vector.
      // 
      //  Note: This is not uniformly random over the sphere and tends
      //  to clump a little at the poles.
      //  See: http://mathworld.wolfram.com/SpherePointPicking.html
      //  but is the same way as in weka. A better method is to sample
      //  from Gaussian Distribution. Needs to be implemented.
      void SplitCentroids(
        index_t k,
        CentroidPoint_t* centroids,
        std::vector<CalcPrecision_t>& centroid_variance,
        CentroidPoint_t* child_centroids) {
          CentroidPoint_t random_vector;
          random_vector.Copy(centroids[0]);	// initialize memory
          for(int i = 0; i < k; i++) {
            child_centroids[2 * i].Copy(centroids[i]);
            child_centroids[2 * i + 1].Copy(centroids[i]);
            for(int j = 0; j < table_->n_attributes(); j++) {
              random_vector.set(j, fl::math::Random<CalcPrecision_t>());
            }
            CalcPrecision_t sigma = fl::math::Pow<CalcPrecision_t, 1,2>(centroid_variance[i]);
            CalcPrecision_t norm = fl::la::LengthEuclidean(random_vector);
            fl::la::SelfScale((sigma / norm), &random_vector);
            fl::la::AddTo(random_vector, &child_centroids[2 * i]);
            fl::la::SelfScale(-1, &random_vector);
            fl::la::AddTo(random_vector, &child_centroids[2 * i + 1]);
          }
      }

      template<typename ContainerType>
      void GetMemberships(ContainerType* memberships_out) {
        // TODO: this should be cached. recalculating for now.
        index_t size = memberships_out->n_entries();
        typename ContainerType::Point_t mpoint;
        for(index_t i=0; i < size; ++i) {
          Point_t point;
          table_->get(i, &point); 
          memberships_out->get(i, &mpoint);
          mpoint.set(0, GetClosestCentroid(&point, final_k_, final_centroids_, metric_));
        }
      }

      static int GetClosestCentroid(const Point_t *point, index_t k, CentroidPoint_t* centroids, Metric_t* metric) {
        CalcPrecision_t min_distance = metric->DistanceSq(centroids[0].
            template dense_point<typename CentroidPoint_t::CalcPrecision_t>(), *point);
        int centroid_idx = 0;
        for (int i = 1; i < k; i++) {
          // get the min distance from node to centroid
          CalcPrecision_t distance = metric->DistanceSq(
              centroids[i].template 
              dense_point<typename CentroidPoint_t::CalcPrecision_t>(), *point);
          if (distance < min_distance) {
            min_distance = distance;
            centroid_idx = i;
          }
        }
        return centroid_idx;
      }

      ///////////////  The following emulates pellegs code

      // This function will return the BIC score for all the centroids together
      // and thus for all the data as well as BIC score for each centroid indi-
      // vidually.
      static CalcPrecision_t GetScore(
        Table_t* table_,
        Metric_t* metric_,
        CentroidPoint_t* centroids_in,
        index_t k_in,
        fl::table::TableVector<index_t> &memberships_in, 
        index_t* counts_in,
        CalcPrecision_t* distortions_out,
        CalcPrecision_t* bic_scores_out) {
          // calculate distortions
          for(int i = 0; i < k_in; i++) { 
            distortions_out[i] = 0; 
          }
          Point_t point;
          for(int i = 0 ; i < table_->n_entries(); i++) {
            table_->get(i, &point);
            index_t centroid_idx = memberships_in[i];
            distortions_out[centroid_idx] += metric_->DistanceSq(
                centroids_in[centroid_idx].template 
                dense_point<typename CentroidPoint_t::CalcPrecision_t>(), point);
          }
          CalcPrecision_t distortion_all = 0;
          std::vector<CalcPrecision_t> individual_variances(k_in);
          for(int i = 0 ; i < k_in; i++) {
            distortion_all += distortions_out[i];
            individual_variances[i] = distortions_out[i] * ((CalcPrecision_t)1.0/(counts_in[i] - 1));
          }
          CalcPrecision_t variance_all = (distortion_all / ((CalcPrecision_t)table_->n_entries() - k_in));
          // now get the overall BIC score and the score for each centroid
          std::vector<CalcPrecision_t> individual_log_likelyhoods(k_in, 0);
          CalcPrecision_t log_likeleyhood_all = 0;
          for(int i = 0; i < k_in; i++) {
            // if center owns just one point, then likelihood of point is 1 and so log-likelihood is zero 
            // The same holds if the center owns more than 1 point which are identical
            // in both cases distortion should be zero
            if(distortions_out[i] > 0) {
              log_likeleyhood_all += counts_in[i]*log((CalcPrecision_t)counts_in[i])
                - counts_in[i]*log((CalcPrecision_t)table_->n_entries())
                - counts_in[i]/2.0*log(2*fl::math::template Const<CalcPrecision_t>::PI)
                - counts_in[i]*table_->n_attributes()/2.0*log(variance_all)
                - distortions_out[i]/(2*variance_all);

              individual_log_likelyhoods[i] = 
                // the following 2 cancel out
                //counts_in[i]*log((CalcPrecision_t)counts_in[i])
                //- counts_in[i]*log((CalcPrecision_t)counts_in[i])
                - counts_in[i]/2.0*log(2*fl::math::template Const<CalcPrecision_t>::PI)
                - counts_in[i]*table_->n_attributes()/2.0*log(individual_variances[i])
                - distortions_out[i]/(2*individual_variances[i]);
            }
            // the following handles an edge case. It is similar when the centroid has 1 point
            // this is the case for a centroid have 1 point OR many points all of which are the 
            // same. Since the distortion is 0 the likelyhood is 1 and log like. is 0
            // BIC SCORE = LogLikelyHood - ((k-1+k*d+k)*Log(n)/2)
            // k = 1 in this case for single centroid and n is number of points owned by single centroid
            bic_scores_out[i] = individual_log_likelyhoods[i] - 
              (((table_->n_attributes() + 1)/2.0) * log((CalcPrecision_t)counts_in[i]));
          }
          // num parameters
          index_t p = (k_in - 1) + (k_in * table_->n_attributes()) + k_in;
          return (log_likeleyhood_all - (((CalcPrecision_t)p / 2.0) * log((CalcPrecision_t)table_->n_entries())));
      }


      // This function returns a BIC score for each model.
      // A model consists of 2 centroids, which form the children of 1
      // centroid that was split previously. 
      // centroids_in = k_in centroids where index 0,1 represent 1 model
      // 2,3 represent the next model and so on. thus there are k_in/2 models
      // memberships_in = size(n), has numbers from 0 to k_in
      // and represents for each point which of the centroids_in it belongs to
      // counts_in = size(k_in) for each centroid how many points belong to it
      // bic_scores_out = size(k_in/2) for each model the final bic score
      void GetPairwiseScore(
        CentroidPoint_t* centroids_in,
        index_t k_in,
        index_t* memberships_in, 
        index_t* counts_in,
        CalcPrecision_t* bic_scores_out) { 
          // calculate distortion for each centroid
          std::vector<CalcPrecision_t> centroid_distortions(k_in, 0);
          Point_t point;
          for(int i = 0 ; i < table_->n_entries(); i++) {
            table_->get(i, &point);
            index_t centroid_idx = memberships_in[i];
            centroid_distortions[centroid_idx] += metric_->DistanceSq(
                centroids_in[centroid_idx].
                template dense_point<typename CentroidPoint_t::CalcPrecision_t>(), point);
          }
          std::vector<CalcPrecision_t> model_variances((k_in/2), 0);
          for(int i = 0 ; i < (k_in/2); i++) {
            model_variances[i] = (centroid_distortions[2*i] + centroid_distortions[2*i+1])
              * ((CalcPrecision_t)1.0 / (counts_in[2*i] + counts_in[2*i+1] - 2));
          }
          // now get the overall BIC score and the score for each centroid
          std::vector<CalcPrecision_t> centroid_loglikelyhood(k_in, 0);
          for(int i = 0; i < (k_in/2); i++) {
            //if center owns just one point, then likelihood of point is 1
            //and so log-likelihood is zero 
            //if(counts_in[2*i] > 1) {
            if(centroid_distortions[2*i] > 0) {
              centroid_loglikelyhood[2*i] = counts_in[2*i]*log((CalcPrecision_t)counts_in[2*i])
                - counts_in[2*i]*log((CalcPrecision_t)(counts_in[2*i] + counts_in[2*i+1]))
                - counts_in[2*i]/2.0*log(2*fl::math::template Const<CalcPrecision_t>::PI)
                - counts_in[2*i]*table_->n_attributes()/2.0*log(model_variances[i])
                - centroid_distortions[2*i]/(2*model_variances[i]);
            }
            //if(counts_in[2*i+1] > 1) {
            if(centroid_distortions[2*i+1] > 0) {
              centroid_loglikelyhood[2*i+1] = counts_in[2*i+1]*log((CalcPrecision_t)counts_in[2*i+1])
                - counts_in[2*i+1]*log((CalcPrecision_t)counts_in[2*i] + counts_in[2*i+1])
                - counts_in[2*i+1]/2.0*log(2*fl::math::template Const<CalcPrecision_t>::PI)
                - counts_in[2*i+1]*table_->n_attributes()/2.0*log((CalcPrecision_t)model_variances[i])
                - centroid_distortions[2*i+1]/(2*model_variances[i]);
            }
            bic_scores_out[i] = centroid_loglikelyhood[2*i] + centroid_loglikelyhood[2*i+1];
          }
          index_t p = (2 - 1) + (2 * table_->n_attributes()) + 2;
          for(int i = 0 ; i < (k_in/2); i++) {
            // if split leads to 1 centroid having both points than it is same as parent
            bic_scores_out[i] = bic_scores_out[i] - ((p/2.0) * log((CalcPrecision_t)(counts_in[i*2] + counts_in[i*2+1])));
          }
      }       
    public:

      void Init(
        Table_t* table_in, 
        const int k_min_in, 
        const int k_max_in,
        CentroidTable_t* initial_centroids_in) {
          DEBUG_ASSERT(k_min_in == initial_centroids_in->n_entries());
          DEBUG_ASSERT(k_min_in < k_max_in);
          DEBUG_ASSERT(k_min_in > 1);
          DEBUG_ASSERT(k_max_in > 1);
          table_ = table_in;
          k_min_ = k_min_in;
          k_max_ = k_max_in;
          initial_centroids_ = new CentroidPoint_t[k_min_];
          for(int i = 0; i < k_min_; i++) {
            typename CentroidTable_t::Point_t point;
            initial_centroids_in->get(i, &point);
            initial_centroids_[i].template
            dense_point<typename CentroidPoint_t::CalcPrecision_t>().Copy(point);
          }
      }

      XMeans() {
        metric_ = new Metric_t();
        final_centroids_ = NULL;
        initial_centroids_ = NULL;	
      }

      ~XMeans() {
        delete metric_;
        delete[] final_centroids_;
      }
  
      const Metric_t &metric() {
        return *metric_;
      }
    private:

      index_t k_min_, k_max_;	// maximum and minimum number of clusters
      std::vector<index_t> model_k_;
      std::vector<CentroidPoint_t*> model_centroids_;
      std::vector<CalcPrecision_t> model_score_;		// bic/aic etc (to be maximized)
      Metric_t* metric_;
      Table_t* table_;

      // cache final model
      CentroidPoint_t* final_centroids_;
      index_t final_k_;
      CalcPrecision_t final_score_;
      CentroidPoint_t * initial_centroids_;

    };
  }}
#endif
