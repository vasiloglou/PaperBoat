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

#ifndef FL_LITE_MLPACK_KMEANS_KMEANS_CV_DEV_H_
#define FL_LITE_MLPACK_KMEANS_KMEANS_CV_DEV_H_
#ifdef _OPENMP
#include <omp.h>
#endif
#include <map>
#include "fastlib/math/fl_math.h"
#include "mlpack/clustering/kmeans_cv.h"
#include "fastlib/workspace/workspace_defs.h"
#include "fastlib/math/fl_math.h"

namespace fl { namespace ml {
  template<typename KMeansType>
  KMeansCV<KMeansType>::KMeansCV() {
    metric_             = NULL;
    references_         = NULL;
    final_centroids_    = NULL;
    traversal_mode_     = "tree";
    kmin_               = 2;
    kmax_               = 10;
    percentage_holdout_ = 0.1;
    restarts_           = 10;
    epochs_             = 1;
    randomize_          = true;
    max_iterations_     = 1000; 
    probability_        = 0.8;
    init_cent_          = "random";
  }

  template<typename KMeansType>
  KMeansCV<KMeansType>::~KMeansCV() {
    for(typename std::map<index_t, CentroidTable_t*>::iterator it=centroid_tables_.begin();
      it!=centroid_tables_.end(); ++it) {
        delete it->second;
    }
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::CrossValidate(double *optimal_score) {
      if (metric_==NULL) {
        fl::logger->Die() << "You haven't set the metric";
      }    
      if (references_==NULL) {
        fl::logger->Die() << "You haven't provided references for kmeans";
      }
      std::string mode=traversal_mode_;

      boost::shared_ptr<Table_t> train_table(new Table_t());
      Table_t test_table;
      Split(*references_, percentage_holdout_, train_table.get(), &test_table);
      if(mode == "tree" || mode == "online_tree") {
		  fl::logger->Message() << "Indexing training table.";
		  typename Table_t::template IndexArgs<fl::math::LMetric<2> > q_index_args;
		  q_index_args.leaf_size = 20;
		  train_table->IndexData(q_index_args);
      }
      for(index_t k=kmin_; k<=kmax_; ++k) {
        scores_[k]=std::numeric_limits<CalcPrecision_t>::max();
        centroid_tables_[k]=new CentroidTable_t();
        for(index_t i=0; i<restarts_; ++i) {
          Kmeans_t kmeans;
          kmeans.set_max_iterations(max_iterations_);
          CentroidTable_t centroids;
          centroids.Init("", 
            std::vector<index_t>(1, references_->n_attributes()),
            std::vector<index_t>(),
            k);
          if (init_cent_=="random") {
            kmeans.AssignInitialCentroids(k, &centroids, *train_table);
          } else {
            if (init_cent_=="kmeans++") {
              fl::ws::WorkSpace ws;
              ws.set_paging_mode(0);
              ws.template LoadTable<Table_t>("train", train_table);
              std::vector<std::string> dummy_vector;
              dummy_vector.push_back("train");
              fl::math::LMetric<2> metric; 
              kmeans.template KMeansPlusPlus<fl::ws::WorkSpace, Table_t>(k,
                probability_, 
                metric,
                &ws,
                dummy_vector,
                &centroids);
            } else {
              fl::logger->Die()<<"Invalid initialization option ("
                <<init_cent_ 
                <<") for centroids"; 
            }
          }
          
          if (traversal_mode_=="online_tree" || traversal_mode_=="online_naive") {
            fl::math::LMetric<2> metric;
            fl::logger->Message()<<"Starting online kmeans"<<std::endl;
            fl::ws::WorkSpace ws;
            ws.set_paging_mode(0);
            ws.template LoadTable<Table_t>("train", train_table);
            std::vector<std::string> dummy_vector;
            dummy_vector.push_back("train");

            fl::ml::KMeansOnline(dummy_vector,
              *train_table,
              &ws,
              metric,
              randomize_, 
              epochs_, 
              &centroids);
            // we need to feed that to the kmeans engine;
            mode=traversal_mode_;
            mode.erase(0, 7);
          }
          typename CentroidTable_t::Point_t* initial_centroids = 
            new typename CentroidTable_t::Point_t[k];
          for(index_t ii = 0; ii < k; ii++) {
            typename CentroidTable_t::Point_t centroid_point;
            centroids.get(ii, &centroid_point);
            initial_centroids[ii].template dense_point<typename CentroidTable_t::CalcPrecision_t>().Copy(centroid_point);
          }
          kmeans.Init(k, train_table.get(), initial_centroids);
          delete[] initial_centroids;
          kmeans.RunKMeans(mode);
          CentroidTable_t* centroids_out = new CentroidTable_t();
          centroids_out->Init(std::vector<index_t>(1, train_table->n_attributes()),  std::vector<index_t>(), k);
          kmeans.GetCentroids(centroids_out);
          std::vector<index_t> cardinality;
          CalcPrecision_t score = ComputeClusteringObjective(*centroids_out, 
            test_table, &cardinality);
          if (scores_[k]> score) {
            scores_[k]=score;
            delete centroid_tables_[k];
            centroid_tables_[k]= centroids_out;
          } else {
            delete centroids_out;
          }
          fl::logger->Message()<<"k="<< k  <<" restart="<< i 
            <<" distortion="<< score/test_table.n_entries() <<std::endl;
        } // for each restart
        std::vector<index_t> cardinality;
        scores_[k]=ComputeKQuality(*references_, *centroid_tables_[k], &cardinality);
        fl::logger->Message() << "***** k="<<k<<" BIC="<<scores_[k]<<std::endl; 
      } // for k_min to k_max
      *optimal_score=-std::numeric_limits<CalcPrecision_t>::max();
      for(index_t k=kmin_; k<=kmax_;  ++k) {
        if (scores_[k]>*optimal_score) {
          *optimal_score=scores_[k];
          final_k_ = k;
          final_centroids_=centroid_tables_[k];
        }
      }
  }

template<typename KMeansType>
void KMeansCV<KMeansType>::GetCentroids(CentroidTable_t* centroids) {
  for (int i = 0; i < final_k_; i++) {
    typename CentroidTable_t::Point_t point, point_two;
    centroids->get(i, &point);
    final_centroids_->get(i, &point_two);
    point.CopyValues(point_two);
  }

}

template<typename KMeansType>
index_t KMeansCV<KMeansType>::GetFinalK() {
  return final_k_;
}

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_metric(const Metric_t *metric) {
    metric_=metric;  
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_references(Table_t *references) {
    references_=references;
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_traversal_mode(const std::string &traversal_mode) {
    traversal_mode_=traversal_mode;
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_kmin(index_t kmin) {
    kmin_=kmin; 
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_kmax(index_t kmax) {
    kmax_=kmax;
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_percentage_hold_out(double percentage_holdout) {
    percentage_holdout_=percentage_holdout;
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_restarts(index_t restarts) {
    restarts_=restarts;
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_epochs(index_t epochs) {
    epochs_=epochs;
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_randomize(bool randomize) {
    randomize_=randomize;
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_max_iterations(index_t max_iterations) {
    max_iterations_=max_iterations;
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_probability(double probability) {
    probability_=probability;
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::set_init_cent(const std::string &init_cent) {
    init_cent_=init_cent;
  }
  template<typename KMeansType>
  void KMeansCV<KMeansType>::Split(Table_t &references, 
    double percentage_holdout, 
    Table_t *train_table, 
    Table_t *test_table) {

      train_table->Init(references.data()->dense_sizes(),
        references.data()->sparse_sizes(),
        0);
      test_table->Init(references.data()->dense_sizes(),
        references.data()->sparse_sizes(),
        0);
      for(index_t i=0; i<references.n_entries(); ++i) {
        typename Table_t::Point_t p;
        references.get(i, &p);
        if (fl::math::Random<double>() > percentage_holdout) {
          train_table->data()->push_back(p);
        } else {
          test_table->data()->push_back(p);
        }
      }
  }

  template<typename KMeansType>
  typename KMeansCV<KMeansType>::CalcPrecision_t 
    KMeansCV<KMeansType>::ComputeClusteringObjective(
    CentroidTable_t &centroids, 
    Table_t &test,
    std::vector<index_t> *cardinalities) {
      cardinalities->resize(centroids.n_entries());
      std::fill(cardinalities->begin(), cardinalities->end(), 0);
      CalcPrecision_t objective=0;
      for(index_t i=0; i<test.n_entries(); ++i) {
        typename Table_t::Point_t point;
        test.get(i, &point);
        CalcPrecision_t best_distance=std::numeric_limits<CalcPrecision_t>::max();
        index_t best_centroid=-1;
        for(index_t k=0; k<centroids.n_entries(); ++k) {
          typename CentroidTable_t::Point_t centroid;
          centroids.get(k, &centroid);
          CalcPrecision_t dist = metric_->DistanceIneq(
              centroid.template dense_point<typename CentroidTable_t::Point_t::CalcPrecision_t>(), point);
          if (dist<best_distance) {
            best_distance=dist;
            best_centroid=k;
          }
        }
        DEBUG_ASSERT(best_distance!=std::numeric_limits<CalcPrecision_t>::max());
        objective+=best_distance;
        cardinalities->operator[](best_centroid)+=1;
      }
      return objective;
  }

  template<typename KMeansType>
  void KMeansCV<KMeansType>::ComputeClusteringObjective(
    CentroidTable_t &centroids, 
    Table_t &test,
    std::vector<index_t> *cardinalities,
    std::vector<CalcPrecision_t> *variances) {
      cardinalities->resize(centroids.n_entries());
      variances->resize(centroids.n_entries());
      std::fill(cardinalities->begin(), cardinalities->end(), 0);
      std::fill(variances->begin(), variances->end(), 0);

      for(index_t i=0; i<test.n_entries(); ++i) {
        typename Table_t::Point_t point;
        test.get(i, &point);
        CalcPrecision_t best_distance=std::numeric_limits<CalcPrecision_t>::max();
        index_t best_centroid=-1;
        for(index_t k=0; k<centroids.n_entries(); ++k) {
          typename CentroidTable_t::Point_t centroid;
          centroids.get(k, &centroid);
          CalcPrecision_t dist = metric_->DistanceIneq(
              centroid.template dense_point<typename CentroidTable_t::CalcPrecision_t>(), point);
          if (dist<best_distance) {
            best_distance=dist;
            best_centroid=k;
          }
        }
        DEBUG_ASSERT(best_distance!=std::numeric_limits<CalcPrecision_t>::max());
        variances->operator[](best_centroid)+=best_distance;
        cardinalities->operator[](best_centroid)+=1;
      }
  }


  template<typename KMeansType>
  typename KMeansCV<KMeansType>::CalcPrecision_t 
    KMeansCV<KMeansType>::ComputeKQuality(Table_t &references, 
    CentroidTable_t &centroid_table,
    std::vector<index_t> *cardinality) {

      index_t k_in = centroid_table.n_entries();

      std::vector<CalcPrecision_t> variances;
      ComputeClusteringObjective(centroid_table, 
        references, cardinality, &variances);
      CalcPrecision_t variance_all = 0;
      for(int i = 0; i < k_in; i++) {
        variance_all += variances[i];
      }
      variance_all *= ((CalcPrecision_t)1.0/(references.n_entries() - k_in));

    	CalcPrecision_t log_likeleyhood_all = 0;
		for(int i = 0; i < k_in; i++) {
			//if center owns just one point, then likelihood of point is 1 and so log-likelihood is zero 
			if((*cardinality)[i] > 1) {
				log_likeleyhood_all += (*cardinality)[i]*log((CalcPrecision_t)(*cardinality)[i])
					- (*cardinality)[i]*log((CalcPrecision_t)references.n_entries())
					- (*cardinality)[i]/2.0*log(2*fl::math::template Const<CalcPrecision_t>::PI)
					- (*cardinality)[i]*references.n_attributes()/2.0*log(variance_all)
					- variances[i]/(2*variance_all);

			}
			
		}
		// num parameters
		index_t p = (k_in - 1) + (k_in * references.n_attributes()) + k_in;
        // BIC SCORE = LogLikelyHood - ((k-1+k*d+k)*Log(n)/2)
		return (log_likeleyhood_all - (((CalcPrecision_t)p / 2.0) * log((CalcPrecision_t)references.n_entries())));
  }

}}

#endif
