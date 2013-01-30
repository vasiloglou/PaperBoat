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

#ifndef FL_LITE_MLPACK_KMEANS_KMEANS_CV_H_
#define FL_LITE_MLPACK_KMEANS_KMEANS_CV_H_
#include <string>
#include <map>
#include "boost/utility.hpp"

namespace fl { namespace ml {

  template<typename KMeansType>
  class KMeansCV : boost::noncopyable {
    public:
     typedef KMeansType Kmeans_t;
     typedef typename KMeansType::Table_t Table_t;
     typedef typename Table_t::CalcPrecision_t CalcPrecision_t;
     typedef typename Table_t::Point_t Point_t;
     typedef typename KMeansType::CentroidTable_t CentroidTable_t;
     typedef typename KMeansType::Metric_t Metric_t;

     KMeansCV();
     ~KMeansCV();
     void CrossValidate(double *optimal_score);
     
      void GetCentroids(CentroidTable_t* centroids); 

      index_t GetFinalK();

      void set_metric(const Metric_t *metric);
     void set_references(Table_t *references);
     void set_traversal_mode(const std::string &traversal_mode) ;
     void set_kmin(index_t kmin);
     void set_kmax(index_t kmax);
     void set_percentage_hold_out(double percentage_holdout);
     void set_restarts(index_t restarts);
     void set_epochs(index_t epochs);
     void set_randomize(bool randomize);
     void set_max_iterations(index_t max_iterations);
     void set_probability(double probability);
     void set_init_cent(const std::string &init_cent);

    private:
      const Metric_t *metric_;
      Table_t *references_;
      std::string traversal_mode_;
      index_t kmin_;
      index_t kmax_;
      double percentage_holdout_;
      index_t restarts_;
      std::map<index_t, CentroidTable_t *> centroid_tables_;
      CentroidTable_t* final_centroids_;
      index_t final_k_;
      std::map<index_t, CalcPrecision_t> scores_;
      index_t epochs_;
      bool randomize_;
      index_t max_iterations_;
      double probability_;
      std::string init_cent_;

      void Split(Table_t &references, 
                 double percentage_holdout, 
                 Table_t *train_table, 
                 Table_t *test_table);
      CalcPrecision_t ComputeClusteringObjective(CentroidTable_t &centroids, 
          Table_t &test, std::vector<index_t> *cardinality);
      void ComputeClusteringObjective(
          CentroidTable_t &centroids, 
          Table_t &test,
          std::vector<index_t> *cardinalities,
          std::vector<CalcPrecision_t> *variances); 
      CalcPrecision_t ComputeKQuality(Table_t &references, 
                           CentroidTable_t &centroid_table,
                           std::vector<index_t> *cardinality);

  };

}}

#endif

