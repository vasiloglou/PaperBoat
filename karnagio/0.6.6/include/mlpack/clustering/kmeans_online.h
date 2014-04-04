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

#ifndef FL_LITE_INCLUDE_MLPACK_CLUSTERING_KMEANS_ONLINE_H_
#define FL_LITE_INCLUDE_MLPACK_CLUSTERING_KMEANS_ONLINE_H_
#include<algorithm>
#include "fastlib/base/base.h"
#include "fastlib/la/linear_algebra.h"

namespace fl { namespace ml {
  class KMeansOnline {
    public:
      template<typename TableType, 
               typename WorkSpaceType,
               typename CentroidTableType,
               typename MetricType>
      KMeansOnline(const std::vector<std::string> &references,
                   TableType &t,
                   WorkSpaceType *ws,
                   const MetricType& metric, 
                   bool randomize,
                   index_t epochs, CentroidTableType *centroids) {
        index_t k_clusters=centroids->n_entries();
        std::vector<long long int> counts(k_clusters, 1);
        std::vector<int> indices;

        for(index_t e=0; e<epochs; ++e) {
          for(size_t i=0; i<references.size(); ++i) {
            boost::shared_ptr<TableType> table;
            ws->Attach(references[i], &table);
            if (randomize==true) {
               indices.resize(table->n_entries());
               for(index_t i=0; i<indices.size(); ++i) {
                 indices[i]=i;
               } 
               std::random_shuffle(indices.begin(), indices.end());
               typename TableType::Point_t point;
               typename CentroidTableType::Point_t cent;
               for(index_t i=0; i<table->n_entries(); ++i) {
                  table->get(indices[i], &point);
                  index_t best_centroid=-1;
                  double best_distance=std::numeric_limits<double>::max();
                  for(index_t k=0; k<k_clusters; ++k) {
                    centroids->get(k, &cent);
                    double distance=metric.DistanceSq(cent.template dense_point<
                      typename CentroidTableType::CalcPrecision_t>(), point);
                    if (distance<best_distance) {
                      best_distance=distance;
                      best_centroid=k;
                    }
                  }
                  DEBUG_ASSERT(best_centroid!=-1);
                  counts[best_centroid]+=1;
                  centroids->get(best_centroid, &cent);
                  double gamma=1.0/counts[best_centroid];
                  fl::la::AddExpert(gamma/(1-gamma), point, 
                     &cent.template dense_point<typename CentroidTableType::CalcPrecision_t>());
                  fl::la::SelfScale(1-gamma, &cent);
               }
               double distortion=0;
               for(index_t i=0; i<table->n_entries(); ++i) {
                  table->get(i, &point);
                  double best_distance=std::numeric_limits<double>::max();
                  for(index_t k=0; k<k_clusters; ++k) {
                    centroids->get(k, &cent);
                    double distance=metric.DistanceSq(cent.template dense_point<
                       typename CentroidTableType::CalcPrecision_t>(), point);
                    if (distance<best_distance) {
                      best_distance=distance;
                    }
                  }
                  distortion+=best_distance;
               }
               distortion/=table->n_entries();
               fl::logger->Message()<<"online epoch="<< e
                  <<", table="<<references[i]<<", distortion="<<distortion 
                  <<std::endl;
            } else {
              typename TableType::Point_t point;
              typename CentroidTableType::Point_t cent;
              for(index_t i=0; i<table->n_entries(); ++i) {
                table->get(i, &point);
                index_t best_centroid=-1;
                double best_distance=std::numeric_limits<double>::max();
                for(index_t k=0; k<k_clusters; ++k) {
                  centroids->get(k, &cent);
                  double distance=metric.DistanceSq(cent.template dense_point<
                      typename CentroidTableType::CalcPrecision_t>(), point);
                  if (distance<best_distance) {
                    best_distance=distance;
                    best_centroid=k;
                  }
                }
                counts[best_centroid]+=1;
                centroids->get(best_centroid, &cent);
                double gamma=1.0/counts[best_centroid];
                fl::la::AddExpert(gamma/(1-gamma), point, 
                    &cent.template dense_point<typename CentroidTableType::CalcPrecision_t>());
                fl::la::SelfScale(1-gamma, &cent);
              }
              double distortion=0;
              for(index_t i=0; i<table->n_entries(); ++i) {
                table->get(i, &point);
                double best_distance=std::numeric_limits<double>::max();
                for(index_t k=0; k<k_clusters; ++k) {
                  centroids->get(k, &cent);
                  double distance=metric.DistanceSq(cent.template dense_point<
                      typename CentroidTableType::CalcPrecision_t>(), point);
                  if (distance<best_distance) {
                    best_distance=distance;
                  }
                }
                distortion+=best_distance;
              }
              fl::logger->Message()<<"online epoch="<< e
              <<", table="<<references[i]<<", distortion="<<distortion 
              <<std::endl;
            }
         }
      }
    }
  };
}}
#endif
