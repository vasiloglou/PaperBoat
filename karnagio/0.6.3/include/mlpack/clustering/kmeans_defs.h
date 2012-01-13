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
#ifndef FL_LITE_MLPACK_CLUSTERING_KMEANS_DEFS_H
#define FL_LITE_MLPACK_CLUSTERING_KMEANS_DEFS_H

#include <string>
#include "fastlib/table/branch_on_table.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "fastlib/base/base.h"
#include "kmeans.h"
#include "mlpack/xmeans/xmeans.h"
#include "kmeans_cv.h"
#include "kmeans_online.h"
#include "fastlib/util/timer.h"
#include "fastlib/table/default/dense/unlabeled/balltree/table.h"
#include "fastlib/workspace/task.h"

namespace fl {
  namespace ml {

  template<typename TableType>
  template<typename DataAccessType, typename KMeansType>
  void KMeans<boost::mpl::void_>::Core<TableType>::AttachResults(
    DataAccessType *data, KMeansType* kmeans,  TableType& table,
    std::string centroids_out, std::string memberships_out, std::string distortions_out, index_t k) {

    boost::shared_ptr<typename KMeansType::CentroidTable_t> centroids;
    std::string final_centroids_out;
    if (centroids_out != "") {
      fl::logger->Message() << "Emitting cluster centroids to " << centroids_out;
      final_centroids_out=centroids_out;
    } else {
      final_centroids_out=data->GiveTempVarName();
    }
    std::vector<index_t> dense_dim(1, table.n_attributes());
    std::vector<index_t> sparse_dim;
    data->Attach(final_centroids_out, dense_dim, sparse_dim, k, &centroids);
    kmeans->GetCentroids(centroids.get());
    data->Purge(final_centroids_out);
    data->Detach(final_centroids_out);
    
    boost::shared_ptr<typename DataAccessType::UIntegerTable_t > memberships; 
    std::string final_memberships_out;
    if (memberships_out != "") {
      fl::logger->Message() << "Emitting cluster memberships to " << memberships_out;
      final_memberships_out=memberships_out;
    } else {
      final_memberships_out=data->GiveTempVarName();
    }
    data->Attach(final_memberships_out, 
        std::vector<index_t>(1,1),
        std::vector<index_t>(),
        table.n_entries(),
        &memberships);   
          
    kmeans->GetMemberships(memberships.get());
    data->Purge(final_memberships_out);
    data->Detach(final_memberships_out);
 

    if (distortions_out!="") {
      boost::shared_ptr<typename DataAccessType::DefaultTable_t> distortions;
      fl::logger->Message()<<"Emitting centroid distortions to "<< distortions_out<<std::endl; 
      data->Attach(distortions_out, 
        std::vector<index_t>(1,1),
        std::vector<index_t>(),
        k,
        &distortions); 
      typename TableType::Point_t point;
      typename KMeansType::CentroidTable_t::Point_t cent;
      typename DataAccessType::UIntegerTable_t::Point_t  mpoint;

      for(index_t i=0; i<table.n_entries(); ++i) {
        table.get(i, &point);
        memberships->get(i, &mpoint);
        int best_centroid=mpoint[0];
        centroids->get(best_centroid, &cent);
        double distance_square_out=kmeans->metric().DistanceSq(cent.
            template dense_point<double>(), point);
        typename DataAccessType::DefaultTable_t::Point_t dpoint;
        distortions->get(best_centroid, &dpoint);
        dpoint.set(0, dpoint[0]+distance_square_out);
      }
      data->Purge(distortions_out);
      data->Detach(distortions_out);
    } 
  }


  template<typename TableType>
  template<class DataAccessType>
  int KMeans<boost::mpl::void_>::Core<TableType>::Main(
    DataAccessType *data,
    boost::program_options::variables_map &vm) {
    FL_SCOPED_LOG(kmeans);
    typedef typename fl::ml::KMeans<KMeansArgs<fl::math::LMetric<2>, 
            typename DataAccessType::DefaultTable_t> >::CentroidTable_t CentroidTable_t;
    std::string memberships_out;
    std::string distortions_out;
    std::string centroids_out;
    std::string references_in;
    std::string queries_in="";
    std::string run_mode;
    index_t k_clusters=0;
    index_t k_min=0;
    index_t k_max=0;
    double percentage_hold_out=0;
    index_t n_restarts=0;
    std::string metric_arg;
    std::string metric_weights_in;
    std::string algorithm;
    std::string centroids_in;
    std::string search_method;
    index_t leaf_size=0;
    index_t iterations=0;
    index_t epochs=0;
    bool randomize=true;
    double probability=0;
        double min_cluster_movement_threshold;
    std::string initialization;

    memberships_out = vm["memberships_out"].as<std::string>();
    distortions_out = vm["distortions_out"].as<std::string>();
    centroids_out = vm["centroids_out"].as<std::string>();
                if (vm.count("references_in")) {
                  references_in = vm["references_in"].as<std::string>();
                }
    if (vm.count("queries_in")) {
      queries_in=vm["queries_in"].as<std::string>();
    }
    k_clusters = vm["k_clusters"].as<index_t>();
    k_min = vm["k_min"].as<index_t>();
    k_max = vm["k_max"].as<index_t>();
    percentage_hold_out = vm["percentage_hold_out"].as<double>();
    n_restarts = vm["n_restarts"].as<index_t>();
    metric_arg = vm["metric"].as<std::string>();
    metric_weights_in = vm["metric_weights_in"].as<std::string>();
    algorithm = vm["algorithm"].as<std::string>();
    centroids_in = vm["centroids_in"].as<std::string>();
    leaf_size = vm["leaf_size"].as<index_t>();
    iterations = vm["iterations"].as<index_t>();
    epochs = vm["epochs"].as<index_t>();
    randomize = vm["randomize"].as<bool>();
    probability = vm["probability"].as<double>();
        min_cluster_movement_threshold = vm["minimum_cluster_movement_threshold"].as<double>();
    initialization = vm["initialization"].as<std::string>();
    run_mode = vm["run_mode"].as<std::string>();
    search_method = vm["search_method"].as<std::string>();

    if (memberships_out == "") {
      fl::logger->Warning() << "No --memberships_out argument. Cluster memberships will not be output";
    }
    if (distortions_out == "") {
      fl::logger->Warning() << "No --distortions_out argument. Cluster distortions will not be output";
    }
    if (centroids_out == "") {
      fl::logger->Warning() << "No --centroids_out argument. Cluster centroids will not be output";
    }

    if(centroids_in != "" && n_restarts > 1) {
      fl::logger->Die() << "You have provided both a starting centroids file and greater "
      "than 1 restarts. If starting centroids are provided each restart will find the "
      " same centroids. It is futile.";
    }

    if(initialization != "kmeans++" && initialization != "random") {
      fl::logger->Die() << "Unknown initialization option. Please refer --help.";
    }
    if(search_method != "xmeans" && search_method != "cv") {
      fl::logger->Die() << "Unknown search method for optimal k. Please refer --help.";
    }


    if (run_mode=="train") {
      if (references_in == "") {
        fl::logger->Die() << "Missing required --references_in";
        return 1;
      }
      std::string not_happy_with_k_error_string = 
      "You must provide either the exact number of clusters you want in"
      " argument --k_clusters or provide a value for both --k_min (the minimum) and --k_max"
      " (the maximum) number of clusters you want.";
      if(k_clusters <= 1) {
        if(k_min <= 1) {
          fl::logger->Die() << not_happy_with_k_error_string;
        }
        if(k_max <= 1) {
          fl::logger->Die() << not_happy_with_k_error_string;
        }
        if(k_max <= k_min) {
          fl::logger->Die() << "'k_max' must be greater than 'k_min'";
        }
      } else {
        if(k_min != -1 || k_max != -1) {
          fl::logger->Die() << not_happy_with_k_error_string;
        }
      }

      if (algorithm=="online_tree" || algorithm=="online_naive") {
        if (randomize==true) {
          fl::logger->Message()<<"Randomization of data requested, this will "
          "improve stability"<<std::endl;
        } else {
          fl::logger->Warning()<<"No randomization of data requested. If your "
          "data are not in random order in the input, online might not have "
          "significant progress"<<std::endl;
        }
      }

      fl::logger->Message() << "Loading data from "<<references_in<<std::endl;
      fl::util::Timer timer;
      timer.Start();
      boost::shared_ptr<TableType> table;
      data->Attach(references_in, &table);
      timer.End();
      fl::logger->Debug() << "Time taken to read data: " << timer.GetTotalElapsedTimeString().c_str();
      if (metric_arg == "l2") {
        fl::logger->Message() << "Using the l2 metric."<<std::endl;
        fl::logger->Message() << "Kmeans algorithm is " << algorithm<<std::endl;
        //if (algorithm=="tree" || algorithm=="online_tree") {
          //fl::logger->Message() << "Indexing reference data"<<std::endl;
          //timer.Start();
          //typename TableType::template IndexArgs<fl::math::LMetric<2> > q_index_args;
          //q_index_args.leaf_size = leaf_size;
          //table.IndexData(q_index_args);
          //timer.End();
          //fl::logger->Debug() << "Time taken to index data: " << timer.GetTotalElapsedTimeString().c_str();
        //}
        timer.Start();
        // if initial centroids are provided we'll read them here rather than
        // in a loop inside. The fact that it makes no sense to do restarts with
        // the same centroids (for standard kmeans) should have already been
        // handled.
        boost::shared_ptr<CentroidTable_t> initial_centroid_table; 
        if(centroids_in != "") {
          fl::logger->Message() << "Reading initial centroids from data source: " << centroids_in;
          data->Attach(centroids_in, &initial_centroid_table);
          if (initial_centroid_table->n_entries()!=(k_clusters>1?k_clusters:k_min)) {
            fl::logger->Warning()<<"The file ("<< centroids_in<< ") contains "
            << initial_centroid_table->n_entries()
            << " clusters while you requested kmeans for "
            << (k_clusters>1?k_clusters:k_min) << " clusters.";
            fl::logger->Warning() << "Ignoring " << (k_clusters>1?"k_clusters":"k_min") << " input and setting k_clusters to " << initial_centroid_table->n_entries();
            if(k_clusters > 1) {
              k_clusters = initial_centroid_table->n_entries();
            } else {
              k_min = initial_centroid_table->n_entries();
              if(k_min > k_max){ 
                fl::logger->Die() << "After reading starting centroids file k_min > k_max. Please give higher k_max value.";
              }
            }
          }
        }
  
        if (k_clusters > 1) {
          typename TableType::CalcPrecision_t min_distortion = std::numeric_limits<typename TableType::CalcPrecision_t>::max();
          fl::ml::KMeans<KMeansArgs<fl::math::LMetric<2>, 
          typename DataAccessType::DefaultTable_t> > *kmeans_final=NULL;
          for(index_t i = 0 ; i < n_restarts; i++) {
            // GET INITIAL CENTROIDS
            if(centroids_in == "") {
              initial_centroid_table.reset(new CentroidTable_t()); 
              initial_centroid_table->Init("dummy",
              table->data()->dense_sizes(),
              table->data()->sparse_sizes(),
              k_clusters);
              if (initialization=="random") {
                fl::logger->Message()<<"Initialization of the centroids with "
                  "random choice"<<std::endl;
                fl::ml::KMeans<KMeansArgs<fl::math::LMetric<2>, 
                  typename DataAccessType::DefaultTable_t> >::AssignInitialCentroids(k_clusters, 
                  initial_centroid_table.get(), *table);
                fl::logger->Message()<<"Initialization done"<<std::endl;
              } else {
                if (initialization=="kmeans++") {
                  fl::logger->Message()<<"Initialization of centroids with kmeans++"
                      <<std::endl;
                  fl::math::LMetric<2> metric; 
                  fl::ml::KMeans<KMeansArgs<fl::math::LMetric<2>, 
                    typename DataAccessType::DefaultTable_t> >::KMeansPlusPlus(k_clusters,
                    probability, 
                    metric,
                    *table,
                    initial_centroid_table.get());
                }
              }
            } else {
              // no need to read again and again, already read once
            }
            // RUN KMEANS
            fl::ml::KMeans<KMeansArgs<fl::math::LMetric<2>, 
            typename DataAccessType::DefaultTable_t> > *kmeans = 
            new fl::ml::KMeans<KMeansArgs<fl::math::LMetric<2>, 
            typename DataAccessType::DefaultTable_t> > ();
            kmeans->set_max_iterations(iterations);   
                  kmeans->set_min_cluster_movement_threshold(min_cluster_movement_threshold);
            std::string traversal=algorithm;
            if (algorithm=="online_tree" || algorithm=="online_naive") {
              fl::math::LMetric<2> metric;
              fl::ml::KMeansOnline(*table,
              metric,
              randomize, 
              epochs, 
              initial_centroid_table.get());
              // we need to feed that to the kmeans engine;
              traversal=algorithm; 
              traversal.erase(0, 7);
            }
            typename DataAccessType::DefaultTable_t::Point_t* initial_centroids = 
            new typename DataAccessType::DefaultTable_t::Point_t[k_clusters];
            // we should really get rid of this and do the initialization
            // directly from the table;
            for(index_t k = 0; k < k_clusters; ++k) {
              typename CentroidTable_t::Point_t centroid_point;
              initial_centroid_table->get(k, &centroid_point);
              std::vector<index_t> sizes(1, centroid_point.size());
              initial_centroids[k].Init(sizes);
              initial_centroids[k]. template
              dense_point<typename DataAccessType::
              DefaultTable_t::Point_t::CalcPrecision_t>().CopyValues(centroid_point);
            }
            kmeans->Init(k_clusters, table.get(), initial_centroids);
            delete[] initial_centroids;
  
            index_t total_iterations = kmeans->RunKMeans(traversal);
            typename TableType::CalcPrecision_t distortion = kmeans->GetDistortion();
            if(distortion < min_distortion) {
              min_distortion = distortion;
              kmeans_final = kmeans;
            } else {
              delete kmeans;
            }
            fl::logger->Message() << "***** batch restart="<<i
              <<", distortion="<<distortion
              <<", iterations="<<total_iterations<<std::endl;
          }
          fl::logger->Message() << "Lowest Distortion found=" << kmeans_final->GetDistortion();
  
          timer.End();
          AttachResults(data, kmeans_final, *table, centroids_out, memberships_out, distortions_out, k_clusters);
          delete kmeans_final;
        } else {  
          if(centroids_in == "") {
            initial_centroid_table.reset(new CentroidTable_t()); 
            initial_centroid_table->Init("dummy",
            table->data()->dense_sizes(),
            table->data()->sparse_sizes(),
            k_min);
            if (initialization=="random") {
              fl::logger->Message()<<"Initialization of the centroids with "
              "random choice"<<std::endl;
              fl::ml::KMeans<KMeansArgs<fl::math::LMetric<2>, 
                typename DataAccessType::DefaultTable_t> >::AssignInitialCentroids(k_min, 
                    initial_centroid_table.get(), *table);
            } else {
              if (initialization=="kmeans++") {
                fl::logger->Message()<<"Initialization of centroids with kmeans++"
                   <<std::endl;
                fl::math::LMetric<2> metric; 
                fl::ml::KMeans<KMeansArgs<fl::math::LMetric<2>, 
                typename DataAccessType::DefaultTable_t> >::KMeansPlusPlus(k_min,
                  probability, 
                  metric,
                  *table,
                  initial_centroid_table.get());
              }
            }
          } else {
            // no need to read again and again, already read once
          }
  
          if(search_method == "xmeans") {
            fl::ml::XMeans<fl::ml::KMeans<
              KMeansArgs<
              fl::math::LMetric<2>, 
              typename DataAccessType::DefaultTable_t
              >
            > 
            > xmeans;
            xmeans.Init(table.get(), k_min, k_max, initial_centroid_table.get());
            xmeans.Run();
            AttachResults(data, &xmeans, *table, centroids_out, memberships_out, distortions_out, xmeans.GetFinalK());
          } else { 
            if(search_method == "cv") {
              //cross validate for optimal K
              fl::ml::KMeansCV<
                fl::ml::KMeans<
                KMeansArgs<
                fl::math::LMetric<2>, 
                typename DataAccessType::DefaultTable_t
                >
                > 
              > kmeans;
              fl::math::LMetric<2> metric;
              kmeans.set_metric(&metric);
              kmeans.set_references(table.get());
              kmeans.set_traversal_mode(algorithm) ;
              kmeans.set_kmin(k_min);
              kmeans.set_kmax(k_max);
              kmeans.set_percentage_hold_out(percentage_hold_out);
              kmeans.set_restarts(n_restarts);
              kmeans.set_randomize(randomize);
              kmeans.set_epochs(epochs);
              kmeans.set_max_iterations(iterations);
              kmeans.set_probability(probability);
              kmeans.set_init_cent(initialization);
      
              double optimal_score=0;
              kmeans.CrossValidate(&optimal_score);
              fl::logger->Message() << "optimal k=" << kmeans.GetFinalK()
                << ", optimal BIC score=" << optimal_score << std::endl;
              timer.End();
            
              if (centroids_out != "") {
                fl::logger->Message() << "Emitting cluster centroids to " << centroids_out;
                std::vector<index_t> dense_dim(1, table->n_attributes());
                std::vector<index_t> sparse_dim;
                boost::shared_ptr<CentroidTable_t> centroids;
              
                data->Attach(centroids_out, dense_dim, sparse_dim, kmeans.GetFinalK(), &centroids);
                        kmeans.GetCentroids(centroids.get());
                data->Purge(centroids_out);
                data->Detach(centroids_out);
              }
              if (memberships_out != "" || distortions_out !="") {
                fl::logger->Message() << "Emitting cluster memberships to " << memberships_out;
                boost::shared_ptr<typename DataAccessType::UIntegerTable_t > memberships;
                if (memberships_out!="") {
                  data->Attach(memberships_out, 
                    std::vector<index_t>(1,1),
                    std::vector<index_t>(),
                    table->n_entries(),
                    &memberships);  
                }            
                boost::shared_ptr<typename DataAccessType::DefaultTable_t> distortions;
                if (distortions_out!="") {
                  data->Attach(distortions_out, 
                    std::vector<index_t>(1,1),
                    std::vector<index_t>(),
                    table->n_entries(),
                    &distortions);  
                }/*
                typename TableType::Point_t point;
                typename DataAccessType::DefaultTable_t::Point_t cent;
                fl::math::LMetric<2> metric;
                for(index_t i=0; i<table.n_entries(); ++i) {
                table.get(i, &point);
                index_t best_centroid=-1;
                double best_distance=std::numeric_limits<double>::max();
                for(int k=0; k<centroids.n_entries(); ++k) {
                  centroids.get(k, &cent);
                  double distance=metric.DistanceSq(point, cent);
                  if (distance<best_distance) {
                  best_distance=distance;
                  best_centroid=k;
                  }    
                }
                if (memberships_out!="") {
                  memberships.set(i, best_centroid);
                }
                if (distortions_out!="") {
                  distortions.set(i, best_distance);
                }
                }
                if (memberships_out!="") {
                data->Purge(memberships);
                data->Detach(memberships);
                }
                if (distortions_out!="") {
                data->Purge(distortions);
                data->Detach(distortions);
                }*/
              }
            } else {
              fl::logger->Die() << "Unknown search method to find optimal k. Please use --help. ";
            }
          }
        }
        fl::logger->Debug() << "Time taken to compute clusters: " << timer.GetTotalElapsedTimeString().c_str();
        return 0;
      } else {
        if (metric_arg == "weighted_l2") {
          fl::logger->Die() << "Weighted L2 metric " << fl::NOT_SUPPORTED_MESSAGE;
          if (algorithm == "tree") {

          }

          if (algorithm == "naive") {

          }
        } else {
          fl::logger->Die() << "Unknown metric " << metric_arg;
        }
      }
    } else {
      if (run_mode=="eval") {
        if (centroids_in=="") {
          fl::logger->Die() << "You didn't specify an input for the current centroids";
        }
  
        if (memberships_out=="") {
          fl::logger->Die() << "You didn't specify an output for the cluster memberships";
        }
  
        if (queries_in=="") {
          fl::logger->Die() << "You didn't specify an input for the query points";
        }
  
        fl::logger->Message()<<"Reading the centroids from " << centroids_in<<std::endl;
        boost::shared_ptr<typename DataAccessType::DefaultTable_t> initial_centroid_table; 
        data->Attach(centroids_in, &initial_centroid_table);
        k_clusters=initial_centroid_table->n_entries();
        fl::logger->Message()<<"Reading the queries from " << queries_in<<std::endl;
        boost::shared_ptr<TableType> table;
        data->Attach(queries_in, &table);
        typename DataAccessType::DefaultTable_t::Point_t* initial_centroids = 
          new typename DataAccessType::DefaultTable_t::Point_t[k_clusters];
        for(index_t k = 0; k < k_clusters; ++k) {
          typename DataAccessType::DefaultTable_t::Point_t centroid_point;
          initial_centroid_table->get(k, &centroid_point);
          initial_centroids[k].Copy(centroid_point);
        }
        fl::ml::KMeans<KMeansArgs<fl::math::LMetric<2>, 
          typename DataAccessType::DefaultTable_t> > kmeans;
        kmeans.Init(k_clusters, table.get(), initial_centroids);
        boost::shared_ptr<typename DataAccessType::UIntegerTable_t> memberships;
        boost::shared_ptr<typename DataAccessType::DefaultTable_t> distortions;
        data->Attach(memberships_out, 
          std::vector<index_t>(1,1),
          std::vector<index_t>(),
          table->n_entries(),
          &memberships);                       
        data->Attach(distortions_out, 
          std::vector<index_t>(1,1),
          std::vector<index_t>(),
          k_clusters,
          &distortions); 
        typename TableType::Point_t point;
        typename DataAccessType::DefaultTable_t::Point_t cent;
        for(index_t i=0; i<table->n_entries(); ++i) {
          table->get(i, &point);
          double distance_square_out=0;
          int best_centroid=
          kmeans.GetClosestCentroid(&point, distance_square_out);
          typename DataAccessType::UIntegerTable_t::Point_t mpoint;
          memberships->get(i, &mpoint);
          mpoint.set(0, best_centroid);
          typename DataAccessType::DefaultTable_t::Point_t dpoint;
          distortions->get(best_centroid, &dpoint);
          dpoint.set(0, dpoint[0]+distance_square_out);
        }
  
        fl::logger->Message() << "Emitting cluster memberships to " << memberships_out;
        data->Purge(memberships_out);
        data->Detach(memberships_out);
  
        fl::logger->Message() << "Emitting point distortions to " << distortions_out;
        data->Purge(distortions_out);
        data->Detach(distortions_out);
      } else {
        fl::logger->Die() << "--run_mode can only be train or eval";
      } 
    }
    return 1;
  }

  template<typename DataAccessType, typename BranchType>
  int KMeans<boost::mpl::void_>::Main(
    DataAccessType *data,
    const std::vector<std::string> &args
    ) {
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
      )(
      "references_in",
      boost::program_options::value<std::string>(),
      "REQUIRED in the --run_mode=train, file containing data to be clustered."
      )(
      "queries_in",
      boost::program_options::value<std::string>(),
      "REQUIRED in the --run_mode=eval, file containing the data to be assigned in the clusters"
      )(
      "memberships_out",
      boost::program_options::value<std::string>()->default_value(""),
      "OPTIONAL file to store cluster memberships."
      )(
      "distortions_out",
      boost::program_options::value<std::string>()->default_value(""),
      "OPTIONAL file to store the sum of sqaured distance of all points belonging to the "
      "same cluster"
      )(
      "centroids_out",
      boost::program_options::value<std::string>()->default_value(""),
      "OPTIONAL file to store cluster means."
      )(
      "search_method",
      boost::program_options::value<std::string>()->default_value("xmeans"),
      "The method used to find optimal K if k_max and k_min are specified."
      )(
      "k_clusters",
      boost::program_options::value<index_t>()->default_value(-1),
      "Number of clusters for KMeans. Must tbe greater than 1. "
      "If you want to do kmeans for a single k you should set this option. "
      " Otherwise you should set --k_min and --k_max"
      )(
      "k_min",
      boost::program_options::value<index_t>()->default_value(-1),
      "Minimum number of clusters to try in x-means. Must be greater than 1. "
      "If you don't set --k_min and --k_max then --k_clusters must be set."
      )(
      "k_max",
      boost::program_options::value<index_t>()->default_value(-1),
      "Maximum number of clusters to try in x-means. Must be greater than 1."
      "If you don't set --k_min and --k_max then --k_clusters must be set."
      )(
      "percentage_hold_out",
      boost::program_options::value<double>()->default_value(0.20),
      "Percentage of the dataset that will be held out as a validation set. "
      "Available for x-means."
      )(
      "n_restarts",
      boost::program_options::value<index_t>()->default_value(1),
      "Number for restarts per k-means training"
      )(
      "centroids_in",
      boost::program_options::value<std::string>()->default_value(""),
      "Initial set of centroids for --run_mode=train, or centroids for --run_mode=eval"
      )(
      "initialization",
      boost::program_options::value<std::string>()->default_value("random"),
      "When you initialize the centroids for kmeans, you have the following options:\n"
      " random   : it will pick randomly k_cluster number of points from your dataset\n"
      " kmeans++ : will use the kmeans++ algorithm that is more heavy but picks better"
      " initial centroids, that will most likely make kmeans converge faster to a better solution."
      " if you choose that option, you might want to set the --probability option to a different value"
      )(
      "probability",
      boost::program_options::value<double>()->default_value(0.8),
      " This option is valid only if you choose --initialization=kmeans++ . This is the probability of"
      " choosing the next centroid to be the furthest point in the dataset. In general this value"
      " should be set to something greater than 0.5. If you set it close to one the algorithm might be "
      " sensitive to outliers."
      )(
      "metric",
      boost::program_options::value<std::string>()->default_value("l2"),
      "Metric function used by k-means.  One of:\n"
      "  l2, weighted_l2"
      )(
      "metric_weights_in",
      boost::program_options::value<std::string>()->default_value(""),
      "A file containing weights for use with --metric=weighted_l2"
      )(
      "dense_sparse_scale",
      boost::program_options::value<double>()->default_value(1.0),
      "The scaling factor for the sparse part of --point=dense_sparse or "
      "--point=dense_categorical for use with --metric=weighted_l2"
      )(
      "algorithm",
      boost::program_options::value<std::string>()->default_value("tree"),
      "Algorithm used to compute clusters.  One of:\n"
      " online_tree  : online kmeans followed by tree based \n"
      " online_naive : online kmeans and followed by naive \n"
      " tree         : tree based kmeans \n"
      " naive        : naive kmeans"
      )(
      "tree",
      boost::program_options::value<std::string>()->default_value("kdtree"),
      "Tree structure used by k-means.  One of:\n"
      "  kdtree, balltree"
      )(
      "leaf_size",
      boost::program_options::value<index_t>()->default_value(20),
      "Maximum number of points at a leaf of the tree.  More saves on tree "
      "overhead but too much hurts asymptotic run-time."
      )(
      "iterations",
      boost::program_options::value<index_t>()->default_value(-1),
      "K-means can run in either batch or progressive mode.  If --iterations=i "
      "is omitted, K-means finds exact clusters; otherwise, it terminates after "
      "i progressive refinements."
      )(
      "epochs",
      boost::program_options::value<index_t>()->default_value(1),
      "When you run kmeans in the online mode epochs controls how many times you will pass "
      "through the entire dataset"
      )(
      "minimum_cluster_movement_threshold",
      boost::program_options::value<double>()->default_value(0.0001),
          "If in any iteration the cluster which moved most moved less than this value kmeans will terminate. "
          "Seting this to zero is a valid option and behaves as expected."
          )(
      "randomize",
      boost::program_options::value<bool>()->default_value(true),
      "When you use kmeans in online mode it is important that the points in your dataset "
      "are in a random order. For example if you have a dataset where data are coming "
      "from 2 clusters and you suspect that there are continuous chunks of datapoints "
      "belonging in the same cluster in your file, then it is a good idea to turn this "
      "flag on. In general online methods want your data to be shuffled and appear "
      "in a random order. If you have made sure this condition is satisfied by preprocessing "
      "your data, then you can turn this flag off. It will save memory and be faster"
      )(
      "run_mode",
      boost::program_options::value<std::string>()->default_value("train"),
      " Kmeans as every machine learning algorithm has two modes, the training and the evaluations."
      " When you set this flag to --run_mode=train it will find the optimal clusters for kmeans."
      " Once you have found these clusters and you want to assign points to these clusters, you"
      " should run it in the --run_mode=eval. Of course you should provide the data to be evaluated"
      " over these clusters, by setting the --queries_in flag appropriately."
      )(
      "log",
      boost::program_options::value<std::string>()->default_value(""),
      "A file to receive the log, or omit for stdout."
      )(
      "loglevel",
      boost::program_options::value<std::string>()->default_value("debug"),
      "Level of log detail.  One of:\n"
      "  debug: log everything\n"
      "  verbose: log messages and warnings\n"
      "  warning: log only warnings\n"
      "  silent: no logging"
      );

    boost::program_options::variables_map vm;
    boost::program_options::command_line_parser clp(args);
    clp.style(boost::program_options::command_line_style::default_style
      ^boost::program_options::command_line_style::allow_guessing );
    try {
      boost::program_options::store(clp.options(desc).run(), vm);  
    }
    catch(const boost::program_options::invalid_option_value &e) {
      fl::logger->Die() << "Invalid Argument: " << e.what();
    }
    catch(const boost::program_options::invalid_command_line_syntax &e) {
      fl::logger->Die() << "Invalid command line syntax: " << e.what(); 
    }
    catch (const boost::program_options::unknown_option &e) {
      fl::logger->Die() << "Unknown option: " << e.what() ;
    }

    boost::program_options::notify(vm);
    if (vm.count("help")) {
      std::cout << fl::DISCLAIMER << "\n";
      std::cout << desc << "\n";
      return 1;
    }

    fl::util::Timer timer;
    timer.Start();
    int return_value = 
      BranchType::template BranchOnTable<KMeans<boost::mpl::void_>, DataAccessType>(data, vm);
    timer.End();
    fl::logger->Message() << "Total time taken: " << timer.GetTotalElapsedTimeString().c_str();
    return return_value;
  }

  template<typename DataAccessType>
  void KMeans<boost::mpl::void_>::Run(
      DataAccessType *data,
      const std::vector<std::string> &args) {
    fl::ws::Task<
      DataAccessType,
      &Main<
        DataAccessType, 
        typename DataAccessType::Branch_t
      >
    > task(data, args);
    data->schedule(task); 
  }


  } // ml
} //fl

#endif
