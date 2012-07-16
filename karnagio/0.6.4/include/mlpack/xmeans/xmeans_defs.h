
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
#ifndef FL_LITE_MLPACK_CLUSTERING_XMEANS_DEFS_H
#define FL_LITE_MLPACK_CLUSTERING_XMEANS_DEFS_H

#include <string>
#include "boost/mpl/void.hpp"
#include "fastlib/table/branch_on_table.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "fastlib/base/base.h"
#include "xmeans.h"
#include "fastlib/util/timer.h"
#include "fastlib/table/default/dense/unlabeled/balltree/table.h"

namespace fl {
namespace ml {
template<typename TableType>
template<class DataAccessType>
int XMeans<boost::mpl::void_>::Core<TableType>::Main(
  DataAccessType *data,
  boost::program_options::variables_map &vm
) {
  if (!vm.count("references_in")) {
    fl::logger->Die() << "Missing required --references_in";
    return 1;
  }

  std::string references_in = vm["references_in"].as<std::string>();
  std::string memberships_out = vm["memberships_out"].as<std::string>();
  std::string centroids_out = vm["centroids_out"].as<std::string>();
  std::string centroids_in = vm["centroids_in"].as<std::string>();
  index_t k_clusters_max = vm["k_clusters_max"].as<index_t>();
  index_t k_clusters_min = vm["k_clusters_min"].as<index_t>();
  // point already handled
  std::string metric_arg = vm["metric"].as<std::string>();
  std::string metric_weights_in = vm["metric_weights_in"].as<std::string>();
  //double dense_sparse_scale = vm["dense_sparse_scale"].as<double>();
  std::string algorithm = vm["algorithm"].as<std::string>();
  // tree already handled
  index_t leaf_size = vm["leaf_size"].as<index_t>();
  index_t iterations = vm["iterations"].as<index_t>();
  // log already handled
  // loglevel already handled
  fl::util::Timer timer;
  timer.Start();
  TableType table;
  data->Attach(references_in, &table);
  timer.End();
  fl::logger->Debug() << "Time taken to read data: " << timer.GetTotalElapsedTimeString().c_str();

  if (iterations > 0) {
    fl::logger->Die() << "Progressive mode " << fl::NOT_SUPPORTED_MESSAGE;
    if (metric_arg == "l2") {
      if (algorithm == "tree") {

      }
      else if (algorithm == "naive") {

      }
      else {
        fl::logger->Die() << "Unknown algorithm " << algorithm;
      }
    }
    else if (metric_arg == "weighted_l2") {
      if (algorithm == "tree") {

      }
      else if (algorithm == "naive") {

      }
      else {
        fl::logger->Die() << "Unknown algorithm " << algorithm;
      }
    }
    else {
      fl::logger->Die() << "Unknown metric " << metric_arg;
    }
  }
  else {
    if (metric_arg == "l2") {
      fl::logger->Message() << "Using the l2 metric";
      
      if (algorithm=="tree") {
        fl::logger->Message() << "Indexing reference data";
		timer.Start();
		typename TableType::template IndexArgs<fl::math::LMetric<2> > q_index_args;
		q_index_args.leaf_size = leaf_size;
        table.IndexData(q_index_args);
		timer.End();
        fl::logger->Debug() << "Time taken to index data: " << timer.GetTotalElapsedTimeString().c_str();
      }

	  fl::ml::XMeans<XMeansArgs<fl::math::LMetric<2>, 
              typename DataAccessType::DefaultTable_t> > xmeans;
      if(centroids_in == "") {
        xmeans.Init(&table, k_clusters_min, k_clusters_max, initial_centroids);
      } else {
          fl::logger->Message() << "Reading initial centroids from data source: " << centroids_in;
          typename DataAccessType::DefaultTable_t initial_centroid_table;
          data->Attach(centroids_in, &initial_centroid_table);
		  k_clusters_min = initial_centroid_table.n_entries();
		  fl::logger->Message() << "Setting minimum K to " << k_clusters_min << " as read from centroids_in file.";
		  // TODO: It is terrible to die after reading the data and building the tree
		  if(k_clusters_min > k_clusters_max) {
				fl::logger->Die() << "Minimum clusters is > Maximum clusters. Please set higher value for k_clusters_max.";
		  }
		  
		  typename DataAccessType::DefaultTable_t::Point_t* initial_centroids = new typename DataAccessType::DefaultTable_t::Point_t[k_clusters_min];
          for(index_t i = 0; i < k_clusters_min; i++) {
            typename DataAccessType::DefaultTable_t::Point_t centroid_point;
            initial_centroid_table.get(i, &centroid_point);
            initial_centroids[i].Copy(centroid_point);
          }
          xmeans.Init(&table, k_clusters_min, k_clusters_max, initial_centroids);
          delete[] initial_centroids;
      }

      
      fl::logger->Message() << "Computing clusters";
	  timer.Start();
      xmeans.Run();
	  timer.End();
      fl::logger->Debug() << "Time taken to compute clusters: " << timer.GetTotalElapsedTimeString().c_str();

      if (centroids_out != "") {
        fl::logger->Message() << "Emitting cluster centroids to " << centroids_out;
        typename fl::ml::XMeans<XMeansArgs<fl::math::LMetric<2>, 
              typename DataAccessType::DefaultTable_t> >::CentroidTable_t centroids;
	    std::vector<index_t> dense_dim(1, table.n_attributes());
	    std::vector<index_t> sparse_dim;
		index_t final_k = xmeans.GetFinalK();
		fl::logger->Message() << "Final K Value: " << final_k;
        data->Attach(centroids_out,
            dense_dim, sparse_dim,
            final_k, 
            &centroids);
        xmeans. GetCentroids(&centroids);
        data->Purge(centroids);
        data->Detach(centroids);
      }

      if (memberships_out != "") {
        fl::logger->Message() << "Emitting cluster memberships to " << memberships_out;
        typename DataAccessType::template TableVector<index_t> memberships;
        data->Attach(memberships_out, 
                     std::vector<index_t>(1,1),
                     std::vector<index_t>(),
                     table.n_entries(),
                     &memberships);                       
        xmeans.GetMemberships(&memberships);
        data->Purge(memberships);
        data->Detach(memberships);
      }

      return 0;
    }
    else if (metric_arg == "weighted_l2") {
      fl::logger->Die() << "Weighted L2 metric " << fl::NOT_SUPPORTED_MESSAGE;
      if (algorithm == "tree") {

      }

      if (algorithm == "naive") {

      }
    }
    else {
      fl::logger->Die() << "Unknown metric " << metric_arg;
    }
  }
  return 1;
}

template<typename DataAccessType, typename BranchType>
int XMeans<boost::mpl::void_>::Main(
  DataAccessType *data,
  const std::vector<std::string> &args
) {
  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing data to be clustered."
  )(
    "memberships_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file to store cluster memberships."
  )(
    "centroids_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file to store cluster means."
  )(
    "k_clusters_min",
    boost::program_options::value<int>()->default_value(2),
    "Minimum number of clusters to return."
  )(
    "k_clusters_max",
    boost::program_options::value<int>()->default_value(100),
    "Maximum number of clusters to return"
  )(
    "point",
    boost::program_options::value<std::string>()->default_value("dense"),
    "Point type used by k-means.  One of:\n"
    "  dense, sparse, dense_sparse, categorical, dense_categorical"
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
    "centroids_in",
	boost::program_options::value<std::string>()->default_value(""),
    "The initial centroids"
    
  )(
    "algorithm",
    boost::program_options::value<std::string>()->default_value("tree"),
    "Algorithm used to compute clusters.  One of:\n"
    "  tree, naive"
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
    "cores",
    boost::program_options::value<int>()->default_value(1),
    "Number of cores to use for running the algorithm. If you use large number of cores "
    "increase the leaf_size"   
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
  boost::program_options::store(
    boost::program_options::command_line_parser(args).options(desc).run(), vm);
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout << fl::DISCLAIMER << "\n";
    std::cout << desc << "\n";
    return 1;
  }

  fl::util::Timer timer;
  timer.Start();
  int return_value = 
	  BranchType::template BranchOnTable<XMeans<boost::mpl::void_>, DataAccessType>(data, vm);
  timer.End();
  fl::logger->Message() << "Total time taken: " << timer.GetTotalElapsedTimeString().c_str();
  return return_value;
}



} // ml
} //fl

#endif
