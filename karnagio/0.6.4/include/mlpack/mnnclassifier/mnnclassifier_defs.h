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
#ifndef FL_LITE_MLPACK_INCLUDE_NNCLASSIFIER_NNCLASSIFIER_DEFS_H_
#define FL_LITE_MLPACK_INCLUDE_NNCLASSIFIER_NNCLASSIFIER_DEFS_H_

#include <string>
#include <iostream>
#include "mnnclassifier.h"
#include "fastlib/base/base.h"
#include "boost/mpl/void.hpp"
#include "fastlib/table/default_table.h"
#include "fastlib/table/table_vector.h"
#include "fastlib/util/timer.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "fastlib/metric_kernel/hellinger_metric.h"

namespace fl {
namespace ml {

  template<typename TableType>
  template<typename DataAccessType>
  int MNNClassifier::Core<TableType>::Main(
      DataAccessType *data,
      boost::program_options::variables_map &vm ) {
  
    std::string references_in;
    std::string queries_in;
    std::string indices_out;
    std::string distances_out;
    index_t k_neighbors=0;
    double r_neighbors=0;
    std::string metric;
    std::string metric_weights_in;
    std::string algorithm;
    index_t leaf_size=0;
    index_t iterations=0;
    std::string references_out;
    std::string queries_out;
    std::string serialize;
    std::string labels_in;
    std::string labels_out;

    try {
  	// warnings and missing
      if (!vm.count("references_in")) {
        fl::logger->Die() << "Missing required --references_in";
      }
  
      references_in = vm["references_in"].as<std::string>();
      queries_in = vm["queries_in"].as<std::string>();
      indices_out = vm["indices_out"].as<std::string>();
      distances_out = vm["distances_out"].as<std::string>();
      k_neighbors = vm["k_neighbors"].as<index_t>();
      r_neighbors = vm["r_neighbors"].as<double>();
      metric = vm["metric"].as<std::string>();
      metric_weights_in = vm["metric_weights_in"].as<std::string>();
      algorithm = vm["algorithm"].as<std::string>();
      leaf_size = vm["leaf_size"].as<index_t>();
      iterations = vm["iterations"].as<index_t>();
      references_out = vm["references_out"].as<std::string>();
      queries_out = vm["queries_out"].as<std::string>();
      serialize = vm["serialize"].as<std::string>();
      labels_in = vm["labels_in"].as<std::string>();
      labels_out = vm["labels_out"].as<std::string>();
      if(distances_out == "") {
  		  fl::logger->Warning() << "No --distances_out argument. Nearest Neighbor distances will not be output";
  	  }
  	  if(indices_out == "") {
  		  fl::logger->Warning() << "No --indices_out argument. Nearest Neighbor indices will not be output";
  	  }
      if (labels_in=="") {
        fl::logger->Die() << "Option --labels_in is not set, this is required for "
          "nn classification.";
      }
    }
    catch(const boost::program_options::invalid_option_value &e) {
  	  fl::logger->Die() << "Invalid Argument: " << e.what(); 
    }
    
  
    if (k_neighbors < 0 && r_neighbors < 0) {
      fl::logger->Die() << "You must give either --k_neighbors or --r_neighbors";
      return 1;
    } else {
      if (k_neighbors >= 0 && r_neighbors >= 0) {
        fl::logger->Die() << "You may give only one of --k_neighbors and --r_neighbors";
        return 1;
      }
    }
  
    TableType *reference_table = NULL;
  
    fl::logger->Message() << "Loading reference data from file " << references_in;
    reference_table = new TableType();
    fl::util::Timer timer;
    timer.Start();
    data->Attach(references_in, reference_table);
    timer.End();
    fl::logger->Message() << "Time taken to read reference data: " << timer.GetTotalElapsedTimeString().c_str();
    typename DataAccessType::template TableVector<index_t> labels_input_table;
    fl::logger->Message() << "Reading provided labels for classification." <<std::endl;
    timer.Start();
    if (metric=="hellinger") {
      fl::logger->Warning()<<"Hellinger distance requires data to points to be normalized "
        "such that every element is nonnegative and the L1 norm of the point is 1."
        "If your data is not normalized use table_util --hell_norm"<<std::endl;
      fl::logger->Message()<<"Preprocessing dataset for the Hellinger distance"<<std::endl;
      typename TableType::Point_t point;
      for(index_t i=0; i<reference_table->n_entries(); ++i) {
        MySqrt f;
        reference_table->get(i, &point);
        point.Transform(f);
        if (f.sum==0) {
          fl::logger->Die()<<"Point "<< i << " is all zeros";
        }
        Norm n(sqrt(f.sum));
        point.Transform(n);        
      }
    }

    data->Attach(labels_in, &labels_input_table);
    timer.End();
    fl::logger->Message() << 
      "Done. Time taken to read the labels: " 
      << timer.GetTotalElapsedTimeString()<<std::endl;
    if (labels_input_table.n_entries()!=reference_table->n_entries()) {
      fl::logger->Die() << "Reference table has " << reference_table->n_entries() 
        <<", while labels table has " << labels_input_table.n_entries();
    }


    TableType *query_table = NULL;
    if (queries_in != "") {
      fl::logger->Message() << "Loading query data from file " << queries_in;
      query_table = new TableType();
      fl::util::Timer timer;
      timer.Start();
      data->Attach(queries_in, query_table);
      timer.End();
      fl::logger->Message() << "Time taken to read query data: " << timer.GetTotalElapsedTimeString().c_str();
      if (metric=="hellinger") {
        fl::logger->Warning()<<"Hellinger distance requires data to points to be normalized "
          "such that every element is nonnegative and the L1 norm of the point is 1."
          "If your data is not normalized use table_util --hell_norm"<<std::endl;
        fl::logger->Message()<<"Preprocessing dataset for the Hellinger distance"<<std::endl;
        typename TableType::Point_t point;
        for(index_t i=0; i<query_table->n_entries(); ++i) {
          MySqrt f;
          query_table->get(i, &point);
          point.Transform(f);
          if (f.sum==0) {
            fl::logger->Die()<<"Point "<< i << " is all zeros";
          }
          Norm n(sqrt(f.sum));
          point.Transform(n);        
        }
      }
    }
  
    // Allocate the table for writing out the labels.      
    typename DataAccessType::UIntegerTable_t result_labels_table;
    if (labels_out!="") {
      data->Attach(labels_out,
  		      std::vector<int>(1, 1),
  		      std::vector<int>(),
  		      query_table==NULL?reference_table->n_entries():query_table->n_entries(),
  		      &result_labels_table);
    }

    std::vector<index_t> k_ind_neighbors;
    std::vector<std::pair<index_t, index_t> > r_ind_neighbors;
    std::vector<double>  dist_neighbors;
    typedef fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<
      typename TableType::CalcPrecision_t> > WLMetric_t;
    typename TableType::template IndexArgs<fl::math::LMetric<2> > l_index_args;
    typename TableType::template IndexArgs<WLMetric_t> w_index_args;
    typename DataAccessType::DefaultTable_t weights_table;
    typename DataAccessType::DefaultTable_t::Point_t weights_point;
    typename TableType::template IndexArgs<fl::math::HellingerMetric> hel_index_args;
  
  
    
    if (metric == "l2") {
      fl::logger->Message() << "L2 metric selected";
      l_index_args.leaf_size = leaf_size;
  
      if (reference_table->is_indexed()) {
        fl::logger->Message() << "Reference table is already indexed, skipping indexing"
          <<std::endl;
      } else {
        fl::logger->Message() << "Building index on reference data.";
        reference_table->IndexData(l_index_args);
        if (references_out!="") {
          fl::logger->Message() <<"Serializing references in "<< serialize
            <<" format"<<std::endl;
          reference_table->filename()=references_out;
          data->Purge(*reference_table, serialize);
        }
      }
  
      if (vm["log_tree_stats"].as<bool>()==true) {
          reference_table->LogTreeStats();
      }
  
      if (query_table) {
        if (query_table->is_indexed()) {
          fl::logger->Message() << "Query table is already indexed, skipping indexing"
            <<std::endl;
        } else {
          fl::logger->Message() << "Building index on query data.";
          query_table->IndexData(l_index_args);
          if (queries_out!="") {
            fl::logger->Message() <<"Serializing queries in "<< serialize
              <<" format"<<std::endl;
            query_table->filename()=queries_out;
            data->Purge(*query_table, serialize);
          }
        }
        if (vm["log_tree_stats"].as<bool>()==true) {
          query_table->LogTreeStats();
        }
      }
    } else {
      if (metric == "weighted_l2") {
        data->Attach(metric_weights_in, &weights_table);
        if (weights_table.n_entries()!=1) {
          fl::logger->Die() << "The file with the weights must have a point on "
                            << " a single line";
        }
        weights_table.get(0, &weights_point);
        w_index_args.metric.set_weights(weights_point);
    
        fl::logger->Message() << "Weighted L2 metric selected";
        w_index_args.leaf_size = leaf_size;
        if (reference_table->is_indexed()) {
          fl::logger->Message() << "Reference table is already indexed, skipping indexing"
            <<std::endl;
        } else {
          fl::logger->Message() << "Building index on reference data.";
          reference_table->IndexData(w_index_args);
          if (references_out!="") {
            fl::logger->Message() <<"Serializing references in "<< serialize
              <<" format"<<std::endl;
            reference_table->filename()=references_out;
            data->Purge(*reference_table, serialize);
          }
        }
    
        if (query_table) {
          if (query_table->is_indexed()) {
            fl::logger->Message() << "Query table is already indexed, skipping indexing"
              <<std::endl;
          } else {
            fl::logger->Message() << "Building index on query data.";
            query_table->IndexData(w_index_args);
            if (queries_out!="") {
              fl::logger->Message() <<"Serializing queries in "<< serialize
                <<" format"<<std::endl;
              query_table->filename()=queries_out;
              data->Purge(*query_table, serialize);
            }
          }
        }
      } else { 
        if (metric == "hellinger") {
          fl::logger->Message() << "Hellinger metric selected";
          hel_index_args.leaf_size = leaf_size;
  
          fl::logger->Message() << "Building index on reference data.";
          if (reference_table->is_indexed()) {
             fl::logger->Message() << "Reference table is already indexed, skipping indexing"
              <<std::endl;
          } else {
            reference_table->IndexData(hel_index_args);
            if (references_out!="") {
              fl::logger->Message() <<"Serializing references in "<< serialize
                <<" format"<<std::endl;
              reference_table->filename()=references_out;
              data->Purge(*reference_table, serialize);
            }
          }
          if (vm["log_tree_stats"].as<bool>()==true) {
            fl::logger->Message() << "Building index on reference data.";
            reference_table->LogTreeStats();
          }
  
          if (query_table) {
            if (query_table->is_indexed()) {
              fl::logger->Message() << "Query table is already indexed, skipping indexing"
                <<std::endl;
            } else {
              fl::logger->Message() << "Building index on query data.";
              query_table->IndexData(hel_index_args);
              if (queries_out!="") {
                fl::logger->Message() <<"Serializing queries in "<< serialize
                  <<" format"<<std::endl;
                query_table->filename()=queries_out;
                data->Purge(*query_table, serialize);
              }
            }
            if (vm["log_tree_stats"].as<bool>()==true) {
              query_table->LogTreeStats();
            }
          }
        } else {
          fl::logger->Die() << "Unrecognized metric " << metric;
        }
      }
    }
    
    fl::logger->Message() << "Running neighbors task.";
    timer.Start();
    AllKN<DefaultAllKNNMap> allknn;
    allknn.Init(reference_table, query_table);
  
    if (k_neighbors >= 0) {
      fl::logger->Message() << "Finding k=" << k_neighbors
      << " nearest neighbors.";
      if (iterations < 0) {
        if (metric=="l2") {
          allknn.ComputeNeighbors(
            algorithm,
            l_index_args.metric,
            k_neighbors,
            &dist_neighbors,
            &k_ind_neighbors);
        } else {
          if (metric=="weighted_l2") {
            allknn.ComputeNeighbors(
              algorithm,
              w_index_args.metric,
              k_neighbors,
              &dist_neighbors,
              &k_ind_neighbors);
          } else {
            if (metric=="hellinger") {
              allknn.ComputeNeighbors(
              algorithm,
              hel_index_args.metric,
              k_neighbors,
              &dist_neighbors,
              &k_ind_neighbors);
            }
          } 
        }
      }
      else { // iterations >= 0
        if (metric=="l2") {
          typename AllKN<DefaultAllKNNMap>
          ::template iterator<fl::math::LMetric<2>, index_t, std::vector<double>, std::vector<index_t> > it;
          it.Init(
            &allknn,
            algorithm,
            l_index_args.metric,
            k_neighbors,
            &dist_neighbors,
            &k_ind_neighbors);
  
          while (iterations > 0 && *it != 0) {
            --iterations;
            ++it;
          };
        } else {
          if (metric=="weighted_l2") {
            typename AllKN<DefaultAllKNNMap>
            ::template iterator<WLMetric_t, index_t, std::vector<double>, std::vector<index_t> > it;
            it.Init(
              &allknn,
              algorithm,
              w_index_args.metric,
              k_neighbors,
              &dist_neighbors,
              &k_ind_neighbors);
  
            while (iterations > 0 && *it != 0) {
              --iterations;
              ++it;
            };
          } else {
            if (metric=="hellinger") {
              typename AllKN<DefaultAllKNNMap>
              ::template iterator<fl::math::HellingerMetric, index_t, std::vector<double>, std::vector<index_t> > it;
              it.Init(
                &allknn,
                algorithm,
                hel_index_args.metric,
                k_neighbors,
                &dist_neighbors,
                &k_ind_neighbors);
  
              while (iterations > 0 && *it != 0) {
                --iterations;
                ++it;
              };
            }
          }
        }
      }
    }
    else { // r_neighbors >= 0.0
      fl::logger->Message() << "Finding all neighbors within radius "
      << r_neighbors << ".";
  
      if (iterations < 0) {
        if (metric=="l2") {
          allknn.ComputeNeighbors(
            algorithm,
            l_index_args.metric,
            r_neighbors,
            &dist_neighbors,
            &r_ind_neighbors);
        } else {
          if (metric=="weighted_l2") {
            allknn.ComputeNeighbors(
              algorithm,
              w_index_args.metric,
              r_neighbors,
              &dist_neighbors,
              &r_ind_neighbors);
          } else {
            if (metric=="hellinger") {
              allknn.ComputeNeighbors(
                algorithm,
                hel_index_args.metric,
                r_neighbors,
                &dist_neighbors,
                &r_ind_neighbors);
            }          
          }              
        }
      } else { // iterations >= 0
        if (metric=="l2") {
          typename AllKN<DefaultAllKNNMap>
          ::template iterator<fl::math::LMetric<2>, double, std::vector<double>, std::vector<std::pair<index_t, index_t> > > it;
          it.Init(
            &allknn,
            algorithm,
            l_index_args.metric,
            r_neighbors,
            &dist_neighbors,
            &r_ind_neighbors);
          while (iterations > 0 && *it != 0) {
            --iterations;
            ++it;
          };
        } else {
          if (metric=="weighted_l2") {
            typename AllKN<DefaultAllKNNMap>
            ::template iterator<WLMetric_t, double, std::vector<double>, std::vector<std::pair<index_t, index_t> > > it;
            it.Init(
              &allknn,
              algorithm,
              w_index_args.metric,
              r_neighbors,
              &dist_neighbors,
              &r_ind_neighbors);
            while (iterations > 0 && *it != 0) {
              --iterations;
              ++it;
            };
          } else {
            if (metric=="hellinger") {
              typename AllKN<DefaultAllKNNMap>
                ::template iterator<fl::math::HellingerMetric, double, std::vector<double>, std::vector<std::pair<index_t, index_t> > > it;
              it.Init(
                &allknn,
                algorithm,
                hel_index_args.metric,
                r_neighbors,
                &dist_neighbors,
                &r_ind_neighbors);
              while (iterations > 0 && *it != 0) {
                --iterations;
                ++it;
              };
            }         
          }       
        } 
      }
    }
    timer.End();
    fl::logger->Message() << "Finished running the neighbors task. Time taken: " << timer.GetTotalElapsedTimeString().c_str();
    // Export the results
    if (k_neighbors >= 0) {
      index_t num_outputs = query_table ?
                            query_table->n_entries() : reference_table->n_entries();
  
      typename DataAccessType::UIntegerTable_t indices_output_table;
      data->Attach(indices_out,
        std::vector<index_t>(1, k_neighbors),
        std::vector<index_t>(),
        num_outputs,
        &indices_output_table);
      typename DataAccessType::DefaultTable_t dists_output_table;
      data->Attach(distances_out,
        std::vector<index_t>(1, k_neighbors),
        std::vector<index_t>(),
        num_outputs,
        &dists_output_table);
      for (int i = 0; i < num_outputs; ++i) {
        typename DataAccessType::UIntegerTable_t::Point_t index_point;
        typename DataAccessType::DefaultTable_t::Point_t dist_point;
        indices_output_table.get(i, &index_point);
        dists_output_table.get(i, &dist_point);
        for (int j = 0; j < k_neighbors; j++) {
          index_point.set(j, k_ind_neighbors[i*k_neighbors+j]);
          dist_point.set(j, dist_neighbors[i*k_neighbors+j]);
        }
      }
      
      // Write out to the file.
      
      if (indices_out != "") {
        fl::logger->Message() << "Emitting index results to " << indices_out;
        timer.Start();
        data->Purge(indices_output_table);
        data->Detach(indices_output_table);
        timer.End();
        fl::logger->Message() << "Timer taken to write indices: " << timer.GetTotalElapsedTimeString().c_str();
      }
      if (distances_out != "") {
        fl::logger->Message() << "Emitting distance results to " << distances_out;
        timer.Start();
        data->Purge(dists_output_table);
        data->Detach(dists_output_table);
        timer.End();
        fl::logger->Message() << "Time taken to write distances: " << timer.GetTotalElapsedTimeString().c_str();
      }
      double score;
      fl::logger->Message() << "Starting the classification task. ";
      timer.Start();
      ComputeNNClassification(indices_output_table,
  		  	      dists_output_table,
  			        labels_input_table,
  			        &result_labels_table,
  			        (queries_in == ""),
  			        &score);
      timer.End();
      fl::logger->Message() << "Finished classifying. Time taken to classify: " << timer.GetTotalElapsedTimeString().c_str();
      if(queries_in == "") {
    	  fl::logger->Message() << "Classification score on the reference set is " << score;
      }
    } else { // r_neighbors >= 0.0
      typename DataAccessType::DefaultSparseIntTable_t indices_output_table;
      data->Attach(indices_out,
                   std::vector<index_t>(),
                   std::vector<index_t>(1, reference_table->n_entries()),
                   query_table==NULL?reference_table->n_entries():query_table->n_entries(),
                   &indices_output_table);
  
      typename DataAccessType::DefaultSparseDoubleTable_t dists_output_table;
      data->Attach(distances_out,
         std::vector<index_t>(),
         std::vector<index_t>(1, reference_table->n_entries()),
         query_table==NULL?reference_table->n_entries():query_table->n_entries(),
        &dists_output_table);
  
      fl::logger->Message() << "Got " << dist_neighbors.size() <<
      " results for range neighbor";
      for (unsigned int i = 0; i < dist_neighbors.size(); ++i) {
        typename DataAccessType::DefaultSparseIntTable_t::Point_t index_point;
        typename DataAccessType::DefaultSparseDoubleTable_t::Point_t dist_point;
        indices_output_table.get(r_ind_neighbors[i].first, &index_point);
        dists_output_table.get(r_ind_neighbors[i].first, &dist_point);
        index_point.set(r_ind_neighbors[i].second, 1);
        dist_point.set(r_ind_neighbors[i].second, dist_neighbors[i]);
      }

 
      // Write out to the file.
      if (indices_out != "") {
        fl::logger->Message() << "Emitting index results to " << indices_out;
        timer.Start();
        fl::logger->Message() << "Emitting index results to " << indices_out;
        data->Purge(indices_output_table);
        data->Detach(indices_output_table);
        timer.End();
        fl::logger->Message() << "Timer taken to write indices: " << timer.GetTotalElapsedTimeString().c_str();
      }
      if (distances_out != "") {
        fl::logger->Message() << "Emitting distance results to " << distances_out;
        timer.Start();
        data->Purge(dists_output_table);
        data->Detach(dists_output_table);
        timer.End();
        fl::logger->Message() << "Time taken to write distances: " << timer.GetTotalElapsedTimeString().c_str();
      }    
      double score;
      fl::logger->Message() << "Starting the classification task. ";
      timer.Start();
      ComputeNNClassification(indices_output_table,
  		  	      dists_output_table,
  			        labels_input_table,
  			        &result_labels_table,
  			        (queries_in == ""),
  			        &score);
      timer.End();
      fl::logger->Message() << "Finished classifying. Time taken to classify: " << timer.GetTotalElapsedTimeString().c_str();
      if(queries_in == "") {
    	  fl::logger->Message() << "Classification score on the reference set is " << score;
      }
    }
  
   
    if( labels_out != "" ) {
      fl::logger->Message() << "Writing classification results to " << labels_out;
      timer.Start();  
      data->Purge(result_labels_table);
      data->Detach(result_labels_table);
      timer.End();
      fl::logger->Message() << "Finished writing output labels. Time taken to write: " << timer.GetTotalElapsedTimeString().c_str();
    }
    
   
    // free memory
    delete reference_table;
    delete query_table;
  
    return 0;
  }


  template<typename DataAccessType, typename BranchType>
  int MNNClassifier::Main(DataAccessType *data,
    const std::vector<std::string> &args) {
  
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_in",
      boost::program_options::value<std::string>(),
      "REQUIRED file containing reference data"
    )(
      "references_out",
      boost::program_options::value<std::string>()->default_value(""),
      "OPTIONAL file where the references_in data will be serialized. You can"
      " use this option to save your data after they have been indexed with a tree."
      " Then you can reuse them for this or another algorithm without having to"
      " rebuild the tree"
    )(
      "queries_in",
      boost::program_options::value<std::string>()->default_value(""),
      "OPTIONAL file containing query positions.  If omitted, allkn "
      "finds leave-one-out neighbors for each reference point."
    )(
      "queries_out", 
      boost::program_options::value<std::string>()->default_value(""),
      "OPTIONAL file where the queries_in data will be serialized. You can"
      " use this option to save your data after they have been indexed with a tree."
      " Then you can reuse them for this or another algorithm without having to"
      " rebuild the tree"
    )(
      "serialize",
      boost::program_options::value<std::string>()->default_value("binary"),
      "OPTIONAL when you serialize the tables to files, you have the option to use:\n "
      "  binary for smaller files\n"
      "  text   for portability  (bigger files)\n"
      "  xml    for interpretability and portability (even bigger files)"
    )(
      "indices_out",
      boost::program_options::value<std::string>()->default_value(""),
      "OPTIONAL file to store found neighbor indices"
    )(
      "distances_out",
      boost::program_options::value<std::string>()->default_value(""),
      "OPTIONAL file to store found neighbor distances"
    )(
      "distances_in",
      boost::program_options::value<std::string>()->default_value(""),
      "OPTIONAL file containing distances between neighboring points. "
      "if flag --method=classification then you should provide --distances_in"
    )(
      "labels_in",
      boost::program_options::value<std::string>()->default_value(""),
      "OPTIONAL file containing the labels of reference points. "
      "if flag --method=classification then you should provide --labels_in"
    )(
      "labels_out",
      boost::program_options::value<std::string>()->default_value(""),
      "OPTIONAL FILE for outputting the computed nearest neighbor classification "
      "labels (for --method=classification or --method=nnclassification)."
    )(
      "k_neighbors",
      boost::program_options::value<index_t>()->default_value(-1),
      "The number of neighbors to find for the all-k-neighbors method."
    )(
      "r_neighbors",
      boost::program_options::value<double>()->default_value(-1.0),
      "The query radius for the all-range-neighbors method.\n"
      "One of --k_neighbors or --r_neighbors must be given."
    )(
      "point",
      boost::program_options::value<std::string>()->default_value("dense"),
      "Point type used by allkn.  One of:\n"
      "  dense, sparse, dense_sparse, categorical, dense_categorical"
    )(
      "metric",
      boost::program_options::value<std::string>()->default_value("l2"),
      "Metric function used by allkn.  One of:\n"
      "  l2, weighted_l2, hellinger"
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
      boost::program_options::value<std::string>()->default_value("dual"),
      "Algorithm used to compute densities.  One of:\n"
      "  dual, single"
    )(
      "tree",
      boost::program_options::value<std::string>()->default_value("kdtree"),
      "Tree structure used by allkn.  One of:\n"
      "  kdtree, balltree"
    )(
      "leaf_size",
      boost::program_options::value<index_t>()->default_value(20),
      "Maximum number of points at a leaf of the tree.  More saves on tree "
      "overhead but too much hurts asymptotic run-time."
    )(
      "iterations",
      boost::program_options::value<index_t>()->default_value(-1),
      "Allkn can run in either batch or progressive mode.  If --iterations=i "
      "is omitted, allkn finds exact neighbors; otherwise, it terminates after "
      "i progressive refinements."
    )(
      "log_tree_stats",
      boost::program_options::value<bool>()->default_value(true),
      "If this flag is set true then it outputs some statistics about the tree after it is built. "
      "We suggest you set that flag on. If the tree is not correctly built, due to wrong options"
      " or due to pathological data then there is not point in running nearest neighbors"
    )(
      "cores",
      boost::program_options::value<int>()->default_value(1),
      "Number of cores to use for running the algorithm. If you use large number of cores "
      "increase the leaf_size, This feature is disabled for the moment" 
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
         fl::logger->Die() << "Unknown option: " << e.what();
    }
  
    boost::program_options::notify(vm);
    if (vm.count("help")) {
      std::cout << fl::DISCLAIMER << "\n";
      std::cout << desc << "\n";
      return 1;
    }
   
    return BranchType::template BranchOnTable<MNNClassifier, DataAccessType>(data, vm);
  }

  template<typename IndTableType,
           typename DistTableType,
           typename ClassLabelTableType,
           typename ResultTableType>
  void MNNClassifier::ComputeNNClassification(const fl::table::Table<IndTableType> &ind,
  			     const fl::table::Table<DistTableType> &dist,
  			     const fl::table::Table<ClassLabelTableType> &reference_labels,
             const fl::table::Table<ClassLabelTableType> &query_labels,
  			     fl::table::Table<ResultTableType> *result,
  			     bool compute_score,
             int auc_label,
  			     double *total_score,
             double *auc,
             std::vector<std::pair<double, double> > *roc,
             std::map<int, int> *points_per_class,
             std::map<int, double> *partial_score) {
  
    BOOST_ASSERT(ind.n_entries() == dist.n_entries());
    // Loop over each query point.
    *total_score = 0;
    std::vector<double> a_class;
    std::vector<double> b_class;
    if (auc==NULL && roc !=NULL) {
      fl::logger->Warning()<<"You requested compute ROC without computing AUC."
        <<" ROC will not be computed";
    }
    for (int i = 0; i < ind.n_entries(); i++) {
      // Get the reference to the query result so that we can write ont
      // it.
      typename fl::table::Table<ResultTableType>::Point_t result_point;
      result->get(i, &result_point);
  
      // For each query, we maintain the votes for each class label.
      std::map<int, int> votes;
  
      // Get the neighbor indices for the current query point.
      typename fl::table::Table<IndTableType>::Point_t point;
      ind.get(i, &point);
  
      // Make all neighbors vote.
      for (typename fl::table::Table<IndTableType>::Point_t::iterator it=point.begin(); 
          it != point.end(); ++it) {
        
        // Get the reference index of the nearest neighbor.
        int reference_index = point[it.attribute()];
        
        // Get the label of the reference nearest neighbor.
        typename fl::table::Table<ClassLabelTableType>::Point_t reference_point_label;
        reference_labels.get(reference_index, &reference_point_label);
        int reference_label = reference_point_label[0];
  
        if (votes.find(reference_label) != votes.end() ) {
          votes[reference_label] += 1;
        }
        else {
  	      votes[reference_label] = 1;
        }
      }
  
      // Find the class label with the highest vote.
      int max_class_label_count = 0;
      int max_class_label = -1;
      for(std::map<int, int>::iterator it = votes.begin();
  	    it != votes.end(); it++) {
        if(it->second > max_class_label_count) {
        	max_class_label = it->first;
    	    max_class_label_count = it->second;
        }
      }
      result_point.set(0, max_class_label);
  
      // Compute accuracy.
      if(compute_score) {
        typename fl::table::Table<ClassLabelTableType>::Point_t query_point_label;
        query_labels.get(i, &query_point_label);
        if (points_per_class!=NULL) {
          (*points_per_class)[query_point_label[0]]+=1;
        }
        if (auc!=NULL) {
          if (query_point_label[0]==auc_label) {
            a_class.push_back(votes[auc_label]);
          } else {
            b_class.push_back(votes[auc_label]);
          }
        }
        if( max_class_label == query_point_label[0] ) {
          (*total_score) += 1.0;
          if (partial_score!=NULL) {
            (*partial_score)[query_point_label[0]]+=1;
          }
        }
      }
    } // end of looping over each query point.
  
    // Normalize the score.
    (*total_score) /= ( (double) ind.n_entries() );
    // Normalize the partial scores
    if (partial_score!=NULL && compute_score==true) {
      for(std::map<int, double>::iterator it=partial_score->begin();
        it!=partial_score->end(); ++it) {
        it->second/=(*points_per_class)[it->first];
      }
    }
    if (auc!=NULL && compute_score==true) {
      ComputeAUC(a_class, b_class, auc, roc);      
    }
  }

  template<typename IndTableType,
           typename DistTableType,
           typename ClassLabelTableType,
           typename ResultTableType>
  void MNNClassifier::ComputeNNClassification(const fl::table::Table<IndTableType> &ind,
  			     const fl::table::Table<DistTableType> &dist,
  			     const fl::table::Table<ClassLabelTableType> &reference_labels,
  			     fl::table::Table<ResultTableType> *result,
  			     bool compute_score,
  			     double *total_score) {

    ComputeNNClassification(ind,
  			     dist,
  			     reference_labels,
             reference_labels,
  			     result,
  			     compute_score,
  			     total_score,
             NULL,
             NULL,
             NULL);
  }
}} // namespaces

#endif

