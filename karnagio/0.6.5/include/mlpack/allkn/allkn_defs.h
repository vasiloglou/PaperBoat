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
#ifndef FL_LITE_MLPACK_ALLKN_ALLKN_DEFS_H_
#define FL_LITE_MLPACK_ALLKN_ALLKN_DEFS_H_

#include <string>
#include <iostream>
#include "allkn.h"
#include "fastlib/base/base.h"
#include "boost/mpl/void.hpp"
#include "fastlib/util/timer.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "fastlib/metric_kernel/hellinger_metric.h"
#include "mlpack/mnnclassifier/mnnclassifier_defs.h"
#include "fastlib/workspace/task.h"

namespace fl {
namespace ml {

template<typename TableType>
template<typename DataAccessType>
int AllKN<boost::mpl::void_>::Core<TableType>::Main(
    DataAccessType *data,
    boost::program_options::variables_map &vm ) {
  FL_SCOPED_LOG(Allkn);
  std::string references_in;
  std::string reference_labels_in;
  std::string query_labels_in;
  std::string queries_in;
  std::string indices_out;
  std::string distances_out;
  std::string method;
  index_t k_neighbors=0;
  double r_neighbors=0;
  std::string metric;
  std::string metric_weights_in;
  std::string algorithm;
  // index_t leaf_size=0;
  index_t iterations=0;
  std::string labels_out;
  std::string references_out;
  std::string queries_out;
  std::string serialize;
  bool auc=true;
  index_t auc_label=1; 
  std::string roc_out="";

  try {
	// warnings and missing
    if (!vm.count("references_in") && vm["method"].as<std::string>() != "classification") {
      fl::logger->Die() << "Missing required --references_in";
    }

    references_in = vm["references_in"].as<std::string>();
    reference_labels_in = vm["reference_labels_in"].as<std::string>();
    query_labels_in = vm["query_labels_in"].as<std::string>();
    queries_in = vm["queries_in"].as<std::string>();
    indices_out = vm["indices_out"].as<std::string>();
    if (indices_out=="") {
      indices_out=data->GiveTempVarName();
    }
    distances_out = vm["distances_out"].as<std::string>();
    if (distances_out=="") {
      distances_out= data->GiveTempVarName();
    }
    method = vm["method"].as<std::string>();
    k_neighbors = vm["k_neighbors"].as<index_t>();
    r_neighbors = vm["r_neighbors"].as<double>();
    metric = vm["metric"].as<std::string>();
    metric_weights_in = vm["metric_weights_in"].as<std::string>();
    algorithm = vm["algorithm"].as<std::string>();
    // leaf_size = vm["leaf_size"].as<index_t>();
    iterations = vm["iterations"].as<index_t>();
    labels_out = vm["labels_out"].as<std::string>();
    references_out = vm["references_out"].as<std::string>();
    queries_out = vm["queries_out"].as<std::string>();
    serialize = vm["serialize"].as<std::string>();
    auc=vm["auc"].as<bool>();
    auc_label=vm["auc_label"].as<int>();
    roc_out=vm["roc_out"].as<std::string>();
	  if(distances_out == "") {
		  fl::logger->Warning() << "No --distances_out argument. Nearest Neighbor distances will not be output";
	  }
	  if(indices_out == "") {
		  fl::logger->Warning() << "No --indices_out argument. Nearest Neighbor indices will not be output";
	  }
	  if((method == "nnclassification" || "classification") && labels_out == "") {
		  fl::logger->Warning() << "No --labels_out argument. Classification labels will not be output";
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

  boost::shared_ptr<TableType> reference_table;

  if (vm["method"].as<std::string>()!="classification") {
    fl::logger->Message() << "Loading reference data from file " << references_in;
    reference_table.reset(new TableType());
    fl::util::Timer timer;
    timer.Start();
    data->Attach(references_in, &reference_table);
    if (reference_table->is_indexed()==false) {
      fl::logger->Die()<<"Reference table ("<<references_in
        <<")  is not indexed"; 
    }
    timer.End();
    fl::logger->Message() << "Time taken to read reference data: " << timer.GetTotalElapsedTimeString().c_str();
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
  }

  boost::shared_ptr<TableType> query_table;
  if (queries_in != "") {
    fl::logger->Message() << "Loading query data from file " << queries_in;
    query_table.reset(new TableType());
    fl::util::Timer timer;
    timer.Start();
    data->Attach(queries_in, &query_table);
    if (query_table->is_indexed()==false) {
      if (algorithm=="dual") {
        fl::logger->Die()<<"Query table ("<<queries_in
          <<") is not indexed. When --algorithm="
          <<algorithm<<" then query table must be indexed";
      }
    }
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

  std::vector<index_t> k_ind_neighbors;
  std::vector<std::pair<index_t, index_t> > r_ind_neighbors;
  std::vector<double>  dist_neighbors;
  typedef fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<
    typename TableType::CalcPrecision_t> > WLMetric_t;
  typename TableType::template IndexArgs<fl::math::LMetric<2> > l_index_args;
  typename TableType::template IndexArgs<WLMetric_t> w_index_args;
  boost::shared_ptr<typename DataAccessType::DefaultTable_t> weights_table;
  typename DataAccessType::DefaultTable_t::Point_t weights_point;
  typename TableType::template IndexArgs<fl::math::HellingerMetric> hel_index_args;


//  if (method=="furthest" || method=="nearest" || method=="nnclassification") {
//    if (metric == "l2") {
//      fl::logger->Message() << "L2 metric selected";
//      l_index_args.leaf_size = leaf_size;
//  
//      if (reference_table->is_indexed()) {
//        fl::logger->Message() << "Reference table is already indexed, skipping indexing"
//          <<std::endl;
//      } else {
//        fl::logger->Message() << "Building index on reference data.";
//        reference_table->IndexData(l_index_args);
//        if (references_out!="") {
//          fl::logger->Message() <<"Serializing references in "<< serialize
//            <<" format"<<std::endl;
//          reference_table->filename()=references_out;
//          data->Purge(*reference_table, serialize);
//        }
//      }
//
//      if (vm["log_tree_stats"].as<bool>()==true) {
//          reference_table->LogTreeStats();
//      }
//  
//      if (query_table) {
//        if (query_table->is_indexed()) {
//          fl::logger->Message() << "Query table is already indexed, skipping indexing"
//            <<std::endl;
//        } else {
//          fl::logger->Message() << "Building index on query data.";
//          query_table->IndexData(l_index_args);
//          if (queries_out!="") {
//            fl::logger->Message() <<"Serializing queries in "<< serialize
//              <<" format"<<std::endl;
//            query_table->filename()=queries_out;
//            data->Purge(*query_table, serialize);
//          }
//        }
//        if (vm["log_tree_stats"].as<bool>()==true) {
//          query_table->LogTreeStats();
//        }
//      }
//    } else {
//      if (metric == "weighted_l2") {
//        data->Attach(metric_weights_in, weights_table);
//        if (weights_table->n_entries()!=1) {
//          fl::logger->Die() << "The file with the weights must have a point on "
//                            << " a single line";
//        }
//        weights_table->get(0, &weights_point);
//        w_index_args.metric.set_weights(weights_point);
//    
//        fl::logger->Message() << "Weighted L2 metric selected";
//        w_index_args.leaf_size = leaf_size;
//        if (reference_table->is_indexed()) {
//          fl::logger->Message() << "Reference table is already indexed, skipping indexing"
//            <<std::endl;
//        } else {
//          fl::logger->Message() << "Building index on reference data.";
//          reference_table->IndexData(w_index_args);
//          if (references_out!="") {
//            fl::logger->Message() <<"Serializing references in "<< serialize
//              <<" format"<<std::endl;
//            reference_table->filename()=references_out;
//            data->Purge(*reference_table, serialize);
//          }
//        }
//    
//        if (query_table) {
//          if (query_table->is_indexed()) {
//            fl::logger->Message() << "Query table is already indexed, skipping indexing"
//              <<std::endl;
//          } else {
//            fl::logger->Message() << "Building index on query data.";
//            query_table->IndexData(w_index_args);
//            if (queries_out!="") {
//              fl::logger->Message() <<"Serializing queries in "<< serialize
//                <<" format"<<std::endl;
//              query_table->filename()=queries_out;
//              data->Purge(*query_table, serialize);
//            }
//          }
//        }
//      } else { 
//        if (metric == "hellinger") {
//          fl::logger->Message() << "Hellinger metric selected";
//          hel_index_args.leaf_size = leaf_size;
//  
//          fl::logger->Message() << "Building index on reference data.";
//          if (reference_table->is_indexed()) {
//             fl::logger->Message() << "Reference table is already indexed, skipping indexing"
//              <<std::endl;
//          } else {
//            reference_table->IndexData(hel_index_args);
//            if (references_out!="") {
//              fl::logger->Message() <<"Serializing references in "<< serialize
//                <<" format"<<std::endl;
//              reference_table->filename()=references_out;
//              data->Purge(*reference_table, serialize);
//            }
//          }
//          if (vm["log_tree_stats"].as<bool>()==true) {
//            fl::logger->Message() << "Building index on reference data.";
//            reference_table->LogTreeStats();
//          }
//  
//          if (query_table) {
//            if (query_table->is_indexed()) {
//              fl::logger->Message() << "Query table is already indexed, skipping indexing"
//                <<std::endl;
//            } else {
//              fl::logger->Message() << "Building index on query data.";
//              query_table->IndexData(hel_index_args);
//              if (queries_out!="") {
//                fl::logger->Message() <<"Serializing queries in "<< serialize
//                  <<" format"<<std::endl;
//                query_table->filename()=queries_out;
//                data->Purge(*query_table, serialize);
//              }
//            }
//            if (vm["log_tree_stats"].as<bool>()==true) {
//              query_table->LogTreeStats();
//            }
//          }
//        } else {
//          fl::logger->Die() << "Unrecognized metric " << metric;
//        }
//      }
//    }
//  } 
  if (method == "classification") {
    std::string indices_in = vm["indices_in"].as<std::string>();
    if (indices_in=="") {
      fl::logger->Die() << "Option --indices_in is not set, this is required for "
        "knn classification ";
    }
    std::string distances_in = vm["distances_in"].as<std::string>();
    if (distances_in=="") {
      fl::logger->Die() << "Option --distances_in is not set, this is required for "
        "knn classification.";
    }
    std::string query_labels_in = vm["query_labels_in"].as<std::string>();
    if (query_labels_in=="") {
      fl::logger->Die() << "Option --query_labels_in is not set, this is required for "
        "knn classification.";
    }
    if (k_neighbors <=0) {
      fl::logger->Die() << "Option --k_neighbors must be greater than zero for knn "
                        << "classification.";
    }
    boost::shared_ptr<typename DataAccessType::UIntegerTable_t> indices_input_table;
    boost::shared_ptr<typename DataAccessType::DefaultTable_t> distances_input_table;
    boost::shared_ptr<typename DataAccessType::IntegerTable_t> reference_labels_input_table;
    boost::shared_ptr<typename DataAccessType::IntegerTable_t> query_labels_input_table;

    fl::logger->Message() << "Reading indices, labels and corresponding distances for classification task.";
    fl::util::Timer timer;
    timer.Start();
    data->Attach(indices_in, &indices_input_table);
    data->Attach(distances_in, &distances_input_table);
    data->Attach(reference_labels_in, &reference_labels_input_table);
    if (query_labels_in!="") {
      data->Attach(query_labels_in, &query_labels_input_table);    
    }
    timer.End();
    fl::logger->Message() << "Time taken to read input indices, labels and distances: " << timer.GetTotalElapsedTimeString().c_str();

    boost::shared_ptr<typename DataAccessType::IntegerTable_t> result_labels_table;
    data->Attach(labels_out,
		 std::vector<index_t>(1, 1),
		 std::vector<index_t>(),
		 indices_input_table->n_entries(),
		 &result_labels_table);

    double score=0;
    double auc_score=0;
    std::map<int, int> points_per_class;
    std::map<int, double> partial_score;
    std::vector<std::pair<double, double> > roc_vec;
    fl::logger->Message() << "Starting classification process.";
    timer.Start();
    if (auc==false) {
      MNNClassifier::ComputeNNClassification(*indices_input_table,
			    *distances_input_table,
			    *reference_labels_input_table,
          query_labels_in==""?*reference_labels_input_table:*query_labels_input_table,
			    result_labels_table.get(),
          (queries_in == "") || (query_labels_in!=""),
          auc_label,
          &score,
          &auc_score,
          roc_out==""?NULL:&roc_vec,
          &points_per_class,
          &partial_score);
    } else {
      MNNClassifier::ComputeNNClassification(*indices_input_table,
			    *distances_input_table,
			    *reference_labels_input_table,
          query_labels_in==""?*reference_labels_input_table:*query_labels_input_table,
			    result_labels_table.get(),
          (queries_in == "") || (query_labels_in!=""),
          auc_label,
          &score,
          NULL,
          NULL,
          &points_per_class,
          &partial_score);
    }
    timer.End();
    fl::logger->Message() << "Finished. Time taken for classification: " << timer.GetTotalElapsedTimeString().c_str();
    
    if(labels_out != "") {
      fl::logger->Message() << "Writing classification results to output file " << labels_out;
      timer.Start();
      data->Purge(labels_out);
      data->Detach(labels_out);
      timer.End();
      fl::logger->Message() << "Finished writing results. Time taken: " << timer.GetTotalElapsedTimeString().c_str();
    }
    if( labels_out != "") {
      fl::logger->Message() << "Writing classification results to " << labels_out;
      timer.Start();  
      data->Purge(labels_out);
      data->Detach(labels_out);
      timer.End();
      fl::logger->Message() << "Finished writing output labels. Time taken to write: " << timer.GetTotalElapsedTimeString().c_str();
    }
    if(queries_in == "" || query_labels_in!="") {
      if (queries_in =="") {
        fl::logger->Message() << "Classification score on the reference set is " <<
	          score;
      } else {
        fl::logger->Message() << "Classification score on the query set is " <<
	          score;
      }
      fl::logger->Message() << "The score per class is ";
      for(std::map<int, int>::iterator it=points_per_class.begin();
          it!=points_per_class.end(); ++it) {
        fl::logger->Message()<<"Class: "<<it->first<<", points: "<<it->second
            <<", score: "<<100*partial_score[it->first];
      }

      if (auc) {
        fl::logger->Message() << "The Area Under the curve (AUC) is "<< auc_score;
        if (roc_out!="") {
          fl::logger->Message()<<"Exporting roc curve to "<<roc_out;
          boost::shared_ptr<typename DataAccessType::DefaultTable_t> roc_table;
          data->Attach(roc_out, 
              std::vector<index_t>(1,2),
              std::vector<index_t>(),
              roc_vec.size(),
              &roc_table);
          typename DataAccessType::DefaultTable_t::Point_t p;
          for(int i=0; i<roc_table->n_entries(); ++i) {
            roc_table->get(i, &p);
            p.set(0, roc_vec[i].first);
            p.set(1, roc_vec[i].second);
          }
          data->Purge(roc_out);
          data->Detach(roc_out);
        }
      }
    }
    return 0;   
  }
  
  fl::logger->Message() << "Running neighbors task.";
  fl::util::Timer timer;
  timer.Start();
  if (method == "nearest" || method == "nnclassification") {
    AllKN<DefaultAllKNNMap> allknn;
    if (reference_table.get()!= query_table.get()) {
      allknn.Init(reference_table.get(), query_table.get());
    } else {
      allknn.Init(reference_table.get(), NULL);
    }

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
            if (metric=="hel") {
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
  } else {
    if (method == "furthest") {
      AllKN<DefaultAllKFNMap> allknn;
      if (reference_table.get()!= query_table.get()) {
        allknn.Init(reference_table.get(), query_table.get());
      } else {
        allknn.Init(reference_table.get(), NULL);
      }
      if (k_neighbors >= 0) {
        fl::logger->Message() << "Finding k=" << k_neighbors
        << " furthest neighbors.";
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
        } else { // iterations >= 0
          if (metric=="l2") {
            typename AllKN<DefaultAllKFNMap>
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
              typename AllKN<DefaultAllKFNMap>
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
                typename AllKN<DefaultAllKFNMap>
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
      } else { // r_neighbors >= 0.0
        fl::logger->Message() << "Finding all neighbors outside radius "
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
            typename AllKN<DefaultAllKFNMap>
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
              typename AllKN<DefaultAllKFNMap>
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
                typename AllKN<DefaultAllKFNMap>
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
    } else {
      fl::logger->Die() << "Unrecognized method " << method;
    }
  }
  timer.End();
  fl::logger->Message() << "Finished running the neighbors task. Time taken: " << timer.GetTotalElapsedTimeString().c_str();
  // Export the results
  if (k_neighbors >= 0) {
    index_t num_outputs = query_table ?
                          query_table->n_entries() : reference_table->n_entries();

    boost::shared_ptr<typename DataAccessType::UIntegerTable_t> indices_output_table;
    data->Attach(indices_out,
      std::vector<index_t>(1, k_neighbors),
      std::vector<index_t>(),
      num_outputs,
      &indices_output_table);
    boost::shared_ptr<typename DataAccessType::DefaultTable_t> dists_output_table;
    data->Attach(distances_out,
      std::vector<index_t>(1, k_neighbors),
      std::vector<index_t>(),
      num_outputs,
      &dists_output_table);
    for (int i = 0; i < num_outputs; ++i) {
      typename DataAccessType::UIntegerTable_t::Point_t index_point;
      typename DataAccessType::DefaultTable_t::Point_t dist_point;
      indices_output_table->get(i, &index_point);
      dists_output_table->get(i, &dist_point);
      for (int j = 0; j < k_neighbors; j++) {
        index_point.set(j, k_ind_neighbors[i*k_neighbors+j]);
        dist_point.set(j, dist_neighbors[i*k_neighbors+j]);
      }
    }
    
    // Write out to the file.
    
    if (indices_out != "") {
//      fl::logger->Message() << "Emitting index results to " << indices_out;
      timer.Start();
      data->Purge(indices_out);
      data->Detach(indices_out);
      timer.End();
      fl::logger->Message() << "Timer taken to write indices: " << timer.GetTotalElapsedTimeString().c_str();
    }
    if (distances_out != "") {
//      fl::logger->Message() << "Emitting distance results to " << distances_out;
      timer.Start();
      data->Purge(distances_out);
      data->Detach(distances_out);
      timer.End();
      fl::logger->Message() << "Time taken to write distances: " << timer.GetTotalElapsedTimeString().c_str();
    }
    if (method=="nnclassification") {
      boost::shared_ptr<typename DataAccessType::IntegerTable_t > reference_labels_input_table;
      boost::shared_ptr<typename DataAccessType::IntegerTable_t > query_labels_input_table;
      fl::logger->Message() << "Reading provided labels for classification.";
      timer.Start();
      data->Attach(reference_labels_in, &reference_labels_input_table);
      if (query_labels_in!="") {
        data->Attach(query_labels_in, &query_labels_input_table);
      }
      timer.End();
      fl::logger->Message() << "Done. Time taken to read the labels: " << timer.GetTotalElapsedTimeString().c_str();
      if (reference_labels_input_table->n_entries()!=reference_table->n_entries()) {
        fl::logger->Die() << "Reference table has " << reference_table->n_entries() 
          <<", while labels table has " << reference_labels_input_table->n_entries();
      }
      // Allocate the table for writing out the labels.      
      boost::shared_ptr<typename DataAccessType::IntegerTable_t> result_labels_table;
      data->Attach(labels_out,
		      std::vector<index_t>(1, 1),
		      std::vector<index_t>(),
		      indices_output_table->n_entries(),
		      &result_labels_table);

      double score=0;
      double auc_score=0;
      std::map<int, int> points_per_class;
      std::map<int, double> partial_score;
      std::vector<std::pair<double, double> > roc_vec;
      fl::logger->Message() << "Starting the classification task. ";
      timer.Start();
      if (auc==true ) {
        MNNClassifier::ComputeNNClassification(*indices_output_table,
			        *dists_output_table,
			        *reference_labels_input_table,
              query_labels_in==""?*reference_labels_input_table:*query_labels_input_table,
			        result_labels_table.get(),
			        (queries_in == "") || (query_labels_in!=""),
			        auc_label,
              &score,
              &auc_score,
              roc_out==""?NULL:&roc_vec,
              &points_per_class,
              &partial_score);
      } else { 
        MNNClassifier::ComputeNNClassification(*indices_output_table,
			        *dists_output_table,
			        *reference_labels_input_table,
              query_labels_in==""?*reference_labels_input_table:*query_labels_input_table,
			        result_labels_table.get(),
			        (queries_in == "") || (query_labels_in!=""),
			        auc_label,
              &score,
              NULL,
              NULL,
              &points_per_class,
              &partial_score);
      }
      timer.End();
      fl::logger->Message() << "Finished classifying. Time taken to classify: " << timer.GetTotalElapsedTimeString().c_str();
      
      if( labels_out != "" ) {
        fl::logger->Message() << "Writing classification results to " << labels_out;
        timer.Start();  
        data->Purge(labels_out);
        data->Detach(labels_out);
        timer.End();
        fl::logger->Message() << "Finished writing output labels. Time taken to write: " << timer.GetTotalElapsedTimeString().c_str();
      }
      
      if(queries_in == "" || query_labels_in!="") {
        if (queries_in =="") {
	        fl::logger->Message() << "Classification score on the reference set is " <<
	          score;
        } else {
          fl::logger->Message() << "Classification score on the query set is " <<
	          score;
        }
        fl::logger->Message() << "The score per class is ";
        for(std::map<int, int>::iterator it=points_per_class.begin();
            it!=points_per_class.end(); ++it) {
          fl::logger->Message()<<"Class: "<<it->first<<", points: "<<it->second
            <<", score: "<<100*partial_score[it->first];
        }

        if (auc) {
          fl::logger->Message() << "The Area Under the curve (AUC) is "<< auc_score;
          if (roc_out!="") {
            fl::logger->Message()<<"Exporting roc curve to "<<roc_out;
            boost::shared_ptr<typename DataAccessType::DefaultTable_t> roc_table;
            data->Attach(roc_out, 
                std::vector<index_t>(1,2),
                std::vector<index_t>(),
                roc_vec.size(),
                &roc_table);
            typename DataAccessType::DefaultTable_t::Point_t p;
            for(index_t i=0; i<roc_table->n_entries(); ++i) {
              roc_table->get(i, &p);
              p.set(0, roc_vec[i].first);
              p.set(1, roc_vec[i].second);
            }
            data->Purge(roc_out);
            data->Detach(roc_out);
          }
        }
      }
      return 0;    
    }
  }
  else { // r_neighbors >= 0.0
    boost::shared_ptr<typename DataAccessType::DefaultSparseIntTable_t> indices_output_table;
    data->Attach(indices_out,
                 std::vector<index_t>(),
                 std::vector<index_t>(1, reference_table->n_entries()),
                 query_table==NULL?reference_table->n_entries():query_table->n_entries(),
                 &indices_output_table);

    boost::shared_ptr<typename DataAccessType::DefaultSparseDoubleTable_t> dists_output_table;
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
      indices_output_table->get(r_ind_neighbors[i].first, &index_point);
      dists_output_table->get(r_ind_neighbors[i].first, &dist_point);
      index_point.set(r_ind_neighbors[i].second, 1);
      dist_point.set(r_ind_neighbors[i].second, dist_neighbors[i]);
    }

    // Write out to the file.
    if (indices_out != "") {
      fl::logger->Message() << "Emitting index results to " << indices_out;
      timer.Start();
      fl::logger->Message() << "Emitting index results to " << indices_out;
      data->Purge(indices_out);
      data->Detach(indices_out);
      timer.End();
      fl::logger->Message() << "Timer taken to write indices: " << timer.GetTotalElapsedTimeString().c_str();
    }
    if (distances_out != "") {
      fl::logger->Message() << "Emitting distance results to " << distances_out;
      timer.Start();
      data->Purge(distances_out);
      data->Detach(distances_out);
      timer.End();
      fl::logger->Message() << "Time taken to write distances: " << timer.GetTotalElapsedTimeString().c_str();
    }
    if (method=="nnclassification") {
      boost::shared_ptr<typename DataAccessType::template TableVector<index_t> > reference_labels_input_table;
      boost::shared_ptr<typename DataAccessType::template TableVector<index_t> > query_labels_input_table;
      data->Attach(reference_labels_in, &reference_labels_input_table);
      if (query_labels_in!="") {
        data->Attach(query_labels_in, &query_labels_input_table);
      }
      // Allocate the table for writing out the labels.      
      boost::shared_ptr<typename DataAccessType::IntegerTable_t> result_labels_table;
      data->Attach(labels_out,
		   std::vector<index_t>(1, 1),
		   std::vector<index_t>(),
		   indices_output_table->n_entries(),
		   &result_labels_table);
      double score=0;
      std::map<int, int> points_per_class;
      std::map<int, double> partial_score;
      std::vector<std::pair<double, double> > roc_vec;
      double auc_score=0;
      fl::logger->Message() << "Starting the classification task. ";
      timer.Start();
      if (auc==true) {
        std::vector<std::pair<double, double> > roc_vec;
        MNNClassifier::ComputeNNClassification(*indices_output_table,
			      *dists_output_table,
			      *reference_labels_input_table,
            query_labels_in==""?*reference_labels_input_table:*query_labels_input_table,
			      result_labels_table.get(),
	          (queries_in == "") || (query_labels_in!=""),
	 	        auc_label,
            &score,
            &auc_score,
            roc_out==""?NULL:&roc_vec,
            &points_per_class,
            &partial_score);
      } else {
        MNNClassifier::ComputeNNClassification(*indices_output_table,
			      *dists_output_table,
			      *reference_labels_input_table,
            query_labels_in==""?*reference_labels_input_table:*query_labels_input_table,
			      result_labels_table.get(),
	          (queries_in == "") || (query_labels_in!=""),
	 	        auc_label,
            &score,
            NULL,
            NULL,
            &points_per_class,
            &partial_score);
      }
      timer.End();
      fl::logger->Message() << "Finished classifying. Time taken to classify: " << timer.GetTotalElapsedTimeString().c_str();

      if( labels_out != "") {
        fl::logger->Message() << "Writing classification results to " << labels_out;
        timer.Start();  
        data->Purge(labels_out);
        data->Detach(labels_out);
        timer.End();
        fl::logger->Message() << "Finished writing output labels. Time taken to write: " << timer.GetTotalElapsedTimeString().c_str();
      }
      if(queries_in == "" || query_labels_in!="") {
        if (queries_in =="") {
	        fl::logger->Message() << "Classification score on the reference set is " <<
	          score;
        } else {
          fl::logger->Message() << "Classification score on the query set is " <<
	          score;
        }
        fl::logger->Message() << "The score per class is ";
        for(std::map<int, int>::iterator it=points_per_class.begin();
            it!=points_per_class.end(); ++it) {
          fl::logger->Message()<<"Class: "<<it->first<<", points: "<<it->second
            <<", score: "<<100*partial_score[it->first];
        }

        if (auc) {
          fl::logger->Message() << "The Area Under the curve (AUC) is "<< auc_score;
          if (roc_out!="") {
            fl::logger->Message()<<"Exporting roc curve to "<<roc_out;
            boost::shared_ptr<typename DataAccessType::DefaultTable_t> roc_table;
            data->Attach(roc_out, 
                std::vector<index_t>(1,2),
                std::vector<index_t>(),
                roc_vec.size(),
                &roc_table);
            typename DataAccessType::DefaultTable_t::Point_t p;
            for(index_t i=0; i<roc_table->n_entries(); ++i) {
              roc_table->get(i, &p);
              p.set(0, roc_vec[i].first);
              p.set(1, roc_vec[i].second);
            }
            data->Purge(roc_out);
            data->Detach(roc_out);
          }
        }
      }
      return 0;    
    }
  }


  return 0;
}


template<typename DataAccessType, typename BranchType>
int AllKN<boost::mpl::void_>::Main(
  DataAccessType *data,
  const std::vector<std::string> &args
) {
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
    "indices_in",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing neighbor pairs. "
    "if flag --method=classification then you should provide --indices_in"
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
    "reference_labels_in",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing the labels of reference points. "
    "if flag --method=classification then you should provide --reference_labels_in"
  )(
    "query_labels_in",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing the labels of the query points."
  )(
    "labels_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL FILE for outputting the computed nearest neighbor classification "
    "labels (for --method=classification or --method=nnclassification)."
  )(
    "method",
    boost::program_options::value<std::string>()->default_value("nearest"),
    "Which neighbors method to perform.  One of:\n"
    "  nearest: computes nearest neighbors \n"
    "  furthest: computes furthest neighbors \n"
    "  nnclassification: compute  nearest neighbors and do classification\n"
    "  classification: compute the classification scores based on precomputed neighbors" 
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
    "Algorithm used to compute neighbors.  One of:\n"
    "  dual, single"
  )("auc", 
    boost::program_options::value<bool>()->default_value("true"),
    "If this flag is set to true then the classifier computes the "
    "Area Under Curve (AUC) score"
  )("auc_label",
    boost::program_options::value<int>()->default_value(1),
    "label of the class based on which the AUC will be computed"
  )(
    "roc_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file for exporting the ROC curve"
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
    fl::logger->Message() << fl::DISCLAIMER << "\n";
    fl::logger->Message() << desc << "\n";
    return 1;
  }
  
  if(vm["method"].as<std::string>() == "nnclassification" && 
         vm["reference_labels_in"].as<std::string>() == "") {
    fl::logger->Die() << "You need to specify the file that contains the labels.\n";
  }
 
  if(vm["method"].as<std::string>() == "classification" &&
     vm["indices_in"].as<std::string>() == "") {
    fl::logger->Die() << "You need to specify the file that contains the "
      "precomputed neighbor indices.\n";
  }
 
  return BranchType::template BranchOnTable<AllKN<boost::mpl::void_>, DataAccessType>(data, vm);
}

template<typename DataAccessType>
void fl::ml::AllKN<boost::mpl::void_>::Run(
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

}} // namespaces

#endif
