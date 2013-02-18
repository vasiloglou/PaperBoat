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

#ifndef FL_LITE_MLPACK_KDE_KDE_DEFS_H
#define FL_LITE_MLPACK_KDE_KDE_DEFS_H
#include <iostream>
#include "boost/program_options.hpp"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/trim.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/math/special_functions/fpclassify.hpp"
#include "fastlib/base/base.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "mlpack/kde/kde_lscv_function.h"
#include "fastlib/optimization/lbfgs/lbfgs_dev.h"
#include "mlpack/kde/dualtree_dfs_dev.h"
#include "fastlib/util/timer.h"
#include "kde.h"
#include "kde_stat.h"
#include "mlpack/mnnclassifier/auc.h"
#include "fastlib/workspace/task.h"

template<typename TableType1>
template<class DataAccessType>
int fl::ml::Kde<boost::mpl::void_>::Core<TableType1>::Main(
  DataAccessType *data,
  boost::program_options::variables_map &vm) {
  std::vector<std::string> references_in;
  boost::algorithm::split(references_in, vm["references_in"].as<std::string>(), 
      boost::algorithm::is_any_of(",:"));
  int reference_set_count =references_in.size();

  if (reference_set_count > 1) {
    fl::logger->Message() << "Running in KDA mode.";
  }
  else {
    fl::logger->Message() << "Running in KDE mode.";
  }
  boost::shared_ptr<TableType1> queries;
  if (vm.count("queries_in")) {
    fl::logger->Message()<<"Loading query data from " <<
      vm["queries_in"].as<std::string>() <<std::endl;
    std::string queries_in=vm["queries_in"].as<std::string>();
    data->Attach(queries_in, &queries);
  }

  std::string kernel = vm["kernel"].as<std::string>();
  fl::logger->Message() << "Using the " << kernel <<
  " kernel."<<std::endl;

  // Parse the bandwidth argument.
  double  bandwidth =
    (vm.count("bandwidth") > 0) ?
    (vm["bandwidth"].as<double>()) : -1.0;
  if (vm.count("bandwidth") > 0) {
    fl::logger->Message() << "Bandwidth: " << bandwidth<<std::endl;
  } else {
    if (vm.count("bandwidth_selection")>0) {
      fl::logger->Message() << "Doing metric learning."<<std::endl;
      if (kernel == "epan") {
        fl::logger->Die() << "Cannot do metric learning with the Epanechnikov "
        "kernel.";
      }
    }
  }

  std::string bandwidth_selection = (vm.count("bandwidth_selection") > 0) ?
                                  vm["bandwidth_selection"].as<std::string>() : std::string("");

  // Parse the probability argument.
  double probability = vm["probability"].as<double>();
  fl::logger->Message() << "Probability guarantee: " << probability<<std::endl;

  // Parse the relative error argument.
  double relative_error = vm["relative_error"].as<double>();
  fl::logger->Message() << "Relative error guarantee: " << relative_error<<std::endl;

  // Parse the metric argument.
  std::string metric = vm["metric"].as<std::string>();
  fl::logger->Message() << "Using " << metric << " metric"
    <<std::endl;

  // Read in the metric weights argument, if requested.
  boost::shared_ptr<typename DataAccessType::DefaultTable_t> metric_weights;
  if (metric == "weighted_l2") {
    std::string metric_weights_in =
      vm["metric_weights_in"].as<std::string>();
    fl::logger->Message() << "Reading in the metric from " <<
    metric_weights_in << std::endl;
    data->Attach(metric_weights_in, &metric_weights);
  }

  // Read in the dense sparse scale.
//  double dense_sparse_scale=0;
//  if (vm.count("dense_sparse")!=0 || vm.count("dense_categorical")!=0) {
//    dense_sparse_scale = vm["dense_sparse_scale"].as<double>();
//  }

  // Read in the algorithm type.
  std::string algorithm = vm["algorithm"].as<std::string>();

  // Read in the leaf size.
  //int leaf_size = vm["leaf_size"].as<int>();

  // Read in the number of iterations.
  index_t iterations = vm["iterations"].as<index_t>();

  std::vector<double> priors;
  if (vm.count("priors")) {
    std::vector<std::string> tokens ;
    boost::algorithm::split(tokens, vm["priors"].as<std::string>(), 
        boost::algorithm::is_any_of(",:"));
    
    for (int i=0; i<tokens.size(); ++i) {
      try {
        priors.push_back(boost::lexical_cast<double>(tokens[i]));
      }
      catch(const boost::bad_lexical_cast &e) {
        fl::logger->Die()<<"The --priors must be a list of numbers";
      }
    }
  } else {
    priors.resize(reference_set_count);
    std::fill(priors.begin(), priors.end(), 1.0);
  }

  std::vector<double> kda_bandwidths;
  if (vm.count("kda_bandwidths")) {
    std::vector<std::string> tokens ;
    boost::algorithm::split(tokens, vm["kda_bandwidths"].as<std::string>(), 
        boost::algorithm::is_any_of(",:"));
    
    for (int i=0; i<tokens.size(); ++i) {
      try {
        kda_bandwidths.push_back(boost::lexical_cast<double>(tokens[i]));
        if (kda_bandwidths.back()<=0) {
          fl::logger->Die()<<"--kda_bandwidths must contain positive numbers only "
            <<"one of them was ("<<kda_bandwidths.back()<< ") which is not valid";
        }
      }
      catch(const boost::bad_lexical_cast &e) {
        fl::logger->Die()<<"The --kda_bandwidths must be a list of numbers";
      }
    }
  }

  // Read in the number of L-BFGS optimizer restarts.
  index_t num_lbfgs_restarts = vm["num_lbfgs_restarts"].as<index_t>();

  // Read in the number of line searches.
  index_t num_line_searches = vm["num_line_searches"].as<index_t>();

  // Open the table for writing out the KDA labels.
  boost::shared_ptr<typename DataAccessType::template TableVector<index_t> > result_table;
  if (vm.count("result_out") > 0) {
    if (queries.get()==NULL) {
      fl::logger->Die()<<"You asked for --result_out but you haven't provided "
        " a --queries_in option";
    }
    
  }

  
  typename TableType1::template IndexArgs<fl::math::WeightedLMetric < 2,
        fl::data::MonolithicPoint<double> > > w_index_args;
  if (metric=="weighted_l2") {
    typename DataAccessType::DefaultTable_t::Point_t weight_point;
    metric_weights->get(0, &weight_point);
    w_index_args.metric.set_weights(weight_point.template dense_point<double>());
  }

  // first thing to do is load the reference data and index them
  std::vector<boost::shared_ptr<TableType1> > references(reference_set_count);
//  if (vm.count("queries_in")) {
//     if (metric=="l2") {
//      typename TableType1::template IndexArgs<fl::math::LMetric<2> > index_args;
//      index_args.leaf_size = leaf_size;
//      queries->IndexData(index_args);
//    } else {
//      if (metric=="weighted_l2") {
//        w_index_args.leaf_size = leaf_size;
//        queries->IndexData(w_index_args);
//      } else {
//        fl::logger->Die() << "Unknown metric ";     
//      }
//    }
//          
//  }

  for(index_t i=0; i<reference_set_count; ++i) {
    references[i].reset(new TableType1());
    data->Attach(references_in[i], &references[i]);
//    if (metric=="l2") {
//      typename TableType1::template IndexArgs<fl::math::LMetric<2> > index_args;
//      index_args.leaf_size = leaf_size;
//      references[i]->IndexData(index_args);
//    } else {
//      if (metric=="weighted_l2") {
//        w_index_args.leaf_size =leaf_size;
//        references[i]->IndexData(w_index_args);
//      } else {
//        fl::logger->Die() << "Unknown metric "<< metric;     
//      }
//    }
  }


  std::vector<KdeResult<std::vector<double> > > result(reference_set_count);
  if (references.size()==1) {
    if (bandwidth>0) {
      fl::logger->Message() << "Computing KDE";
      if (metric=="l2") {
        if (kernel=="epan") {
          typedef fl::ml::Kde< KdeArgs<fl::math::EpanKernel<double>, 
                   fl::math::LMetric<2>, 
                   KdeStructArgs> > Kde_t;
          // Declare a KDE instance.
          Kde_t kde_instance;
          if (!vm.count("queries_in")) {
            kde_instance.Init(references[0].get(),
                NULL, bandwidth,
                relative_error, probability);
          } else {
            kde_instance.Init(references[0].get(),
                  queries.get(), bandwidth,
                  relative_error, probability);
          }

          // Initialize the dual-tree engine for the KDE instance.
          fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
          dualtree_engine.Init(kde_instance);
          fl::util::Timer timer;
          if (iterations<0) {
            fl::logger->Message() << "Non progressive mode"<<std::endl; 
            
            timer.Start();
            dualtree_engine.Compute(fl::math::LMetric<2>(), &result[0]);
            timer.End();
            fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
          } else {
            fl::logger->Message() << "Progressive mode"<<std::endl;
            typename fl::ml::DualtreeDfs<Kde_t>::template
            iterator<fl::math::LMetric<2> > it = dualtree_engine.get_iterator(
                                       fl::math::LMetric<2>(), &result[0]);
            timer.Start();
            for (int ii = 0; ii < iterations; ii++) {
              ++it;
            }
            timer.End();
            fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."
              <<std::endl;
            // Tell the iterator that we are done using it so that the
            // result can be finalized.
            it.Finalize();
          }
        } else {
          if (kernel=="gaussian") {
            typedef fl::ml::Kde< KdeArgs<fl::math::GaussianKernel<double>, 
                   fl::math::LMetric<2>, 
                   KdeStructArgs> > Kde_t;
            // Declare a KDE instance.
            Kde_t kde_instance;
            if (!vm.count("queries_in")) {
              kde_instance.Init(references[0].get(),
                  NULL, bandwidth,
                  relative_error, probability);
            } else {
              kde_instance.Init(references[0].get(),
                  queries.get(), bandwidth,
                  relative_error, probability);
            }
            // Initialize the dual-tree engine for the KDE instance.
            fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
            dualtree_engine.Init(kde_instance);
            fl::util::Timer timer;
            // Gaussian Kernel
            if (iterations<0) {
              fl::logger->Message() << "Non progressive mode"<<std::endl; 
              timer.Start();
              dualtree_engine.Compute(fl::math::LMetric<2>(), &result[0]);
              timer.End();
              fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
            } else {
              fl::logger->Message() << "Progressive mode"<<std::endl;
              typename fl::ml::DualtreeDfs<Kde_t>::template
              iterator<fl::math::LMetric<2> > it = dualtree_engine.get_iterator(
                                         fl::math::LMetric<2>(), &result[0]);
              timer.Start();
              for (int ii = 0; ii < iterations; ii++) {
                ++it;
              }
              timer.End();
              fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
              // Tell the iterator that we are done using it so that the
              // result can be finalized.
              it.Finalize();
            }
          } else {
            fl::logger->Die() << "Unknown kernel "<<kernel;
          }
        } 
      } else {
        // weighted l2
        if (metric=="weighted_l2") {
          if (kernel=="epan") {
            typedef fl::ml::Kde< KdeArgs<fl::math::EpanKernel<double>, 
                   fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> >, 
                   KdeStructArgs> > Kde_t;
  
            // Declare a KDE instance.
            Kde_t kde_instance;
            if (!vm.count("queries_in")) {
              kde_instance.Init(references[0].get(),
                  NULL, bandwidth,
                  relative_error, probability);
            } else {
               kde_instance.Init(references[0].get(),
                  queries.get(), bandwidth,
                  relative_error, probability);
            }

            // Initialize the dual-tree engine for the KDE instance.
            fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
            dualtree_engine.Init(kde_instance);
            fl::util::Timer timer;

            if (iterations<0) {
              fl::logger->Message() << "Non progressive mode"<<std::endl; 
              timer.Start();
              dualtree_engine.Compute(w_index_args.metric, &result[0]);
              timer.End();
              fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
            } else {
              fl::logger->Message() << "Progressive mode"<<std::endl;
              typename fl::ml::DualtreeDfs<Kde_t>::template
              iterator<fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double>  > > it 
              = dualtree_engine.get_iterator(w_index_args.metric, &result[0]);
              timer.Start();
              for (index_t ii = 0; ii < iterations; ii++) {
                ++it;
              }
              timer.End();
              fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds.";
              // Tell the iterator that we are done using it so that the
              // result can be finalized.
              it.Finalize();
            }
          } else {
            if (kernel=="gaussian") {
              // Gaussian Kernel
              typedef fl::ml::Kde< KdeArgs<fl::math::GaussianKernel<double>, 
                 fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> >, 
                 KdeStructArgs> > Kde_t;
  
              // Declare a KDE instance.
              Kde_t kde_instance;
              if (!vm.count("queries_in")) {
                kde_instance.Init(references[0].get(),
                    NULL, bandwidth,
                    relative_error, probability);
              } else {
                kde_instance.Init(references[0].get(),
                   queries.get(), bandwidth,
                   relative_error, probability);
              }
  
              // Initialize the dual-tree engine for the KDE instance.
              fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
              dualtree_engine.Init(kde_instance);
              fl::util::Timer timer;

              if (iterations<0) {
                fl::logger->Message() << "Non progressive mode"<<std::endl; 
                timer.Start();
                dualtree_engine.Compute(w_index_args.metric, &result[0]);
                timer.End();
                fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
              } else {
                fl::logger->Message() << "Progressive mode"<<std::endl;
                typename fl::ml::DualtreeDfs<Kde_t>::template
                iterator<fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> > > it = dualtree_engine.get_iterator(
                    w_index_args.metric, &result[0]);
                timer.Start();
                for (index_t ii = 0; ii < iterations; ii++) {
                  ++it;
                }
                timer.End();
                fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds.";
                // Tell the iterator that we are done using it so that the
                // result can be finalized.
                it.Finalize();
              }
            } else {
              fl::logger->Die() << "Unknown kernel "<<kernel;
            }
          }
        } else {
          fl::logger->Die() << "Unknown metric" << metric ;
        }
      }
      // Collect the result now
      if (vm["densities_out"].as<std::string>()!="") {
        boost::shared_ptr<
          typename DataAccessType::DefaultTable_t > densities;
        std::string densities_out=vm["densities_out"].as<std::string>();
        data->Attach(densities_out,
          std::vector<index_t>(1,1),
          std::vector<index_t>(),
          vm.count("queries_in")?queries->n_entries():references[0]->n_entries(),
          &densities);

        result[0].template GetDensities<1>(densities.get());

        data->Purge(densities_out);
        data->Detach(densities_out);
      }
    } else {
      if (kernel=="epan") {
        fl::logger->Die() << 
          "Bandwidth learning is not supported for the Epanechnikov kernel";
      } else {
        if (kernel=="gaussian") {
          std::pair< fl::data::MonolithicPoint<double>, double >
          global_min_point_iterate;
          if (metric=="l2") {
            // Choose the starting (plugin) bandwidth in log scale.
            typedef fl::ml::KdeLscvFunction< KdeArgs<fl::math::GaussianKernel<double>, 
                    fl::math::LMetric<2>, KdeStructArgs> > FunctionType;
            typedef fl::ml::Kde< KdeArgs<fl::math::GaussianKernel<double>, 
                 fl::math::LMetric<2>, KdeStructArgs> > Kde_t;
              // Declare a KDE instance.
              Kde_t kde_instance;
              kde_instance.Init(references[0].get(),
                   NULL, bandwidth,
                   relative_error, probability);
 
            FunctionType lscv_function;
            lscv_function.Init(kde_instance, fl::math::LMetric<2>(), result[0]);
            bandwidth = log(lscv_function.plugin_bandwidth());

            // Initialize the vector to be optimized.
            fl::data::MonolithicPoint<double> lscv_result;
            lscv_result.Init(index_t(1));
            lscv_result[0] = bandwidth;

            // Initialize the LBFGS optimizer and optimize.
            fl::ml::Lbfgs< FunctionType > lbfgs_engine;
            lbfgs_engine.Init(lscv_function, 1);
	            lbfgs_engine.set_max_num_line_searches(num_line_searches);

            // Let us just do 10 re-starts.
            global_min_point_iterate.first.Init(index_t(1));
            global_min_point_iterate.first[0] = lscv_result[0];
            global_min_point_iterate.second = std::numeric_limits<double>::max();

            if (bandwidth_selection == "monte_carlo") {
              fl::logger->Warning() << "Monte Carlo method might be unreliable for "
                 "reporting the least squares score for low bandwidths.";
	              fl::logger->Warning() << "Use a large value of relative error for "
	                 "stable answers.";
              for (index_t i = 0; i < num_lbfgs_restarts; i++) {
                fl::logger->Message() << "LBFGS Restart number: " << i;
                lbfgs_engine.Optimize(-1, &lscv_result);
                const std::pair< fl::data::MonolithicPoint<double>, double >
                    &min_point_iterate = lbfgs_engine.min_point_iterate();
                if (global_min_point_iterate.second > min_point_iterate.second) {
                  global_min_point_iterate.first.CopyValues(min_point_iterate.first);
                  global_min_point_iterate.second = min_point_iterate.second;
                  lscv_result[0] = min_point_iterate.first[0];
                  fl::logger->Message() << "The optimal bandwidth currently is " <<
                      exp(global_min_point_iterate.first[0]) << " with LSCV score of " <<
                      min_point_iterate.second;
                } else {
                  break;
                }
              }
            } 
          } else {
            if (metric=="weighted_l2") {
              // Choose the starting (plugin) bandwidth in log scale.
              typedef fl::ml::KdeLscvFunction< KdeArgs<fl::math::GaussianKernel<double>, 
                  fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> >, KdeStructArgs> > FunctionType;
              FunctionType lscv_function;
              typedef fl::ml::Kde< KdeArgs<fl::math::GaussianKernel<double>, 
                 fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> >, KdeStructArgs> > Kde_t;
              // Declare a KDE instance.
              Kde_t kde_instance;
              kde_instance.Init(references[0].get(),
                 NULL, bandwidth,
                 relative_error, probability);
 
              lscv_function.Init(kde_instance, w_index_args.metric, result[0]);
              bandwidth = log(lscv_function.plugin_bandwidth());
              // Initialize the vector to be optimized.
              fl::data::MonolithicPoint<double> lscv_result;
              lscv_result.Init(index_t(1));
              lscv_result[0] = bandwidth;
              // Initialize the LBFGS optimizer and optimize.
              fl::ml::Lbfgs<FunctionType> lbfgs_engine;
              lbfgs_engine.Init(lscv_function, 1);
              lbfgs_engine.set_max_num_line_searches(num_line_searches);

              // Let us just do 10 re-starts.
              global_min_point_iterate.first.Init(index_t(1));
              global_min_point_iterate.first[0] = lscv_result[0];
              global_min_point_iterate.second = std::numeric_limits<double>::max();

              if (bandwidth_selection == "monte_carlo") {
                fl::logger->Warning() << "Monte Carlo method might be unreliable for "
                    "reporting the least squares score for low bandwidths.";
	              fl::logger->Warning() << "Use a large value of relative error for "
	                  "stable answers.";
                for (index_t i = 0; i < num_lbfgs_restarts; i++) {
                  fl::logger->Message() << "LBFGS Restart number: " << i;
                  lbfgs_engine.Optimize(-1, &lscv_result);
                  const std::pair< fl::data::MonolithicPoint<double>, double >
                        &min_point_iterate = lbfgs_engine.min_point_iterate();
                  if (global_min_point_iterate.second > min_point_iterate.second) {
                    global_min_point_iterate.first.CopyValues(min_point_iterate.first);
                    global_min_point_iterate.second = min_point_iterate.second;
                    lscv_result[0] = min_point_iterate.first[0];
                    fl::logger->Message() << "The optimal bandwidth currently is " <<
                      exp(global_min_point_iterate.first[0]) << " with LSCV score of " <<
                      min_point_iterate.second;
                  } else {
                    break;
                  }
                }
              }
            } else {
              fl::logger->Die()<<"Unknown metric "<<metric;
            }
          }
          double optimal_bandwidth = exp(global_min_point_iterate.first[0]);
          fl::logger->Message() << "The optimal bandwidth is " <<
                optimal_bandwidth<<std::endl;
          boost::shared_ptr<typename DataAccessType::DefaultTable_t> bandwidth_table;
          if (vm.count("bandwidth_out")>0) {
            data->Attach(vm["bandwidth_out"].as<std::string>(),
                std::vector<index_t>(1,1),
                std::vector<index_t>(),
                1,
                &bandwidth_table);
            bandwidth_table->set(0, 0, optimal_bandwidth);
            data->Purge(bandwidth_table->filename());
            data->Detach(bandwidth_table->filename());
          }

          // Collect the result now
          if (vm.count("densities_out")) {
            boost::shared_ptr<
              typename DataAccessType::DefaultTable_t> densities;
            std::string densities_out=vm["densities_out"].as<std::string>();
            data->Attach(densities_out,
              std::vector<index_t>(1,1),
              std::vector<index_t>(),
              vm.count("queries_in")?queries->n_entries():references[0]->n_entries(),
              &densities);

            result[0].template GetDensities<1>(densities.get());

            data->Purge(densities_out);
            data->Detach(densities_out);
          }
        } else {
          fl::logger->Die() <<"Uknown kernel " <<kernel;
        }
      }
    }
  } else {
    if (vm.count("kda_bandwidths")==0) {
      fl::logger->Die()<<"You chose to run KDE but you haven't provided"
        " the --kda_bandwidths option";
    }
    double total_accuracy=0;
    double total_points=0;
    std::vector<index_t> points_per_class;
    std::vector<double> partial_scores;
    std::vector<double> a_class_scores;
    std::vector<double> b_class_scores;
    index_t auc_label=vm["auc_label"].as<index_t>();
    bool auc=vm["auc"].as<bool>();

    // KDA
    fl::logger->Message() << "Running KDA"<<std::endl;
    if (vm.count("queries_in")) {
      // Prepare to collect density results
      std::vector<boost::shared_ptr<typename DataAccessType::DefaultTable_t>  >densities(reference_set_count);
      std::string density_prefix;
      if (vm.count("densities_prefix_out")>0) {
        density_prefix = vm["densities_prefix_out"].as<std::string>();
      } else {
        density_prefix = data->GiveTempVarName();
      }
      int32 density_num;
      if (vm.count("densities_num_out")>0) {
        density_num = vm["densities_num_out"].as<int32>();
        if (density_num!=reference_set_count) {
          fl::logger->Die()<<"--densities_num_out ("<< density_num
              <<") must be equal to --references_in ("
              <<reference_set_count<<")";
        }
      } else {
        density_num=reference_set_count;
      }

      for(index_t i=0; i<reference_set_count; ++i) {
        densities[i].reset(new typename DataAccessType::DefaultTable_t());
        
        std::string new_name = data->GiveFilenameFromSequence(density_prefix, i);
        data->Attach(new_name,
            std::vector<index_t>(1,1),
            std::vector<index_t>(),
            queries->n_entries(),
            &densities[i]);
      }
      if (metric=="l2") {
        if (kernel=="epan") {
          typedef fl::ml::Kde< KdeArgs<fl::math::EpanKernel<double>, 
                  fl::math::LMetric<2>, 
                  KdeStructArgs> > Kde_t;

          // Declare a KDE instance.
          Kde_t kde_instance;
          fl::util::Timer timer;
          if (iterations<0) {
            fl::logger->Message() << "Non progressive mode"<<std::endl; 
            timer.Start();
            for(index_t i=0; i<references.size(); ++i) {
               kde_instance.Init(references[i].get(),
                  queries.get(), kda_bandwidths[i],
                  relative_error, probability);
               // Initialize the dual-tree engine for the KDE instance.
               fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
               dualtree_engine.Init(kde_instance);
               dualtree_engine.Compute(fl::math::LMetric<2>(), &result[i]);
               
            }
            timer.End();
            fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
          } else {
            fl::logger->Message() << "Progressive mode"<<std::endl;
            timer.Start();
            for(index_t i=0; i<references.size(); ++i) {
              kde_instance.Init(references[i].get(),
                  queries.get(), kda_bandwidths[i],
                  relative_error, probability);
              // Initialize the dual-tree engine for the KDE instance.
              fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
              dualtree_engine.Init(kde_instance);
              typename fl::ml::DualtreeDfs<Kde_t>::template
              iterator<fl::math::LMetric<2> > it = dualtree_engine.get_iterator(
                                       fl::math::LMetric<2>(), &result[i]);
              for (index_t ii = 0; ii < iterations; ii++) {
                ++it;
              }
              // Tell the iterator that we are done using it so that the
              // result can be finalized.
              it.Finalize();
            }
            timer.End();
            fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
         }
        } else {
          if (kernel=="gaussian") {
            typedef fl::ml::Kde< KdeArgs<fl::math::GaussianKernel<double>, 
                fl::math::LMetric<2>, 
                KdeStructArgs> > Kde_t;
            // Declare a KDE instance.
            Kde_t kde_instance;
            fl::util::Timer timer;
            if (iterations<0) {
              fl::logger->Message() << "Non progressive mode"<<std::endl; 
              timer.Start();
              for(index_t i=0; i<references.size(); ++i) {
                 kde_instance.Init(references[i].get(),
                    queries.get(), kda_bandwidths[i],
                    relative_error, probability);
            

                 // Initialize the dual-tree engine for the KDE instance.
                 fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                 dualtree_engine.Init(kde_instance);
                 dualtree_engine.Compute(fl::math::LMetric<2>(), &result[i]);
              }
              timer.End();
              fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
            } else {
              fl::logger->Message() << "Progressive mode"<<std::endl;
              timer.Start();
              for(index_t i=0; i<references.size(); ++i) {
                kde_instance.Init(references[i].get(),
                    queries.get(), kda_bandwidths[i],
                    relative_error, probability);
                // Initialize the dual-tree engine for the KDE instance.
                fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                dualtree_engine.Init(kde_instance);
                typename fl::ml::DualtreeDfs<Kde_t>::template
                iterator<fl::math::LMetric<2> > it = dualtree_engine.get_iterator(
                                       fl::math::LMetric<2>(), &result[i]);
                for (index_t ii = 0; ii < iterations; ii++) {
                  ++it;
                }
                // Tell the iterator that we are done using it so that the
                // result can be finalized.
                it.Finalize();
              }
              timer.End();
              fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds.";
            }
          } else {
            fl::logger->Die() << "Unknown kernel ";
          }
        }
      } else {
        // weighted l2
        if (metric=="weighted_l2") {
          if (kernel=="epan") {
            typedef fl::ml::Kde< KdeArgs<fl::math::EpanKernel<double>, 
                fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> >, 
                KdeStructArgs> > Kde_t;
            // Declare a KDE instance.
            Kde_t kde_instance;
            fl::util::Timer timer;
            if (iterations<0) {
              fl::logger->Message() << "Non progressive mode"<<std::endl; 
              timer.Start();
              for(index_t i=0; i<references.size(); ++i) {
                 kde_instance.Init(references[i].get(),
                    queries.get(), kda_bandwidths[i],
                    relative_error, probability);
              
  
                 // Initialize the dual-tree engine for the KDE instance.
                 fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                 dualtree_engine.Init(kde_instance);
                 dualtree_engine.Compute(w_index_args.metric, &result[i]);
              }
              timer.End();
              fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
            } else {
              fl::logger->Message() << "Progressive mode"<<std::endl;
              timer.Start();
              for(index_t i=0; i<references.size(); ++i) {
                kde_instance.Init(references[i].get(),
                    queries.get(), kda_bandwidths[i],
                    relative_error, probability);
                // Initialize the dual-tree engine for the KDE instance.
                fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                dualtree_engine.Init(kde_instance);

                typename fl::ml::DualtreeDfs<Kde_t>::template
                iterator<fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> > > it = dualtree_engine.get_iterator(
                                         w_index_args.metric, &result[i]);
                for (index_t ii = 0; ii < iterations; ii++) {
                  ++it;
                  // Tell the iterator that we are done using it so that the
                  // result can be finalized.
                }
                it.Finalize();
              }
              timer.End();
              fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds.";
            }
          } else {
            if (kernel=="gaussian") {
              typedef fl::ml::Kde< KdeArgs<fl::math::GaussianKernel<double>, 
                  fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> >, 
                  KdeStructArgs> > Kde_t;
              // Declare a KDE instance.
              Kde_t kde_instance;
              fl::util::Timer timer;
              if (iterations<0) {
                fl::logger->Message() << "Non progressive mode"<<std::endl; 
                timer.Start();
                for(index_t i=0; i<references.size(); ++i) {
                   kde_instance.Init(references[i].get(),
                      queries.get(), kda_bandwidths[i],
                      relative_error, probability);
              
  
                   // Initialize the dual-tree engine for the KDE instance.
                   fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                   dualtree_engine.Init(kde_instance);
                   dualtree_engine.Compute(fl::math::LMetric<2>(), &result[i]);
                }
                timer.End();
                fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
              } else {
                fl::logger->Message() << "Progressive mode"<<std::endl;
                timer.Start();
                for(index_t i=0; i<references.size(); ++i) {
                  kde_instance.Init(references[i].get(),
                      queries.get(), kda_bandwidths[i],
                      relative_error, probability);
                  // Initialize the dual-tree engine for the KDE instance.
                  fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                  dualtree_engine.Init(kde_instance);
                  typename fl::ml::DualtreeDfs<Kde_t>::template
                  iterator<fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> > > it = dualtree_engine.get_iterator(
                                         w_index_args.metric, &result[i]);
                  for (index_t ii = 0; ii < iterations; ii++) {
                    ++it;
                  }
                  // Tell the iterator that we are done using it so that the
                  // result can be finalized.
                  it.Finalize();
                }
                timer.End();
                fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() 
                  << "seconds."<<std::endl;
              }
            } else {
              fl::logger->Die() << "Unknown kernel ";
            }
          }
        } else {
          fl::logger->Die() << "Unknown metric" << metric ;
        }
      }
      
      for(index_t i=0; i<reference_set_count; ++i) {
        result[i].template GetDensities<1>(densities[i].get());
        std::string new_name = data->GiveFilenameFromSequence(density_prefix, i);
        data->Purge(new_name);
        data->Detach(new_name);
      }
      boost::shared_ptr<typename DataAccessType::IntegerTable_t> query_labels;
      bool compute_score=false;
      if (vm["queries_labels_in"].as<std::string>()!="") {
        data->Attach(vm["queries_labels_in"].as<std::string>(), &query_labels);
        compute_score=true;
      } else {
        fl::logger->Warning()<<"You haven't provided --queries_labels_in, I "
          "make the assumption that query labels might be embedded on the "
          "--queries_in table. If not just ignore the scores";
        query_labels.reset(new typename DataAccessType::IntegerTable_t());
        query_labels->Init("",
            std::vector<index_t>(1,1),
            std::vector<index_t>(),
            queries->n_entries());
         typename TableType1::Point_t point;
         for(index_t i=0; i<queries->n_entries(); ++i) {
           queries->get(i, &point);
           query_labels->set(i, 0, point.meta_data(). template get<0>());
         }
         compute_score=true;
      }

      // we are going to compute the winning labels
      std::vector<index_t> winning_classes;
      ComputeScoresForAuc(
        (index_t)references.size(),
        queries->n_entries(),
        auc_label,
        query_labels,
        0,
        result,
        priors,
        auc,
        compute_score,
        &partial_scores,
        &points_per_class,
        &a_class_scores,
        &b_class_scores,
        &winning_classes,
        &total_accuracy);
      // Copy over to the table so that it can be exported.
      if (vm.count("result_out") > 0) {
        fl::logger->Message() << "Emitting the class labels for queries"<<std::endl;
        boost::shared_ptr<typename DataAccessType::IntegerTable_t > result_out;
        data->Attach(vm["result_out"].as<std::string>(),
            std::vector<index_t>(1, 1),
            std::vector<index_t>(),
            queries->n_entries(),
            &result_out);
        typename DataAccessType::IntegerTable_t::Point_t point;
        for (index_t i = 0; i < queries->n_entries(); i++) {
          result_out->get(i, &point);
          point.set(0, winning_classes[i]);
        }
        data->Purge(vm["result_out"].as<std::string>());
        data->Detach(vm["result_out"].as<std::string>());
      } else {
        fl::logger->Warning() << "You didn't specify --result_out, so no class labels will be "
          "exported"<<std::endl;
      }
      if (compute_score==true) {
        total_points=queries->n_entries();
        fl::logger->Message() << "Classification score on query set "
          <<"(total_points="<< total_points <<")" "set is " <<
	            100.0*total_accuracy/total_points<<"\%";
        fl::logger->Message() << "The score per class is ";
        for(index_t ii=0; ii<points_per_class.size(); ++ii) {
          fl::logger->Message()<<"Class: "<<ii<<", points: "<<points_per_class[ii]
              <<", score: "<<100*partial_scores[ii]/points_per_class[ii]<<"\%";
        }
      }
      if (auc==true && compute_score==true) {
        double auc_score;
        bool compute_roc=false;
        if (vm["roc_out"].as<std::string>()!="") {
          compute_roc=true;
        }
        if (compute_roc==true) {
          boost::shared_ptr<typename DataAccessType::DefaultTable_t> roc;
          std::string file=vm["roc_out"].as<std::string>();
          std::vector<std::pair<double, double> > roc_vec;
          fl::ml::ComputeAUC(a_class_scores, b_class_scores, &auc_score, &roc_vec);
          fl::logger->Message()<<"Exporting roc curve to "<<file;
          data->Attach(file, 
              std::vector<index_t>(1,2),
              std::vector<index_t>(),
              roc_vec.size(),
              &roc);
          typename DataAccessType::DefaultTable_t::Point_t p;
          for(index_t i=0; i<roc->n_entries(); ++i) {
            roc->get(i, &p);
            p.set(0, roc_vec[i].first);
            p.set(1, roc_vec[i].second);
          }
          data->Purge(file);
          data->Detach(file);
        } else {
          fl::ml::ComputeAUC(a_class_scores, b_class_scores, &auc_score);
        }
        fl::logger->Message() << "The Area Under the curve (AUC) is "<< auc_score;
      }
    } else {
      // LOOCV KDA 
      std::string density_prefix="";
      if (vm.count("densities_prefix_out")>0) {
        density_prefix = vm["densities_prefix_out"].as<std::string>();
      } 
      int32 density_num=0;
      if (vm.count("densities_num_out")>0) {
        density_num = vm["densities_num_out"].as<int32>();
        if (density_num!=reference_set_count * reference_set_count) {
          fl::logger->Die()<<"--densities_num_out ("<< density_num
              <<") must be equal to --references_in squared ("
              <<reference_set_count * reference_set_count<<")";
        }
      } else {
        density_num=reference_set_count;
      }

      fl::logger->Message() << "Running LOOCV KDA"<<std::endl;
      for(index_t k=0; k<references.size(); ++k) {
        result.clear();
        result.resize(reference_set_count);
        if (metric=="l2") {
          if (kernel=="epan") {
            typedef fl::ml::Kde< KdeArgs<fl::math::EpanKernel<double>, 
                    fl::math::LMetric<2>, 
                    KdeStructArgs> > Kde_t;
  
            // Declare a KDE instance.
            Kde_t kde_instance;
            fl::util::Timer timer;
            if (iterations<0) {
              fl::logger->Message() << "Non progressive mode"<<std::endl; 
              timer.Start();
              for(index_t i=0; i<references.size(); ++i) {
                 if (k==i) { 
                   kde_instance.Init(references[i].get(),
                      NULL, kda_bandwidths[i],
                      relative_error, probability);
                 } else {
                   kde_instance.Init(references[i].get(),
                      references[k].get(), kda_bandwidths[i],
                      relative_error, probability);
                 }
                 // Initialize the dual-tree engine for the KDE instance.
                 fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                 dualtree_engine.Init(kde_instance);
                 dualtree_engine.Compute(fl::math::LMetric<2>(), &result[i]);
                 
              }
              timer.End();
              fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
            } else {
              fl::logger->Message() << "Progressive mode"<<std::endl;
              timer.Start();
              for(index_t i=0; i<references.size(); ++i) {
                if (k==i) {
                  kde_instance.Init(references[i].get(),
                     NULL, kda_bandwidths[i],
                     relative_error, probability);
                } else {
                  kde_instance.Init(references[i].get(),
                     references[k].get(), kda_bandwidths[i],
                     relative_error, probability);

                }
                // Initialize the dual-tree engine for the KDE instance.
                fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                dualtree_engine.Init(kde_instance);
                typename fl::ml::DualtreeDfs<Kde_t>::template
                iterator<fl::math::LMetric<2> > it = dualtree_engine.get_iterator(
                                         fl::math::LMetric<2>(), &result[i]);
                for (index_t ii = 0; ii < iterations; ii++) {
                  ++it;
                }
                // Tell the iterator that we are done using it so that the
                // result can be finalized.
                it.Finalize(); 
              }
              timer.End();
              fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
           }
          } else {
            if (kernel=="gaussian") {
              typedef fl::ml::Kde< KdeArgs<fl::math::GaussianKernel<double>, 
                  fl::math::LMetric<2>, 
                  KdeStructArgs> > Kde_t;
              // Declare a KDE instance.
              Kde_t kde_instance;
              fl::util::Timer timer;
              if (iterations<0) {
                fl::logger->Message() << "Non progressive mode"<<std::endl; 
                timer.Start();
                for(index_t i=0; i<references.size(); ++i) {
                   if (k==i) {
                     kde_instance.Init(references[i].get(),
                       NULL, kda_bandwidths[i],
                       relative_error, probability);
                   } else {
                      kde_instance.Init(references[i].get(),
                        references[k].get(), kda_bandwidths[i],
                        relative_error, probability);
                   }
              
                   // Initialize the dual-tree engine for the KDE instance.
                   fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                   dualtree_engine.Init(kde_instance);
                   dualtree_engine.Compute(fl::math::LMetric<2>(), &result[i]);
                }
                timer.End();
                fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
              } else {
                fl::logger->Message() << "Progressive mode"<<std::endl;
                timer.Start();
                for(index_t i=0; i<references.size(); ++i) {
                  if (k==i) {
                    kde_instance.Init(references[i].get(),
                        NULL, kda_bandwidths[i],
                        relative_error, probability);
                  } else {
                    kde_instance.Init(references[i].get(),
                       references[k].get(), kda_bandwidths[i],
                       relative_error, probability);
                 
                  }
                  // Initialize the dual-tree engine for the KDE instance.
                  fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                  dualtree_engine.Init(kde_instance);

                  typename fl::ml::DualtreeDfs<Kde_t>::template
                  iterator<fl::math::LMetric<2> > it = dualtree_engine.get_iterator(
                                         fl::math::LMetric<2>(), &result[i]);
                  for (index_t ii = 0; ii < iterations; ii++) {
                    ++it;
                  }
                  // Tell the iterator that we are done using it so that the
                  // result can be finalized.
                  it.Finalize();
                }
                timer.End();
                fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds.";
              }
            } else {
              fl::logger->Die() << "Unknown kernel ";
            }
          }
        } else {
          // weighted l2
          if (metric=="weighted_l2") {
            if (kernel=="epan") {
              typedef fl::ml::Kde< KdeArgs<fl::math::EpanKernel<double>, 
                  fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> >, 
                  KdeStructArgs> > Kde_t;
              // Declare a KDE instance.
              Kde_t kde_instance;
              fl::util::Timer timer;
              if (iterations<0) {
                fl::logger->Message() << "Non progressive mode"<<std::endl; 
                timer.Start();
                for(index_t i=0; i<references.size(); ++i) {
                   if (k==i) {
                     kde_instance.Init(references[i].get(),
                       NULL, bandwidth,
                       relative_error, probability);
                   } else {
                     kde_instance.Init(references[i].get(),
                       references[k].get(), kda_bandwidths[i],
                       relative_error, probability);
                   }
                   // Initialize the dual-tree engine for the KDE instance.
                   fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                   dualtree_engine.Init(kde_instance);
                   dualtree_engine.Compute(w_index_args.metric, &result[i]);
                }
                timer.End();
                fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
              } else {
                fl::logger->Message() << "Progressive mode"<<std::endl;
                timer.Start();
                for(index_t i=0; i<references.size(); ++i) {
                  if (k==i) {
                     kde_instance.Init(references[i].get(),
                        NULL,  kda_bandwidths[i],
                        relative_error, probability);
                  } else {
                    kde_instance.Init(references[i].get(),
                        references[k].get(), kda_bandwidths[i],
                        relative_error, probability);
                  }
                  // Initialize the dual-tree engine for the KDE instance.
                  fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                  dualtree_engine.Init(kde_instance);

                  typename fl::ml::DualtreeDfs<Kde_t>::template
                  iterator<fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> > > it = dualtree_engine.get_iterator(
                                           w_index_args.metric, &result[i]);
                  for (int ii = 0; ii < iterations; ii++) {
                    ++it;
                    // Tell the iterator that we are done using it so that the
                    // result can be finalized.
                    it.Finalize();
                  }
                }
                timer.End();
                fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds.";
              }
            } else {
              if (kernel=="gaussian") {
                typedef fl::ml::Kde< KdeArgs<fl::math::GaussianKernel<double>, 
                    fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> >, 
                    KdeStructArgs> > Kde_t;
                // Declare a KDE instance.
                Kde_t kde_instance;
                fl::util::Timer timer;
                if (iterations<0) {
                  fl::logger->Message() << "Non progressive mode"<<std::endl; 
                  timer.Start();
                  for(index_t i=0; i<references.size(); ++i) {
                    if (k==i) {
                      kde_instance.Init(references[i].get(),
                        NULL, bandwidth,
                        relative_error, probability);
                    } else {
                      kde_instance.Init(references[i].get(),
                         references[k].get(), kda_bandwidths[i],
                         relative_error, probability);              
                    }
                    // Initialize the dual-tree engine for the KDE instance.
                    fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                    dualtree_engine.Init(kde_instance);
                    dualtree_engine.Compute(fl::math::LMetric<2>(), &result[i]);
                  }
                  timer.End();
                  fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds."<<std::endl;
                } else {
                  fl::logger->Message() << "Progressive mode"<<std::endl;
                  timer.Start();
                  for(int i=0; i<references.size(); ++i) {
                    if (i==k) {
                      kde_instance.Init(references[i].get(),
                        NULL, bandwidth,
                        relative_error, probability);
                    } else {
                      kde_instance.Init(references[i].get(),
                        references[k].get(), kda_bandwidths[i],
                        relative_error, probability);
                    }
                    // Initialize the dual-tree engine for the KDE instance.
                    fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
                    dualtree_engine.Init(kde_instance); 
                    typename fl::ml::DualtreeDfs<Kde_t>::template
                    iterator<fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> > > it = dualtree_engine.get_iterator(
                                           w_index_args.metric, &result[i]);
                    for (index_t ii = 0; ii < iterations; ii++) {
                      ++it;
                    }
                    // Tell the iterator that we are done using it so that the
                    // result can be finalized.
                    it.Finalize();
                  }
                  timer.End();
                  fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() 
                    << "seconds."<<std::endl;
                }
              } else {
                fl::logger->Die() << "Unknown kernel ";
              }
            }
          } else {
            fl::logger->Die() << "Unknown metric" << metric ;
          }
        }
        // we are going to compute the winning labels
        double local_accuracy=0;
        std::vector<double> local_a_class_scores;
        std::vector<double> local_b_class_scores;
        std::vector<index_t> winning_classes;
        boost::shared_ptr<typename DataAccessType::IntegerTable_t> temp;
        ComputeScoresForAuc(
            (index_t)references.size(),
            references[k]->n_entries(),
            auc_label,
            temp,
            k,
            result,
            priors,
            auc,
            true,
            &partial_scores,
            &points_per_class,
            &local_a_class_scores,
            &local_b_class_scores,
            &winning_classes,
            &local_accuracy);

            a_class_scores.insert(a_class_scores.end(), 
            local_a_class_scores.begin(), local_a_class_scores.end());

        b_class_scores.insert(b_class_scores.end(), 
            local_b_class_scores.begin(), local_b_class_scores.end());
        total_accuracy+=local_accuracy;
        total_points+=references[k]->n_entries();
        local_accuracy=local_accuracy/references[k]->n_entries();
        partial_scores.push_back(local_accuracy);
        points_per_class.push_back(references[k]->n_entries());
        fl::logger->Message() << "Checking class "<<k<<std::endl;
        fl::logger->Message()<<"Class: "<<k<<", points: "
          <<references[k]->n_entries()
          <<", score: "<<100*local_accuracy<<"\%";
        if (density_prefix!="") {
          for (int32 i=0; i<references.size(); ++i) {
            std::string name=data->GiveFilenameFromSequence(density_prefix, k*references.size()+i);
            boost::shared_ptr<typename DataAccessType::DefaultTable_t> densities_table;
            data->Attach(name, 
                std::vector<index_t>(1, 1),
                std::vector<index_t>(),
                references[k]->n_entries(),
                &densities_table);
            for(index_t l=0; l<result[i].densities_.size(); ++l) {
              densities_table->set(l, 0, result[i].densities_[l]);
            }

            data->Purge(name);
            data->Detach(name);
          }
        }
      }
      fl::logger->Message() << "Classification score on all the references "
        <<"("<< total_points <<")" "set is " <<
	          100.0*total_accuracy/total_points<<"\%";
      
      if (auc) {
        double auc_score;
        bool compute_roc=false;
        if (vm["roc_out"].as<std::string>()!="") {
          compute_roc=true;
        }
        if (compute_roc==true) {
          boost::shared_ptr<typename DataAccessType::DefaultTable_t> roc;
          std::string file=vm["roc_out"].as<std::string>();
          std::vector<std::pair<double, double> > roc_vec;
          fl::ml::ComputeAUC(a_class_scores, b_class_scores, &auc_score, &roc_vec);
          fl::logger->Message()<<"Exporting roc curve to "<<file;
          data->Attach(file, 
              std::vector<index_t>(1,2),
              std::vector<index_t>(),
              roc_vec.size(),
              &roc);
          typename DataAccessType::DefaultTable_t::Point_t p;
          for(index_t i=0; i<roc->n_entries(); ++i) {
            roc->get(i, &p);
            p.set(0, roc_vec[i].first);
            p.set(1, roc_vec[i].second);
          }
          data->Purge(file);
          data->Detach(file);
        } 
        if (auc) {
          fl::ml::ComputeAUC(a_class_scores, b_class_scores, &auc_score);
          fl::logger->Message() << "The Area Under the curve (AUC) is "<< auc_score;
        }
      }
    } 
  }
  return 1;
}

template<typename DataAccessType, typename BranchType>
int fl::ml::Kde<boost::mpl::void_>::Main(
  DataAccessType *data,
  const std::vector<std::string> &args) {
  FL_SCOPED_LOG(Kde);
  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>(),
    "REQUIRED a colon separated list with files containing reference data, "
    "if you want to run kda, you can "
    "provide more than one references files. "
    "Each reference file will contain files from the same class. "
    "ie for a two class problem:  --references_in=a_class.csv:b_class.csv"
  )(
    "queries_in",
    boost::program_options::value<std::string>(),
    "OPTIONAL file containing query positions.  If omitted, KDE computes "
    "the leave-one-out density at each reference point."
  )(
    "queries_labels_in", 
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing the labels of the query points. If you provide "
    "labels for the query points then an accuracy score of KDA will be generated "
    "based on these labels"
  )(
    "densities_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file to store computed densities."
  )(
    "densities_prefix_out", 
    boost::program_options::value<std::string>(),
    "OPTIONAL prefix for outputing densities in kda mode"
  )(
    "densities_num_out",
    boost::program_options::value<int32>(),
    "OPTIONAL number of densities files to use"
  )(
    "kernel",
    boost::program_options::value<std::string>()->default_value("epan"),
    "Kernel function used by KDE.  One of:\n"
    "  epan, gaussian"
  )(
    "bandwidth",
    boost::program_options::value<double>(),
    "REQUIRED (if --bandwidth_selection is not set) kernel bandwidth, "
    "if you set --bandwidth_selection flag, "
    "then the --bandwidth will be ignored."
  )(
    "kda_bandwidths",
    boost::program_options::value<std::string>(),
    "REQUIRED a colon or comma separated list of the bandwidths for every class, "
    "ie for a two class problem --kda_bandwidths=0.4,1.3"
  )(
    "bandwidth_selection",
    boost::program_options::value<std::string>(),
    "OPTIONAL The method used for optimizing the bandwidth."
    "Available options: plugin, monte_carlo"
  )(
    "bandwidth_out",
    boost::program_options::value<std::string>(),
    "OPTIONAL, if you want to save the optimal bandwidth in a file then "
    "you must set this flag"
  )(
    "auc",
    boost::program_options::value<bool>()->default_value(true),
    "OPTIONAL if this flag is set true then it computes the "
    "Area Under Curve (AUC) score"
  )(
    "auc_label",
    boost::program_options::value<index_t>()->default_value(0),
    "OPTIONAL in a multiclass problem where we want to compute the "
    "Area Under Curve (AUC), this label defines the class against all others"
  )(
    "roc_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file for exporting the ROC curve"
  )(
    "probability",
    boost::program_options::value<double>()->default_value(1.0),
    "Probability guarantee for the approximation of KDE."
  )(
    "relative_error",
    boost::program_options::value<double>()->default_value(0.01),
    "Relative error for the approximation of KDE."
  )(
    "metric",
    boost::program_options::value<std::string>()->default_value("l2"),
    "Metric function used by KDE.  One of:\n"
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
    boost::program_options::value<std::string>()->default_value("dual"),
    "Algorithm used to compute densities.  One of:\n"
    "  dual, single"
  )(
    "tree",
    boost::program_options::value<std::string>()->default_value("kdtree"),
    "Tree structure used by KDE.  One of:\n"
    "  kdtree, balltree"
  )(
    "iterations",
    boost::program_options::value<index_t>()->default_value(-1),
    "KDE can run in either batch or progressive mode.  If --iterations=i "
    "is omitted, KDE computes approximatesly to completion; otherwise, "
    "it terminates after i progressive refinements."
  )(
    "num_lbfgs_restarts",
    boost::program_options::value<index_t>()->default_value(1),
    "OPTIONAL The number of restarts for L-BFGS optimizer used for Monte "
    "Carlo-based bandwidth optimizer."
  )(
    "num_line_searches",
    boost::program_options::value<index_t>()->default_value(5),
    "OPTIONAL The number of line seaches used for L-BFGS optimizer."
  )(
    "result_out",
    boost::program_options::value<std::string>(),
    "A file to export the results of KDA. KDA is a classification method "
    "So it outputs the labels of the winning class for every query point. "
    "Labels are enumerated in an ascending order, so if you set the following flags "
    "--references_in=classA.txt,classB.txt,classC.txt "
    "then KDA will classify each query point according to the maximum KDE score with respect "
    "to each reference file. If KDE for point i computed on classC.txt has the higher score, then "
    "the ith row of the result_out file will be 2, if the highest KDE score is classA.txt then it will be 0 "
    "The ordering of the classes depends on the way they appear in the command line, so the first "
    "references_in that appears  will be class 0, the next class 1 and so on"
  )(
    "priors",
    boost::program_options::value<std::string>(),
    "OPTIONAL This is the priors for using KDA, It is a comma or colon separated list"
    "If you omit it, then the program assumes equal priors for every class. "
    "For example --priors=0.2,0.8"
  );

  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(args);
  clp.style(boost::program_options::command_line_style::default_style
     ^boost::program_options::command_line_style::allow_guessing);
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
     fl::logger->Die() << e.what()
      <<" . This option will be ignored";
  }
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    fl::logger->Message() << fl::DISCLAIMER << "\n";
    fl::logger->Message() << desc << "\n";
    return true;
  }
  // Validate the  Only immediate dying is allowed here, the
  // parsing is done later.
  if (vm.count("references_in") == 0) {
    fl::logger->Die() << "Missing required --references_in";
  } else {
    std::vector<std::string> tokens1;
    boost::algorithm::split(tokens1, vm["references_in"].as<std::string>(), 
        boost::algorithm::is_any_of(",:"));
    if (tokens1.size() > 1) {
      // This is running in KDA mode, so check whether if we have an
      // equal number of prior.
      if (vm.count("priors")>0) {
        std::vector<std::string> tokens2;
        boost::algorithm::split(tokens2, vm["priors"].as<std::string>(), 
            boost::algorithm::is_any_of(",:"));
        if (tokens1.size() !=tokens2.size()) {
          fl::logger->Die() << "The number of priors need to equal the number "
          "of classes of reference sets.";
        }
      }
    }
    if (vm.count("kda_bandwidths")>0) {
      std::vector<std::string> tokens3;
      boost::algorithm::split(tokens3, vm["kda_bandwidths"].as<std::string>(), 
            boost::algorithm::is_any_of(",:"));
      if (tokens1.size() !=tokens3.size()) {
        fl::logger->Die() << "The number of --kda_bandwidths need to equal the number "
          "of classes of reference sets.";
      }
    }
  }

  if (vm["kernel"].as<std::string>() != "gaussian" &&
      vm["kernel"].as<std::string>() != "epan") {
    fl::logger->Die() << "We support only epan or gaussian for the kernel.";
  }
  if (vm.count("bandwidth") > 0 && vm["bandwidth"].as<double>() <= 0) {
      fl::logger->Die() << "The --bandwidth requires a positive real number";
  }
  if (vm.count("bandwidth") == 0 && vm.count("bandwidth_selection") == 0
      && vm.count("kda_bandwidths")==0) {
    fl::logger->Die() << "One of --bandwidth_selection, "
    << "--bandwidth, kda_bandwidths must be set";
  }
  if (vm.count("bandwidth") == 0 && vm.count("kda_bandwidths")==0 &&
      vm["bandwidth_selection"].as<std::string>() != "plugin" &&
    vm["bandwidth_selection"].as<std::string>() != "monte_carlo") {
    fl::logger->Die() << "The --bandwidth_selection takes the value of "
      << "either plugin or monte_carlo.";
  }
  if (vm["probability"].as<double>() <= 0 ||
      vm["probability"].as<double>() > 1) {
    fl::logger->Die() << "The --probability requires a real number "
    "$0 < p <= 1$ ";
  }
  if (vm["relative_error"].as<double>() < 0) {
    fl::logger->Die() << "The --relative_error requires a real number $r >= 0$";
  }
  if (vm["metric"].as< std::string >() != "l2" &&
      vm["metric"].as< std::string >() != "weighted_l2") {
    fl::logger->Die() << "The --metric requires either l2 or weighted_l2.";
  }
  if (vm["algorithm"].as<std::string>() != "dual") {
    fl::logger->Die() << "The --algorithm supports: dual";
  }
  if( vm["num_lbfgs_restarts"].as<index_t>() < 0) {
    fl::logger->Die() << "The --num_lbfgs_restarts needs to be a non-negative integer.";
  }
  if( vm["num_line_searches"].as<index_t>() <= 0) {
    fl::logger->Die() << "The --num_line_searches needs to be a positive integer.";
  }
  if (vm.count("help")) {
    std::cout << fl::DISCLAIMER << "\n";
    std::cout << desc << "\n";
    return 1;
  }
  return BranchType::template BranchOnTable<Kde<boost::mpl::void_>, DataAccessType>(data, vm);
}

template<typename DataAccessType>
void fl::ml::Kde<boost::mpl::void_>::Run(
    DataAccessType *data,
    const std::vector<std::string> &args) {
  fl::ws::Task<
    DataAccessType,
    &Main<
      DataAccessType, 
      typename DataAccessType::Branch_t
    > 
  >task(data, args);
  data->schedule(task); 
}

#endif  
