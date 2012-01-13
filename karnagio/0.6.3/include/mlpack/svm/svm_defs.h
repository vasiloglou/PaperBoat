/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/
f
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
#ifndef FL_LITE_MLPACK_SVM_SVM_DEFS_H
#define FL_LITE_MLPACK_SVM_SVM_DEFS_H

#include <boost/mpl/if.hpp>
#include "fastlib/util/timer.h"
#include "svm.h"
#include "fastlib/workspace/task.h"

namespace fl {
  namespace ml {
    template<typename TableType>
    template<class DataAccessType>
    int Svm<boost::mpl::void_>::Core<TableType>::Main(
        DataAccessType *data,
        boost::program_options::variables_map &vm) {
      FL_SCOPED_LOG(Svm);
      std::string task = vm["run_mode"].as<std::string>();
      std::string parameters_in = vm["parameters_in"].as<std::string>();

      // CHECK THAT INPUT VALUES/COMBINATIONS MAKE SENSE
      if(task == "train") {
        if (vm["references_in"].as<std::string>() == "") {
          fl::logger->Die() << "Missing required --references_in";
          return 1;
        }
        
        if(vm["parameters_out"].as<std::string>() == "") {
          fl::logger->Warning() << "--parameters_out not provided, parameters like C and Bias will not be saved.";
        }
        /*if(vm["support_vectors_out"].as<std::string>() == "") {
        fl::logger->Warning() << "--support_vectors_out not provided, support vectors will not be saved.";
        }*/
      } else if(task == "eval") {
        if (vm["queries_in"].as<std::string>() == "") {
          fl::logger->Die() << "Missing required --queries_in for testing.";
          return 1;
        }
        if (vm["results_out"].as<std::string>() == "") {
          fl::logger->Die() << "--results_out is required in eval mode.";
        }
        if(parameters_in == "") {
          fl::logger->Die() << "Missing required --parameters_in not provided.";
        }
      } else {
        fl::logger->Die() << "Unknown task. Please see usage with --help option.";
      }
      std::string kernel_type = vm["kernel"].as<std::string>();
      // I will read kernel info from parameters input
      double bandwidth = -1;
      bandwidth = vm["bandwidth"].as<double>();
      if(kernel_type == "linear") {
        if(task == "eval" && vm["weights_in"].as<std::string>() == "") {
          fl::logger->Die() << "For linear kernel you must provide --weights_in.";
        }
        if(task == "train" && vm["weights_out"].as<std::string>() == "") {
          fl::logger->Warning() << "--weights_out not provided. Model weights will not be output.";
        }
      } else {
        if(task == "eval" && vm["alphas_in"].as<std::string>()  == "") {
          fl::logger->Die() << "Missing required --alphas_in, not provided.";
        }
        if(task=="eval" && vm["queries_in"].as<std::string>() == "") {
          fl::logger->Die() << "Missing required --queries_in, not provided."; 
        }
        if(task == "train" && vm["alphas_out"].as<std::string>() == "") {
          fl::logger->Warning() << "--alphas_out not provided, alphas's will not be saved.";
        }
        if(task == "train" && vm["support_vectors_out"].as<std::string>() == "") {
          fl::logger->Warning() << "--support_vectors_out not provided, support vectors will not be saved.";
        }
      }
      if((kernel_type != "linear" || task != "eval") 
           && vm["references_in"].as<std::string>() == "") {
        fl::logger->Die() << "Missing required --references_in not provided.";
      }

      // all settings are finalized
      boost::shared_ptr<TableType> reference_table;
      std::string references_in = vm["references_in"].as<std::string>();
      boost::shared_ptr<TableType> query_table;
      std::string queries_in = vm["queries_in"].as<std::string>();
      // no need to read references in case of linear kernel evaluation
      if(kernel_type != "linear" || task != "eval") {
        fl::logger->Message() << "Importing references... "<<std::endl;
        data->Attach(references_in, &reference_table);
      }
      if (task=="eval") {
        fl::logger->Message() << "Importing queries... "<<std::endl;
        data->Attach(queries_in, &query_table);
      }

      if (vm["labels_in"].as<std::string>()!="") {
        boost::shared_ptr<typename DataAccessType::template TableVector<signed char> > labels;
        data->Attach(vm["labels_in"].as<std::string>(), &labels);
        if (task=="train") {
          boost::shared_ptr<TableType> reference_table1;
          data->template TieLabels<0>(reference_table, labels, data->GiveTempVarName(), &reference_table1);
          typename TableType::template IndexArgs<fl::math::LMetric<2> > index_args;
          // at some point we should change that and do the indexing
          // according to the diameter. This is implemented but not tested yet
          index_args.leaf_size=20;
          reference_table1->IndexData(index_args);
          reference_table=reference_table1;
        } else {
          if (task=="eval") {
            boost::shared_ptr<TableType> query_table1;
            data->template TieLabels<0>(query_table, labels, data->GiveTempVarName(), &query_table1);
            query_table=query_table1;     
          }
        }
      } 

      index_t iterations=vm["iterations"].as<index_t>();
      double accuracy=vm["accuracy"].as<double>();
      double regularization=vm["regularization"].as<double>();
      double bandwidth_overload_factor=vm["bandwidth_overload_factor"].as<double>();
      if(kernel_type == "linear") {
        if(vm["bandwidth"].as<double>() > 0) {
          fl::logger->Warning() << "Linear Kernels do not support bandwidth. Ignoring --bandwidth.";
        }
        fl::logger->Message() << "Using Linear Kernel.";
      } else {
        if(kernel_type == "gaussian") {
          fl::logger->Message() << "Using Gaussian Kernel with bandwidth: " << bandwidth;
          fl::math::GaussianDotProduct<double, fl::math::LMetric<2> > kernel;
          fl::math::LMetric<2> metric;
          kernel.Init(bandwidth, metric);
          if (task=="train") {
            typename Svm<TableType>::template Trainer<fl::math::GaussianDotProduct<
                typename TableType::CalcPrecision_t, fl::math::LMetric<2> >  > trainer;
            trainer.set_reference_table(reference_table.get());
            trainer.set_regularization(regularization);
            trainer.set_iterations(iterations);
            trainer.set_accuracy(accuracy);
            trainer.set_kernel(kernel);
            trainer.set_bandwidth_overload_factor(bandwidth_overload_factor);
            std::map<index_t, double> support_vectors;
            boost::shared_ptr<TableType> reduced_table;
            trainer.Train(&support_vectors, &reduced_table);
            std::string alphas_out=vm["alphas_out"].as<std::string>();
            std::string support_vectors_out=vm["support_vectors_out"].as<std::string>();
            boost::shared_ptr<typename DataAccessType::DefaultTable_t> alphas_table;
            if (alphas_out!="" || support_vectors_out!="") {
              if (alphas_out!="") {
                data->Attach(alphas_out, 
                    std::vector<index_t>(1,1),
                    std::vector<index_t>(),
                    support_vectors.size(),
                    &alphas_table);
              }
              boost::shared_ptr<TableType> sv_table;
              if (support_vectors_out!="") {
                data->Attach(support_vectors_out, 
                    reference_table->dense_sizes(),
                    reference_table->sparse_sizes(),
                    support_vectors.size(),
                    &sv_table);
              }
                 
              std::map<index_t, double>::const_iterator it;
              typename DataAccessType::DefaultTable_t::Point_t a_point;
              typename TableType::Point_t p1, p2;
              for(it=support_vectors.begin(); it!=support_vectors.end(); ++it) {
                if (alphas_out!="") {
                  alphas_table->get(it->first, &a_point);
                  a_point.set(0, it->second);
                }
                if (support_vectors_out!="") {
                  reduced_table->get(it->first, &p1);
                  sv_table->get(it->first, &p2);
                  p1.CopyValues(p2);
                }
              }
            }
            if (alphas_out!="") {
              data->Purge(alphas_out);
              data->Detach(alphas_out);
            }
            if (support_vectors_out!="") {
              data->Purge(support_vectors_out);
              data->Detach(support_vectors_out);
            }
          } else if(kernel_type == "polynomial") { 
            fl::logger->Message() << "Using Polynomial Kernel.";
          } else {
            fl::logger->Die() << "Unknown Kernel.";
          }
        } else {
          if (task=="eval") {
            typename Svm<TableType>::template Predictor<fl::math::GaussianDotProduct<
                  typename TableType::CalcPrecision_t, fl::math::LMetric<2> >  > predictor;
            predictor.set_query_table(query_table.get());
            predictor.set_support_vectors(reference_table.get());
            fl::math::GaussianDotProduct<
                  typename TableType::CalcPrecision_t, fl::math::LMetric<2> >   kernel;
            kernel.set(vm["bandwidth"].as<double>());
            predictor.set_kernel(kernel);
            boost::shared_ptr<typename DataAccessType::DefaultTable_t> alphas_table;
            data->Attach(vm["alphas_in"].as<std::string>(), &alphas_table);
            std::vector<double> alphas;
            typename DataAccessType::DefaultTable_t::Point_t point;
            for(index_t i=0;i<alphas_table->n_entries(); ++i) {
              alphas_table->get(i, &point);
              alphas.push_back(point[0]);
            }
            predictor.set_alphas(&alphas);
            std::vector<double> margins;
            double accuracy;
            std::string accuracy_out=vm["accuracy_out"].as<std::string>();
            if (accuracy_out!="") {
              predictor.Predict(&margins, NULL);
            } else {
              predictor.Predict(&margins, &accuracy);
              fl::logger->Message()<<"Accuracy on the test set: "<<accuracy<<"100%%"<<std::endl;
              boost::shared_ptr<typename DataAccessType::DefaultTable_t> accuracy_table;
              std::string accuracy_out=vm["prediction_accuracy_out"].as<std::string>();
              data->Attach(accuracy_out, 
                  std::vector<index_t>(1,1),
                  std::vector<index_t>(), 
                  1,
                  &accuracy_table);
              typename DataAccessType::DefaultTable_t::Point_t point;
              accuracy_table->get(0, &point);
              point.set(0, accuracy);
              data->Purge(accuracy_out);
              data->Detach(accuracy_out);
            }
            boost::shared_ptr<typename DataAccessType::DefaultTable_t> margins_table;
            std::string margins_out=vm["margins_out"].as<std::string>();
            if (margins_out!="") {
              data->Attach(margins_out, 
                    std::vector<index_t>(1,1),
                    std::vector<index_t>(), 
                    query_table->n_entries(),
                    &margins_table);
              typename DataAccessType::DefaultTable_t::Point_t point;
              for(index_t i=0; i<margins_table->n_entries(); ++i) {
                margins_table->get(i, &point);
                point.set(0, margins[i]);
              }
              data->Purge(margins_out);
              data->Detach(margins_out);
            }
          }
        }
      }
      return 1;
    }

    template<class DataAccessType, typename BranchType>
    int Svm<boost::mpl::void_>::Main(
       DataAccessType *data,
        const std::vector<std::string> &args) {
      boost::program_options::options_description desc("Available options");
      desc.add_options()(
        "help", "Print this information."
        )(
        "iterations",
        boost::program_options::value<index_t>()->default_value(10),
        "Number of iterations for SMO"
        )(
        "margins_out", 
        boost::program_options::value<std::string>()->default_value(""),
        "The file to store the margins in when task is eval."
        )(
        "alphas_out", 
        boost::program_options::value<std::string>()->default_value(""),
        "The file to store the generated model's alphas when task is one of train or train-test."
        )(
        "support_vectors_out", 
        boost::program_options::value<std::string>()->default_value(""),
        "The file to store the generated model's support vectors when the task is  one of train or train-test."
        )(
        "alphas_in", 
        boost::program_options::value<std::string>()->default_value(""),
        "The file containing the alpha coefficients when task is one of test or train-test."
        )(
        "weights_in", 
        boost::program_options::value<std::string>()->default_value(""),
        "The file containing the weights for the Linear SVM Model."
        )(
        "weights_out", 
        boost::program_options::value<std::string>()->default_value(""),
        "The file to which the weights to the linear SVM will be output."
        )(
        "parameters_in", 
        boost::program_options::value<std::string>()->default_value(""),
        "The file containing the parameters when task is one of test or train-test."
        )(
        "run_mode", 
        boost::program_options::value<std::string>()->default_value("train"),
        "One of train or eval"
        )(
        "regularization", 
        boost::program_options::value<double>()->default_value(1),
        "SVM regularization parameter C. It must be greater than 0."
        )(
        "bandwidth", 
        boost::program_options::value<double>()->default_value(0),
        "Sigma (bandwidth) for Gaussian kernel."
        )(
        "bandwidth_overload_factor", 
        boost::program_options::value<double>()->default_value(3),
        "When it does sampling it searches for a wider region to pick points"
        )(
        "queries_in", 
        boost::program_options::value<std::string>()->default_value(""),
        "file containing the test points when task is one of test or train-test."
        )(
        "kernel", 
        boost::program_options::value<std::string>()->default_value("gaussian"),
        "The kernel to use. One of gaussian, linear."
        )(
        "references_in", 
        boost::program_options::value<std::string>(),
        "REQUIRED. File containing the training points when task is one of train or train-test."
        )(
        "labels_in",
        boost::program_options::value<std::string>()->default_value(""),
        "OPTIONAL file containing the labels of points. "
        "if it is missing it assumes it is coming from the references_in file"
        )(
        "accuracy",
        boost::program_options::value<double>()->default_value(1e-3),
        "OPTIONAL. the precision for evaluating the KKT conditions"
        )(
        "prediction_accuracy_out",
        boost::program_options::value<std::string>()->default_value(""),
        "OPTIONAL. the file to store the predictin accuracy"
        )(
        "log",
        boost::program_options::value<std::string>()->default_value(""),
        "A file to receive the log, or omit for stdout."
        )(
        "loglevel",
        boost::program_options::value<std::string>()->default_value("debug"),
        "Level of log detail.  One of:\n."
        "  debug: log everything\n."
        "  verbose: log messages and warnings\n."
        "  warning: log only warnings\n."
        "  silent: no logging."
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

      return BranchType::template BranchOnTable<Svm<boost::mpl::void_>, DataAccessType>(data, vm);
    }

    template<typename DataAccessType>
    void Svm<boost::mpl::void_>::Run(
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
