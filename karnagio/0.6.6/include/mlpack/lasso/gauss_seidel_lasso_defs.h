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
#ifndef FL_LITE_MLPACK_LASSO_GAUSSEIDEL_LASSO_DEFS_H
#define FL_LITE_MLPACK_LASSO_GAUSSEIDEL_LASSO_DEFS_H

#include <iostream>
#include "boost/program_options.hpp"
#include "fastlib/base/base.h"
#include "gauss_seidel_lasso.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/util/string_utils.h"

namespace fl {
namespace ml {

template<typename DataAccessType, typename BranchType>
int GaussSeidelLasso<boost::mpl::void_>::Main(
  DataAccessType *data,
  const std::vector<std::string> &args) {
  ////////// READING PARAMETERS AND LOADING DATA /////////////////////
  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Display help on ridge regression")
  ("references_in", boost::program_options::value<std::string>(),
   "data file containing the predictors and the predictions")
  ("regularization", boost::program_options::value<double>(),
   "The regularization parameter")
  ("target_column",
   boost::program_options::value<index_t> (),
   "The feature with the specified prefix is chosen to be the "
   "prediction index. If missing, it will default to the zero-th "
   "feature "
  )("remove_column",
   boost::program_options::value< std::vector< std::string> >(),
   "A comma separated list of column indices, each of which denotes columns that  should be"
   " removed from the consideration of predictor set. We remove one "
   "feature for each prefix."
  )("violation_tolerance",
    boost::program_options::value<double>()->default_value(1e-4),
    "if a violation is less than this value  then it is considered as zero"
  )("gradient_tolerance",
    boost::program_options::value<double>()->default_value(0.01),
    "if the percentage change of the gradient norm is less than this value then "
    "the algorithm terminates"
  )("iterations",
    boost::program_options::value<index_t>()->default_value(100),
    "maximum number of iterations to execute optimization"
  )("coefficients_out",
   boost::program_options::value< std::string >()->default_value("coefficients.txt"),
   "The output file for coefficients. The --target_column will be zero "
   "in the --coefficients_out table. The last element is the bias term"
  )("initial_coefficients",
   boost::program_options::value<index_t>()->default_value(1),
   "0: ridge regression solution, 1: zero vector"
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
   std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args, "");
   boost::program_options::command_line_parser clp(args1);
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
   catch ( const boost::program_options::error &e) {
     fl::logger->Die() << e.what();
   } 
   boost::program_options::notify(vm);
   if (vm.count("help")) {
     fl::logger->Message() << fl::DISCLAIMER << "\n";
     fl::logger->Message() << desc << "\n";
     return EXIT_SUCCESS;
   }


  if (vm.count("references_in") == 0) {
    fl::logger->Die() << "You need to provide a reference set to work on.";
  }
  if (vm.count("regularization") == 0 ||
      vm["regularization"].as<double>() <= 0) {
    fl::logger->Die() << "The regularization parameter should be a positive number.";
  }

  return BranchType::template BranchOnTable<GaussSeidelLasso<boost::mpl::void_>, DataAccessType>(data, vm);
}


template<typename TableType>
template<typename DataAccessType>
int GaussSeidelLasso<boost::mpl::void_>::Core<TableType>::Main(DataAccessType *data,
    boost::program_options::variables_map &vm) {

  FL_SCOPED_LOG(Lasso);
  // Read in the table containing both the predictions and the
  // predictors.
  std::string file_name = vm["references_in"].as<std::string>();

  // Setup the indices you want to use for the predictors
  // and the predictions index.
  std::vector<index_t> predictor_indices;
  index_t prediction_index = -1;

  // Setup the table and get the dataset.
  boost::shared_ptr<TableType> table;
  data->Attach(file_name, &table);
  std::vector<std::string> features = table->data()->labels();
  if (features.size()==0) {
    for(index_t i=0; i<table->n_attributes(); ++i) {
     features.push_back(boost::lexical_cast<std::string>(i));
    }
  }

  // Parse the index prefixes that must be removed.
  std::vector<bool> removed_features;
  removed_features.resize(table->n_attributes());
  for (index_t i = 0; i < static_cast<index_t>(removed_features.size()); i++) {
    removed_features[i] = false;
  }

  if (vm.count("remove_column")) {
    const std::vector< std::string > remove_indexes =
      fl::SplitString(vm["remove_column"].as<std::string>(), ",");
    for (index_t j = 0; j < static_cast<index_t>(remove_indexes.size()); j++) {
      removed_features[boost::lexical_cast<index_t>(remove_indexes[j])] = true;
    }
  }

  // Parse the index that must be used for prediction.
  if (vm.count("target_column")) {
    prediction_index = vm["target_column"].as<index_t>();
  }
  else {
    fl::logger->Message() << "Using the default prediction index of 0.\n";
    prediction_index = 0;
  }

  for (index_t i = 0; i < table->n_attributes(); i++) {
    if (i != prediction_index && (! removed_features[i])) {
      predictor_indices.push_back(i);
    }
    if (removed_features[i]) {
      fl::logger->Message() <<  "Pruned " << i;
    }
  }
  // now sort the predictors (they should already be sorted)
  std::sort(predictor_indices.begin(), predictor_indices.end());
  fl::logger->Message() << "Prediction index: " << prediction_index;

  // The LASSO model to be computed.
  fl::ml::LassoModel<TableType> model;
  
  double violation_tolerance=vm["violation_tolerance"].as<double>();
  double gradient_tolerance=vm["gradient_tolerance"].as<double>();
  index_t iterations=vm["iterations"].as<index_t>();
  // Perform the LASSO computation
  double lambda = vm["regularization"].as<double>();
  fl::ml::GaussSeidelLasso<TableType>::template Compute<double, true>(
    *table, violation_tolerance, gradient_tolerance, iterations, predictor_indices, prediction_index, lambda, &model);

  // Output the final model to the screen.
  //model.PrintCoefficients("model.txt");
  boost::shared_ptr<typename DataAccessType::DefaultTable_t> coeff_table;
  if (vm.count("coefficients_out")) {
    fl::logger->Message()<<"Finished computations"<<std::endl;
    std::string coefficients_out = vm["coefficients_out"].as<std::string>();
    fl::logger->Message()<<"Exporting coefficients to "<< coefficients_out<<std::endl;
    data->Attach(coefficients_out,
                 std::vector<index_t>(1, 1),
                 std::vector<index_t>(),
                 table->n_attributes() + 1,
                 &coeff_table);
    model.Export(coeff_table.get());
    data->Purge(coefficients_out);
    data->Detach(coefficients_out);
  }

  return 0;
}

};
};

#endif
