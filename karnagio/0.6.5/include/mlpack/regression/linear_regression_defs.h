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
#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_DEFS_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_DEFS_H

#include "mlpack/regression/linear_regression.h"
#include "mlpack/regression/linear_regression_lic_dev.h"
#include "mlpack/regression/linear_regression_model_dev.h"
#include "mlpack/regression/vif_prune_dev.h"
#include "mlpack/regression/stepwise_regression_dev.h"
#include "mlpack/regression/correlation_prune_dev.h"
#include "fastlib/workspace/task.h"

template<typename TableType1>
template<typename Dataset_t>
void fl::ml::LinearRegression<boost::mpl::void_>::Core<TableType1>::
FindIndexWithPrefix_(
  const Dataset_t &dataset, const char *prefix,
  std::deque<int> &remove_indices,
  std::vector< std::string > *remove_feature_names,
  std::deque<int> *additional_remove_indices,
  bool keep_going_after_first_match) {

  // Get the dataset information containing the feature types and
  // names.
  const std::vector<std::string> &features = dataset.labels();

  for (int i = 0; i < (int) features.size(); i++) {

    // If a feature name with the desired prefix has been found, then
    // make sure it hasn't been selected before. If so, then add to
    // the remove indices.
    const std::string &feature_name = features[i];

    if (!strncmp(prefix, feature_name.c_str(), strlen(prefix) - 1)) {

      bool does_not_exist_yet = true;
      for (int j = 0; j < remove_indices.size(); j++) {
        if (remove_indices[j] == (int) i) {
          does_not_exist_yet = false;
          break;
        }
      }
      if (additional_remove_indices != NULL && does_not_exist_yet) {
        for (int j = 0; j < additional_remove_indices->size(); j++) {
          if ((*additional_remove_indices)[j] == (int) i) {
            does_not_exist_yet = false;
            break;
          }
        }
      }
      if (does_not_exist_yet) {
        fl::logger->Message() << "Found: " << feature_name.c_str() <<
        " at position " << i;
        remove_indices.push_back(i);

        if (remove_feature_names != NULL) {
          remove_feature_names->push_back(feature_name);
        }

        if (!keep_going_after_first_match) {
          break;
        }
      }
    }
  }
}

template<typename TableType1>
void fl::ml::LinearRegression<boost::mpl::void_>::Core<TableType1>
::SetupIndices_(
  TableType1 &table,
  const std::vector< std::string > &remove_index_prefixes,
  const std::vector< std::string > &prune_predictor_index_prefixes,
  const std::string &prediction_index_prefix,
  std::deque<int> *predictor_indices,
  std::deque<int> *prune_predictor_indices,
  std::vector< std::string > *prune_predictor_feature_names,
  int *prediction_index) {

  const typename TableType1::Dataset_t &initial_dataset = *(table.data());

  // Now examine each feature name of the dataset, and construct the
  // indices.
  std::deque<int> remove_indices;
  fl::logger->Message() << "Remove index:";
  for (int i = 0; i < (int) remove_index_prefixes.size(); i++) {
    fl::logger->Message() << remove_index_prefixes[i].c_str();
    FindIndexWithPrefix_(
      initial_dataset,
      remove_index_prefixes[i].c_str(),
      remove_indices,
      (std::vector<std::string> *) NULL,
      (std::deque<int> *) NULL, false);
  }

  if (prediction_index_prefix != "") {
    std::deque<int> prediction_index_arraylist;
    fl::logger->Message() << "Prediction index: "<< prediction_index_prefix.c_str();
    FindIndexWithPrefix_(initial_dataset, prediction_index_prefix.c_str(),
                         prediction_index_arraylist,
                         (std::vector<std::string > *) NULL,
                         (std::deque<int> *) NULL, true);
    if (prediction_index_arraylist.size() == 0) {
      try {
        int index=boost::lexical_cast<int>(prediction_index_prefix);
        prediction_index_arraylist.push_back(index);
        if (index<0 || index>=table.n_attributes()) {
          fl::logger->Die()<<"Prediction index ("<<index<<") must be within [0, "
            <<table.n_attributes()<<"]";
        }
      }
      catch(const boost::bad_lexical_cast &e) {
        fl::logger->Die() << "The prediction index prefix " <<
        prediction_index_prefix << " does not exist.";
      }
    }
    *prediction_index = prediction_index_arraylist[0];
  }
  else {
    fl::logger->Message() << "Using the default prediction index of 0.";
    *prediction_index = 0;
  }

  for (int i = 0; i < table.n_attributes(); i++) {
    bool to_be_removed = false;
    for (int j = 0; j < remove_indices.size(); j++) {
      if (remove_indices[j] == i) {
        to_be_removed = true;
        break;
      }
    }
    if (!to_be_removed && i != (*prediction_index)) {
      predictor_indices->push_back(i);
    }
  }

  fl::logger->Message() << "Prune index:";
  if (prune_predictor_index_prefixes.size() > 0) {
    for (int i = 0; i < (int) prune_predictor_index_prefixes.size();
         i++) {
    //  fl::logger->Message() << prune_predictor_index_prefixes[i].c_str();
      FindIndexWithPrefix_(initial_dataset,
                           prune_predictor_index_prefixes[i].c_str(),
                           *prune_predictor_indices,
                           prune_predictor_feature_names,
                           &remove_indices, true);
    }
  }

  // If the user has not specified the pruning indices explicitly,
  // it becomes the same set as the predictor index set.
  else {
    const std::vector<std::string> &features = initial_dataset.labels();
    for (int i = 0; i < (int) predictor_indices->size(); i++) {
      if ((*predictor_indices)[i] >= 0) {
        prune_predictor_indices->push_back((*predictor_indices)[i]);

        if (features.size() > 0) {
          prune_predictor_feature_names->push_back(
            features[(*predictor_indices)[i]]);
        }
        else {
          char buffer[20];
          sprintf(buffer, "%d", i);
          std::string auto_feature(buffer);
          prune_predictor_feature_names->push_back(
            std::string(auto_feature));
        }
      }
    }
  }
}

template<typename TableType1>
void fl::ml::LinearRegression<boost::mpl::void_>::Core<TableType1>::
ApplyEqualityConstraints_(
  const TableType1 &equality_constraints_table,
  TableType1 *reference_table) {

  // Loop through each constraint and apply the appropriate scaling.
  std::vector<bool> attributes_present_in_equality_constraints(
    equality_constraints_table.n_attributes(), false);

  for (int i = 0; i < equality_constraints_table.n_entries(); i++) {

    // Get the constraint.
    typename TableType1::Point_t equality_constraint;
    equality_constraints_table.get(i, &equality_constraint);

    // Locate the indices with the non-zero values.
    std::vector<int> nonzero_indices;

    for (int j = 0; j < equality_constraint.length(); j++) {
      if (equality_constraint[j] != 0.0) {

        // If the variable already has been encountered in an earlier
        // constraint, then abort.
        if (attributes_present_in_equality_constraints[j]) {
          fl::logger->Die() << j << "-th attribute has been already "
          "used in an earlier equality constraint.";
        }

        // Turn on the flag and push in the indices.
        attributes_present_in_equality_constraints[j] = true;
        nonzero_indices.push_back(j);

        // If there are more than 2 nonzero indices encountered, then
        // abort.
        if (nonzero_indices.size() > 2) {
          fl::logger->Die() << "Currently does not support more than "
          "two attributes per constraint.";
        }
      }
    }
    if (nonzero_indices.size() != 2) {
      fl::logger->Die() << "Currently does not support more than "
      "two attributes per constraint.";
    }

    // Now loop through each reference point apply the scaling.
    double scaling_factor = equality_constraint[nonzero_indices[1]] /
                            equality_constraint[nonzero_indices[0]];
    for (int j = 0; j < reference_table->n_entries(); j++) {
      typename TableType1::Point_t reference_point;
      reference_table->get(j, &reference_point);

      reference_point.set(
        nonzero_indices[0],
        -scaling_factor * reference_point[nonzero_indices[0]]);
    }
  }
}

template<typename TableType1>
template<class DataAccessType>
int fl::ml::LinearRegression<boost::mpl::void_>::Core<TableType1>::Main(
  DataAccessType *data,
  boost::program_options::variables_map &vm) {
  std::string run_mode_in = vm["run_mode"].as<std::string>();

  if (run_mode_in == "train") {
    std::string algorithm_in;
    try {
      algorithm_in = vm["algorithm"].as<std::string>();
    }
    catch (const boost::bad_lexical_cast &e) {
      fl::logger->Die() << "Flag --algorithm must be set to a string";
    }
    if (algorithm_in == "naive") {
      return Branch<true>(data, vm);
    }
    else if (algorithm_in == "fast") {
      return Branch<false>(data, vm);
    }
    else {
      fl::logger->Die() << "--algorithm accepts one of: fast, naive";
    }
  }
  else if (run_mode_in == "eval") {

    // Find where the bias term is in the model, if any.
    int input_coeffs_bias_term_index = -1;
    if (vm.count("input_coeffs_bias_term_index") > 0) {
      input_coeffs_bias_term_index =
        vm["input_coeffs_bias_term_index"].as<int>();
    }

    // Use the coeffs_in and load the model.
    boost::shared_ptr<typename DataAccessType::DefaultTable_t> coefficients_table;
    std::string input_coeffs = vm["input_coeffs"].as<std::string>();
    data->Attach(vm["input_coeffs"].as<std::string>(), &coefficients_table);

    // Read in the query table.
    boost::shared_ptr<TableType1> query_table;
    std::string queries_in = vm["queries_in"].as<std::string>();
    fl::logger->Message() << "Loading query data from file " << queries_in;
    data->Attach(queries_in, &query_table);
    // If the bias term index is not valid, then abort.
    if (query_table->n_attributes() <= input_coeffs_bias_term_index) {
      fl::logger->Die() << "--input_coeffs_bias_term_index specified an index "
      "out of range";
    }

    // Open the output table for the computed predictions.
    boost::shared_ptr<typename DataAccessType::DefaultTable_t> predictions_out_table;
    data->Attach(vm["predictions_out"].as<std::string>(),
                 std::vector<index_t>(1, 1),
                 std::vector<index_t>(),
                 query_table->n_entries(),
                 &predictions_out_table);
    for (int i = 0; i < query_table->n_entries(); i++) {

      // Get the query point.
      typename TableType1::Point_t query_point;
      query_table->get(i, &query_point);

      // Get the output point.
      typename DataAccessType::DefaultTable_t::Point_t
      output_point;
      predictions_out_table->get(i, &output_point);

      // Computed prediction.
      double prediction = 0;
      for (int j = 0; j < query_point.length(); j++) {
        typename DataAccessType::DefaultTable_t::Point_t
        coeff_point;
        coefficients_table->get(j, &coeff_point);
        if (j != input_coeffs_bias_term_index) {
          prediction += coeff_point[j] * query_point[j];
        }
        else {
          prediction += coeff_point[j];
        }
      }
      output_point.set(0, prediction);
    }

    data->Purge(vm["predictions_out"].as<std::string>());
    data->Detach(vm["predictions_out"].as<std::string>());

  }

  return 0;
}

template<typename TableType1>
template<bool do_naive_least_squares, class DataAccessType>
int fl::ml::LinearRegression<boost::mpl::void_>::Core<TableType1>::Branch(
  DataAccessType *data,
  boost::program_options::variables_map &vm) {
  // Read in the equality constraints if available.
  std::string equality_constraints_in;
  if (vm.count("equality_constraints_in")) {
    try {
      equality_constraints_in = vm["equality_constraints_in"].as<std::string>();
    }
    catch (const boost::bad_lexical_cast &e) {
      fl::logger->Die() << "Flag --equality_constraints_in must be set "
      "to a string";
    }
  }

  std::string references_in;
  try {
    references_in = vm["references_in"].as<std::string>();
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --references_in must be set to a string";
  }

  // Read in the reference table.
  boost::shared_ptr<TableType1> reference_table;
  fl::logger->Message() << "Loading reference data from file " << references_in;
  data->Attach(references_in, &reference_table);
  // Read in the equality constraints table.
  boost::shared_ptr<TableType1> equality_constraints_table;
  if (vm.count("equality_constraints_in") > 0) {
    fl::logger->Message() << "Loading equality constraints from file " <<
    equality_constraints_in;
    data->Attach(equality_constraints_in, &equality_constraints_table);

    // Apply the equality constraints transformation here.
    ApplyEqualityConstraints_(*equality_constraints_table, reference_table.get());
  }

  // Setup the indices you want to use for the predictors and a subset
  // of the predictors indices you want to use for cross-validation,
  // and the predictions index.
  std::deque<int> predictor_indices;
  std::deque<int> prune_predictor_indices;
  int prediction_index;
  // Setup the prune predictor feature names for more comprehensive
  // output in the cross-validation.
  std::vector< std::string > prune_predictor_feature_names;

  // Parse the index prefixes that must be removed.
  std::vector< std::string > remove_index_prefixes;
  if (vm.count("remove_index_prefixes")) {
    try {
      remove_index_prefixes =
        vm["remove_index_prefixes"].as< std::vector< std::string> >();
    }
    catch (const boost::bad_lexical_cast &e) {
      fl::logger->Die() << "Flag --remove_index_prefixes must be set to a string";
    }
  }

  // Parse the index prefixes that must be considered for pruning.
  std::vector< std::string > prune_predictor_index_prefixes;
  if (vm.count("prune_predictor_index_prefixes")) {
    try {
      prune_predictor_index_prefixes =
        vm["prune_predictor_index_prefixes"].as< std::vector< std::string > >();
    }
    catch (const boost::bad_lexical_cast &e) {
      fl::logger->Die() << "Flag --prune_predictor_index_prefixes must be set to "
      "a string";
    }
  }

  // Parse the index that must be used for prediction.
  std::string prediction_index_prefix;

  if (vm.count("prediction_index_prefix")) {
    try {
      prediction_index_prefix =
        vm["prediction_index_prefix"].as< std::string >();
    }
    catch (const boost::bad_lexical_cast &e) {
      fl::logger->Die() << "Flag --prediction_index_prefix must be set to a string";
    }
  }
  else {
    prediction_index_prefix = "";
  }

  // Setup the indices in an interactive manner.
  SetupIndices_(
    *reference_table, remove_index_prefixes, prune_predictor_index_prefixes,
    prediction_index_prefix,
    &predictor_indices, &prune_predictor_indices,
    &prune_predictor_feature_names, &prediction_index);
  if (vm.count("check_columns")) {
    if (vm["check_columns"].as<bool>()) {
      std::vector<index_t> red_features=reference_table->RedundantCategoricals();
      /*
      std::cout<<red_features.size()<<":";
      for(size_t i=0; i<red_features.size(); ++i) {
        std::cout<<red_features[i]<<",";
      }
      std::cout<<std::endl;
      */
      for(std::vector<index_t>::const_iterator it=red_features.begin();
            it!=red_features.end(); ++it) {
        std::deque<int>::iterator it1;
        it1=std::find(predictor_indices.begin(),
            predictor_indices.end(), *it);
        predictor_indices.erase(it1); 
      }
    }
  }
/*
  fl::logger->Message() << "Indices for predictors";
  for (int i = 0; i < predictor_indices.size(); i++) {
    fl::logger->Message() << " " << predictor_indices[i];
  }
  fl::logger->Message() << "Indices for prune predictors";
  for (int i = 0; i < prune_predictor_indices.size(); i++) {
    fl::logger->Message() << " " << prune_predictor_indices[i] << " " <<
    prune_predictor_feature_names[i].c_str();
  }
*/  
  fl::logger->Message() << "Prediction index: " << prediction_index;

  // The model computed initially after the VIF selection.
  fl::ml::LinearRegressionModel<TableType1, do_naive_least_squares> initial_model;

  // Perform the VIF-based pruning.
  double vif_threshold=0;
  try {
    vif_threshold = vm["vif_threshold"].as<double>();
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --vif_threshold must be set to a float";
  }

  if (vif_threshold <= 0.0) {
    fl::logger->Die() << "Variance inflation factor threshold must be "
    "positive.";
  }
  double conf_prob = 0;
  try {
    conf_prob = vm["conf_prob"].as<double>();
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --conf_prob must be set to a float";
  }
  if (conf_prob <= 0.0 || conf_prob >= 1.0) {
    fl::logger->Die() << "The coverage probability must be between 0 and 1 "
    "inclusive.";
  }

  bool include_bias_term = !vm["exclude_bias_term"].as<bool>();
  if (include_bias_term) {
    fl::logger->Message() << "Including the bias term.";
  }
  else {
    fl::logger->Message() << "Excluding the bias term.";
  }

  LinearRegressionLIC lic_engine;
  fl::data::MonolithicPoint<double> coefficients;
  ModelStatistics lic_stats;
  initial_model.Init(*reference_table, predictor_indices,
                     prediction_index, include_bias_term, conf_prob);
  fl::logger->Message() << "Initialized the model.";
  if (vm["correlation_pruning"].as<bool>() == false 
      && vm["vif_pruning"].as<bool>() == false 
      && vm["stepwise"].as<bool>() == false
      && vm["ineq_nnls"].as<bool>()==false
      && vm["ineq_ldp"].as<bool>()==false
      && vm["ineq_lsi"].as<bool>()==false) {
    fl::logger->Message()<<"Simple Linear Regression Chosen"<<std::endl;
    initial_model.set_active_right_hand_side_column_index(prediction_index);
    initial_model.Solve(); 
    // Compute the standard errors, etc.
    fl::ml::LinearRegressionResult<TableType1> cross_validation_result;
    initial_model.Predict(*reference_table, &cross_validation_result);
    initial_model.ComputeModelStatistics(cross_validation_result);
  } else {
    if (vm["correlation_pruning"].as<bool>() == true) {
      fl::logger->Message() << "Correlation pruning.";
      double correlation_threshold=0;
      try {
        correlation_threshold = vm["correlation_threshold"].as<double>();
      }
      catch (const boost::bad_lexical_cast &e) {
        fl::logger->Die() << "Flag --correlation_threshold must be set to a float";
      }
      fl::logger->Message() << "Correlation threshold: " << correlation_threshold;
      fl::ml::CorrelationPrune::Compute(correlation_threshold,
                                        prune_predictor_indices,
                                        &initial_model);
    } else {
      if (vm["vif_pruning"].as<bool>()==true) {
        fl::logger->Message() << "VIF threshold: " << vif_threshold;
        fl::logger->Message() << "Doing VIF selection.";
        fl::ml::LinearRegressionResult<TableType1> cross_validation_result;
        VifPrune::Compute(vif_threshold, prune_predictor_indices, &initial_model);
  
        // Compute the standard errors, etc.
        initial_model.Predict(*reference_table, &cross_validation_result);
        initial_model.ComputeModelStatistics(cross_validation_result);
      } else {
        if (vm["ineq_lsi"].as<bool>()==true ||
            vm["ineq_ldp"].as<bool>()==true ||
            vm["ineq_nnls"].as<bool>()==true) {
          if (vm["ineq_lsi"].as<bool>()
              +vm["ineq_ldp"].as<bool>()
              +vm["ineq_nnls"].as<bool>() >1) {
            fl::logger->Die()<<"You can set only one of these flags "
              "--ineq_lsi, --ineq_ldp, --ineq_nnls to 1";
          }
          bool include_bias_term=!vm["exclude_bias_term"].as<bool>();
          if (vm.count("remove_index_prefixes")) {
            fl::logger->Die()<< "remove index_prefixes is not supported in LIC regression";
          }
          if(vm["ineq_ldp"].as<bool>()==true) {
            if (vm.count("ineq_in")>0) {
              fl::logger->Die()<<"For the ldp problem the inequality table is the "
                "references, do not use ineq_in";
            }
          }
          if (vm["ineq_lsi"].as<bool>()==true) {
            if (vm.count("ineq_in")==0) {
              fl::logger->Die()<<"If you set ineq_lsi==1 then you also need to provide "
                "a --ineq_in table";
            }
          }

          fl::dense::Matrix<double> e_mat;
          fl::dense::Matrix<double, true> f_vec;
          fl::dense::Matrix<double> g_mat;
          fl::dense::Matrix<double, true> h_vec; 
          
          index_t rhs_column=vm["ineq_rhs_column"].as<index_t>();
          e_mat.Init(reference_table->n_entries(), 
              reference_table->n_attributes()-(include_bias_term?0:1));
          f_vec.Init(reference_table->n_entries());
          typedef typename TableType1::Point_t Point_t;
          Point_t point;
          // In the case of ldp the g matrix is the references so we have to
          // make this distinction
          if (vm["ineq_ldp"].as<bool>()==false) {
            for(index_t i=0; i<reference_table->n_entries(); ++i) {
              index_t attribute=0;
              reference_table->get(i, &point);
              for(typename Point_t::iterator it=point.begin(); it!=point.end(); ++it) {
                if (it.attribute()==prediction_index) {
                  f_vec[i]=it.value();
                } else {
                  e_mat.set(i, attribute, it.value());
                  attribute++;
                }
              }
              if (include_bias_term==true) {
                e_mat.set(i, e_mat.n_cols()-1, 1);
              }
            }
          } else {
            g_mat.Init(reference_table->n_entries(), 
                reference_table->n_attributes()-(rhs_column<0?0:1));
            h_vec.Init(reference_table->n_entries());
            for(index_t i=0; i<reference_table->n_entries(); ++i) {
              reference_table->get(i, &point);
              index_t attribute=0;
              for(typename Point_t::iterator it=point.begin(); it!=point.end(); ++it) {
                if (it.attribute()==rhs_column) {
                  h_vec[i]=it.value();
                } else {
                  g_mat.set(i, attribute, it.value());
                  attribute++;
                }
              }
            } 
          } 

          boost::shared_ptr<TableType1> ineq_table;
          if (vm.count("ineq_in")!=0) {
            data->Attach(vm["ineq_in"].as<std::string>(), &ineq_table);
            g_mat.Init(ineq_table->n_entries(), 
                ineq_table->n_attributes()-(rhs_column<0?0:1));
            h_vec.Init(ineq_table->n_entries());
            for(index_t i=0; i<ineq_table->n_entries(); ++i) {
              ineq_table->get(i, &point);
              index_t attribute=0;
              for(typename Point_t::iterator it=point.begin(); it!=point.end(); ++it) {
                if (it.attribute()==rhs_column) {
                  h_vec[i]=it.value();
                } else {
                  g_mat.set(i, attribute, it.value());
                  attribute++;
                }
              }
            } 
          
            if (rhs_column<0) {
              if (vm.count("ineq_rhs_in")==0) {
                fl::logger->Die()<<"Since you gave --ineq_rhs_column <0 "
                  "you have to define --ineq_rhs_in";
              } else {
                boost::shared_ptr<typename DataAccessType::DefaultTable_t> rhs_table;
                data->Attach(vm["ineq_rhs_in"].as<std::string>(), &rhs_table);
                if (rhs_table->n_entries()!=ineq_table->n_entries()) {
                  fl::logger->Die()<<"--ineq_rhs_in must have the same number of rows "
                    "with --ineq_table_in";
                }
                typename DataAccessType::DefaultTable_t::Point_t point;
                for(index_t i=0; i<rhs_table->n_entries(); ++i) {
                  rhs_table->get(i, &point);
                  h_vec[i]=point[0];
                }
              }
            }
          }
          if ( vm["ineq_lsi"].as<bool>()==true && g_mat.n_cols()!=e_mat.n_cols()) {
            fl::logger->Die() << "The e matrix must have the same number of columns "
              "with the g matrix. If you choose to include bias in your regression "
              "then the g (constraints) matrix must have a column (the last one) that "
              "represents restrictions on the bias";
          }
          lic_engine.set_tolerance(vm["tolerance"].as<double>());
          lic_engine.set_conf_prob(conf_prob);
          std::set<index_t> p_set;
          if (vm["ineq_nnls"].as<bool>()==true) {
            lic_engine.ComputeNNLS(e_mat,
                        f_vec,
                        &coefficients,
                        &p_set,
                        &lic_stats);
          } else {
            if (vm["ineq_ldp"].as<bool>()==true) {
              bool feasible = lic_engine.ComputeLDP(g_mat,
                         h_vec,
                         &coefficients,
                         &p_set,
                         &lic_stats);
              if (feasible==false) {
                fl::logger->Die() << "The problem is infeasible, "
                  "check your constraints";
              }
            } else {
              if (vm["ineq_lsi"].as<bool>()==true) {
                bool feasible = lic_engine.ComputeLSI(e_mat,
                           f_vec, 
                           g_mat,
                           h_vec,
                           &coefficients,
                           &p_set,
                           &lic_stats);

                if (feasible==false) {
                  fl::logger->Die() << "The problem is infeasible, "
                    "check your constraints";
                }
              }
            }
          }
        }
      }
    }
  
    // If the stepwise regression is specified, then do it.
    if (vm["stepwise"].as<bool>()==true) {
      fl::logger->Message() << "\nStarting stepwise regression.";
      fl::ml::LinearRegressionResult<TableType1> stepwise_result;
      double stepwise_threshold = 0;
      try {
        stepwise_threshold = vm["stepwise_threshold"].as<double>();
      }
      catch (const boost::bad_lexical_cast &e) {
        fl::logger->Die() << "Flag --stepwise_threshold must be set to a float";
      }
      try {
        if (vm["stepdirection"].as<std::string>() == std::string("bidir")) {
          fl::logger->Message() << "Doing bidirectional step...";
          StepwiseRegression<true, true>::Compute(stepwise_threshold,
                                                  &initial_model);
        }
        else if (vm["stepdirection"].as<std::string>() == std::string("forward")) {
          fl::logger->Message() << "Stepping only forward...";
          StepwiseRegression<true, false>::Compute(stepwise_threshold,
              &initial_model);
        }
        else {
          fl::logger->Message() << "Stepping only backward...";
          StepwiseRegression<false, true>::Compute(stepwise_threshold,
              &initial_model);
        }
      }
      catch (const boost::bad_lexical_cast &e) {
        fl::logger->Die() << "Flag --stepdirection must be set to a string";
      }
  
  
      // Compute the standard errors, etc.
      initial_model.Predict(*reference_table, &stepwise_result);
      initial_model.ComputeModelStatistics(stepwise_result);
    }
  } 
  fl::logger->Message()<<"All regressions are done"<<std::endl;
  // Export the result.
  fl::logger->Message()<<"Exporting the results"<<std::endl;
  boost::shared_ptr<typename DataAccessType::DefaultTable_t > coefficients_table,
  standard_errors_table, confidence_interval_los_table,
  confidence_interval_his_table, t_statistics_table, p_values_table,
  adjusted_r_squared_table, f_statistic_table, r_squared_table,
  sigma_table;

  // Adjust the number of coefficients since the attribute that is
  // used for the prediction (target) will be replaced by the bias if
  // requested.
  int num_of_coeff = initial_model.coefficients().size();
  if (include_bias_term) {
    num_of_coeff--;
  }
  if (vm["ineq_lsi"].as<bool>()==true ||
      vm["ineq_ldp"].as<bool>()==true ||
      vm["ineq_nnls"].as<bool>()==true) {
    num_of_coeff=reference_table->n_attributes()
      -(vm["exclude_bias_term"].as<bool>()?1:0);
    if (vm["ineq_rhs_column"].as<index_t>()<0) {
      num_of_coeff++;
    }
  }
  try {
    if (vm.count("coeffs_out")) {
      data->Attach(vm["coeffs_out"].as<std::string>(),
                   std::vector<index_t>(1, 1),
                   std::vector<index_t>(),
                   num_of_coeff,
                   &coefficients_table);
    }
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --coeffs_out must be set to a string";
  }

  try {
    if (vm.count("standard_errors_out")) {
      data->Attach(vm["standard_errors_out"].as<std::string>(),
                   std::vector<index_t>(1, 1),
                   std::vector<index_t>(),
                   num_of_coeff,
                   &standard_errors_table);
    }
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --standard_errors_out must be set to a string";
  }
  try {
    if (vm.count("conf_los_out")) {
      data->Attach(vm["conf_los_out"].as<std::string>(),
                   std::vector<index_t>(1, 1),
                   std::vector<index_t>(),
                   num_of_coeff,
                   &confidence_interval_los_table);
    }
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --conf_los_out must be set to a string";
  }
  try {
    if (vm.count("conf_his_out")) {
      data->Attach(vm["conf_his_out"].as<std::string>(),
                   std::vector<index_t>(1, 1),
                   std::vector<index_t>(),
                   num_of_coeff,
                   &confidence_interval_his_table);
    }
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --oputput_conf_his must be set to a string";
  }
  try {
    if (vm.count("t_values_out")) {
      data->Attach(vm["t_values_out"].as<std::string>(),
                   std::vector<index_t>(1, 1),
                   std::vector<index_t>(),
                   num_of_coeff,
                   &t_statistics_table);
    }
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --t_values_out must be set to a string";
  }
  try {
    if (vm.count("p_values_out")) {
      data->Attach(vm["p_values_out"].as<std::string>(),
                   std::vector<index_t>(1, 1),
                   std::vector<index_t>(),
                   num_of_coeff,
                   &p_values_table);
    }
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --p_values_out must be set to a string";
  }
  try {
    if (vm.count("adjusted_r_squared_out")) {
      data->Attach(vm["adjusted_r_squared_out"].as<std::string>(),
                   std::vector<index_t>(1, 1),
                   std::vector<index_t>(),
                   1,
                   &adjusted_r_squared_table);
    }
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --adjusted_r_squared_out must be set to a string";
  }
  try {
    if (vm.count("f_statistic_out")) {
      data->Attach(vm["f_statistic_out"].as<std::string>(),
                   std::vector<index_t>(1, 1),
                   std::vector<index_t>(),
                   1,
                   &f_statistic_table);
    }
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --f_statistic_out must be set to a string";
  }
  try {
    if (vm.count("r_squared_out")) {
      data->Attach(vm["r_squared_out"].as<std::string>(),
                   std::vector<index_t>(1, 1),
                   std::vector<index_t>(),
                   1,
                   &r_squared_table);
    }
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --r_squared_out must be set to a string";
  }
  try {
    if (vm.count("sigma_out")) {
      data->Attach(vm["sigma_out"].as<std::string>(),
                   std::vector<index_t>(1, 1),
                   std::vector<index_t>(),
                   1,
                   &sigma_table);
    }
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --sigma_out must be set to a string";
  }

  if (vm["ineq_lsi"].as<bool>()==true ||
      vm["ineq_ldp"].as<bool>()==true ||
      vm["ineq_nnls"].as<bool>()==true) {
    if (vm["ineq_ldp"].as<bool>()==true ) {
      prediction_index=vm["ineq_rhs_column"].as<index_t>();
    }
    lic_stats.Export(prediction_index,
                     standard_errors_table.get(),
                     confidence_interval_los_table.get(),
                     confidence_interval_his_table.get(),
                     t_statistics_table.get(),
                     p_values_table.get(),
                     adjusted_r_squared_table.get(),
                     f_statistic_table.get(),
                     r_squared_table.get(),
                     sigma_table.get());
    index_t attribute=0;
    for(index_t i=0; i<coefficients.size()+(prediction_index<0?0:1); ++i) {
      if (i!=prediction_index) {
        coefficients_table->set(i, 0, coefficients[attribute]);
        attribute++;
      } else {
        coefficients_table->set(i, 0, 0);
      }
    }

  } else {
    initial_model.Export(coefficients_table.get(),
                       standard_errors_table.get(),
                       confidence_interval_los_table.get(),
                       confidence_interval_his_table.get(),
                       t_statistics_table.get(),
                       p_values_table.get(),
                       adjusted_r_squared_table.get(),
                       f_statistic_table.get(),
                       r_squared_table.get(),
                       sigma_table.get());

  }

  if (vm.count("coeffs_out")) {
    data->Purge(vm["coeffs_out"].as<std::string>());
    data->Detach(vm["coeffs_out"].as<std::string>());
  }
  
  if (vm.count("standard_errors_out")) {
    data->Purge(vm["standard_errors_out"].as<std::string>());
    data->Detach(vm["standard_errors_out"].as<std::string>());
  }

  if (vm.count("conf_los_out")) {
    data->Purge(vm["conf_los_out"].as<std::string>());
    data->Detach(vm["conf_los_out"].as<std::string>());
  }

  if (vm.count("conf_his_out")) {
    data->Purge(vm["conf_his_out"].as<std::string>());
    data->Detach(vm["conf_his_out"].as<std::string>());
  }

  if (vm.count("t_values_out")) {
    data->Purge(vm["t_values_out"].as<std::string>());
    data->Detach(vm["t_values_out"].as<std::string>());
  }

  if (vm.count("p_values_out")) { 
    data->Purge(vm["p_values_out"].as<std::string>());
    data->Detach(vm["p_values_out"].as<std::string>());
  }

  if (vm.count("adjusted_r_squared_out")) {
    data->Purge(vm["adjusted_r_squared_out"].as<std::string>());
    data->Detach(vm["adjusted_r_squared_out"].as<std::string>());
  }

  if (vm.count("f_statistic_out")) {
    data->Purge(vm["f_statistic_out"].as<std::string>());
    data->Detach(vm["f_statistic_out"].as<std::string>());
  }

  if (vm.count("r_squared_out")) {
    data->Purge(vm["r_squared_out"].as<std::string>());
    data->Detach(vm["r_squared_out"].as<std::string>());
  }


  if (vm.count("sigma_out")) {
    data->Purge(vm["sigma_out"].as<std::string>());
    data->Detach(vm["sigma_out"].as<std::string>());
  }

  fl::logger->Message()<<"Done exporting"<<std::endl;
  return 0;
}

bool fl::ml::LinearRegression<boost::mpl::void_>::ConstructBoostVariableMap(
  const std::vector<std::string> &args,
  boost::program_options::variables_map *vm) {

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////
  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Display help on ridge regression")
  ("algorithm", 
    boost::program_options::value<std::string>()->default_value("fast"),
   "The type of algorithm to use: One of\n"
   "  fast, naive"
  )
  ("conf_prob", 
   boost::program_options::value<double>()->default_value(0.9),
   "Specifies the probability converage of the confidence intervals."
  )
  ("exclude_bias_term",
   boost::program_options::value<bool>()->default_value(false),
   "If present, the bias term will not be included in the linear model."
  )
  ("equality_constraints_in", 
   boost::program_options::value<std::string>(),
   "data file containing the equality constraints. If present, the linear "
   "model will be adjusted according to this constraint."
  )
  ("ineq_in",
   boost::program_options::value<std::string>(),
   "table that has the inequalities, the same form of references_in"
  )
  ("ineq_rhs_column",
   boost::program_options::value<index_t>()->default_value(0),
   "The column of the ineq_in table that will be the right hand side "
   "of the constraints" 
  )
  ("ineq_rhs_in",
   boost::program_options::value<std::string>(),
   "If you declare --ineq_rhs_column=-1 then you can use --ineq_rhs_in to "
   "load the targets" 
  )
  ("queries_in", 
   boost::program_options::value<std::string>(),
   "data file containing the test set to be evaluated on the --run_mode="
   "eval mode."
  )
  ("references_in", 
   boost::program_options::value<std::string>(),
   "data file containing the predictors and the predictions"
  )("check_columns",
    boost::program_options::value<bool>()->default_value(false),
    "checks if a column has all the same value. It removes it from the regression"
  )("remove_index_prefixes",
   boost::program_options::value< std::vector< std::string> >(),
   "The list of strings, each of which denotes the prefix that should be"
   " removed from the consideration of predictor set. We remove one "
   "feature for each prefix."
  )
  ("prune_predictor_index_prefixes",
   boost::program_options::value< std::vector< std::string > >(),
   "The list of strings, each of which denotes the prefix that should be "
   "included in the consideration of variance inflation factor based "
   "pruning.")
  ("prediction_index_prefix",
   boost::program_options::value< std::string >(),
   "The index for which should be used as the prediction. It can be a label "
   "of the table attributes. If it cannot be found in the references_in labels "
   "then it will try to interpret it as the index of the attribute i.e. 0,1,..."
  )
  ("correlation_pruning",
   boost::program_options::value<bool>()->default_value(false),
   "If present, the correlation coefficient should be used for initial "
   "pruning."
  )
  ("ineq_lsi",
   boost::program_options::value<bool>()->default_value(false),
   "Solving regression with inequality constraints "
   "Minimize ||Ex-f|| subject to Gx >= h" 
  )
  ("ineq_nnls",
   boost::program_options::value<bool>()->default_value(false),
   "Solving regression with inequality constraints "
   "Minimize ||Ex-f|| subject to x >= 0"
  )
  ("ineq_ldp",
   boost::program_options::value<bool>()->default_value(false),
   "Solving regression with inequality constraints "
   "Minimize ||x|| subject to Gx >= h"
  )
  ("tolerance", 
   boost::program_options::value<double>()->default_value(1e-5),
   "When solving regression with inequality constraints the tolerance "
   "is the minimum value for a coefficient to be considered zero"
  )
  ("vif_pruning",
   boost::program_options::value<bool>()->default_value(false),
   "if present then it will do vif pruning"
  )
  ("stepwise",
   boost::program_options::value<bool>()->default_value(false),
   "If present, the stepwise regression should be used after VIF "
   "selection."
  )
  ("stepdirection",
   boost::program_options::value< std::string >()->default_value("bidir"),
   "The direction of stepwise regression: One of\n"
   "  bidir, forward, backward"
  )
  ("input_coeffs",
   boost::program_options::value<std::string>(),
   "The input file for the coefficients"
  )
  ("input_coeffs_bias_term_index",
   boost::program_options::value<int>(),
   "The index position of the bias term if present in the input coefficients."
  )
  ("coeffs_out",
   boost::program_options::value<std::string>(),
   "The output file for the coefficients")
  ("standard_errors_out",
   boost::program_options::value<std::string>(),
   "The output file for the standard errors"
  )
  ("conf_los_out",
   boost::program_options::value<std::string>(),
   "The output file for the lower bound for the confidence intervals"
  )
  ("conf_his_out",
   boost::program_options::value<std::string>(),
   "The output file for the upper bound for the confidence intervals"
  )
  ("t_values_out",
   boost::program_options::value<std::string>(),
   "The output file for the t values"
  )
  ("p_values_out", 
   boost::program_options::value<std::string>(),
   "The output file for the p values"
  )
  ("adjusted_r_squared_out",
   boost::program_options::value<std::string>(),
   "The output file for the adjusted r-squared value"
  )
  ("f_statistic_out",
   boost::program_options::value<std::string>(),
   "The output file for the f-statistics")
  ("r_squared_out",
   boost::program_options::value<std::string>(),
   "The output file for the r-squared value"
  )
  ("sigma_out",
   boost::program_options::value<std::string>(),
   "The output file for the sigma"
  )
  ("correlation_threshold",
    boost::program_options::value<double>()->default_value(0.9),
    "During the correlation pruning, each attribute will be pruned if "
    "it has absolute correlation factor more than this value with any other "
    "attributes."
  )
  ("stepwise_threshold",
   boost::program_options::value<double>()->default_value(
     std::numeric_limits<double>::min()),
   "A minimum required improvement in the score needed during stepwise "
   "regression in order to continue the selection."
  )(
    "vif_threshold",
    boost::program_options::value<double>()->default_value(8.0),
    "During the pruning stage via variance inflation factor, "
    "each attribute will be pruned if the threshold exceeds this value."
  )(
    "run_mode",
    boost::program_options::value<std::string>()->default_value("train"),
    "Linear regression as every machine learning algorithm has two modes, "
    "the training and the evaluations."
    " When you set this flag to --run_mode=train it will find the set of "
    "coefficients."
    " Once you have found these coefficients and you want to predict on a "
    "new set of points, you should run it in the --run_mode=eval. Of course "
    "you should provide the data to be evaluated"
    " over the linear regression coefficients, by setting the --queries_in "
    "flag appropriately."
   );

  boost::program_options::command_line_parser clp(args);
  clp.style(boost::program_options::command_line_style::default_style
            ^ boost::program_options::command_line_style::allow_guessing);
  try {
    boost::program_options::store(clp.options(desc).run(), *vm);
  }
  catch (const boost::program_options::invalid_option_value &e) {
    fl::logger->Die() << "Invalid Argument: " << e.what();
  }
  catch (const boost::program_options::invalid_command_line_syntax &e) {
    fl::logger->Die() << "Invalid command line syntax: " << e.what();
  }
  catch (const boost::program_options::unknown_option &e) {
    fl::logger->Die() << "Unknown option: " << e.what() ;
  }
  catch ( const boost::program_options::error &e) {
    std::string cli;
    for(std::vector<std::string>::const_iterator it=args.begin();
        it!=args.end(); ++it) {
      cli+=" "+*it;
    }
    fl::logger->Message()<<"options: "<<cli<<std::endl;
    fl::logger->Die() << e.what();
  } 
  boost::program_options::notify(*vm);
  if (vm->count("help")) {
    fl::logger->Message() << fl::DISCLAIMER << "\n";
    fl::logger->Message() << desc << "\n";
    return true;
  }


  // Do argument checking.
  if (!(vm->count("references_in"))) {
    fl::logger->Die() << "Missing required --references_in";
    return true;
  }
  std::string run_mode_in;
  try {
    run_mode_in = (*vm)["run_mode"].as<std::string>();
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --run_mode must be set to a string";
  }
  if (run_mode_in != "train" && run_mode_in != "eval") {
    fl::logger->Die() << "--run_mode accepts one of: train, eval";
  }
  if (run_mode_in == "eval" && vm->count("queries_in") == 0) {
    fl::logger->Die() << "--run_mode=eval requires --queries_in argument.";
  }

  return false;
}


template<class DataAccessType, typename BranchType>
int fl::ml::LinearRegression<boost::mpl::void_>::Main(
  DataAccessType *data,
  const std::vector<std::string> &args
) {
  FL_SCOPED_LOG(Regression);
  boost::program_options::variables_map vm;
  bool help_specified = ConstructBoostVariableMap(args, &vm);

  if (help_specified) {
    return 1;
  }
  return BranchType::template BranchOnTable<LinearRegression<boost::mpl::void_>, DataAccessType>(data, vm);
}

template<typename DataAccessType>
void fl::ml::LinearRegression<boost::mpl::void_>::Run(
    DataAccessType *data,
    const std::vector<std::string> &args) {
  fl::ws::Task<
    DataAccessType,
    &Main<DataAccessType, typename DataAccessType::Branch_t>
  > task(data, args);
  data->schedule(task); 
}

#endif
