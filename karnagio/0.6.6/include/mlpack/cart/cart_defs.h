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
#ifndef FL_LITE_MLPACK_CART_CART_DEFS_H
#define FL_LITE_MLPACK_CART_CART_DEFS_H
#include <iostream>
#include "boost/program_options.hpp"
#include "boost/mpl/map.hpp"
#include "fastlib/base/base.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "fastlib/table/table.h"
#include "fastlib/tree/spacetree.h"
#include "fastlib/table/default_table.h"
#include "fastlib/tree/kdtree.h"
#include "fastlib/tree/metric_tree.h"
#include "mlpack/cart/cart.h"
#include "fastlib/workspace/task.h"

template<typename TableType1>
template<class DataAccessType>
int fl::ml::Cart<boost::mpl::void_>::Core<TableType1>::Main(
  DataAccessType *data,
  boost::program_options::variables_map &vm
) {
  if (!vm.count("references_in")) {
    fl::logger->Die() << "Missing required --references_in";
    return 1;
  }

  std::string references_in = vm["references_in"].as<std::string>();
  std::string queries_in = vm["queries_in"].as<std::string>();

  typedef typename Cart<boost::mpl::void_>::Core<TableType1>::TableType
  TableType;
  boost::shared_ptr<TableType> reference_table;
  fl::logger->Message() << "Loading reference data from " << references_in <<std::endl;
  boost::shared_ptr<TableType1> temp_table;
  data->Attach(references_in, &temp_table);
  fl::logger->Message() << "Reference data loaded"<<std::endl;
  boost::shared_ptr<typename DataAccessType::IntegerTable_t> labels_table;
  std::string labels_in = vm["labels_in"].as<std::string>();
  if (labels_in != "") {
    fl::logger->Message() << "Loading labels from " << labels_in <<std::endl;
    data->Attach(labels_in, &labels_table); 
    data->template TieLabels<0>(temp_table, labels_table, data->GiveTempVarName(), &reference_table);
    fl::logger->Message() << "Labels loaded"<<std::endl;
  } else {
    data->CopyAndDestruct(temp_table, &reference_table);
  }

  boost::shared_ptr<TableType1> query_table;
  if (queries_in != "") {
    fl::logger->Message() << "Loading query data from " << queries_in <<std::endl;
    data->Attach(queries_in, &query_table);
    fl::logger->Message() << "Query data loaded " << std::endl;
  }


  // Build the CART.
  index_t leaf_size_in = vm["leaf_size"].as<index_t>();
  std::string impurity_in = vm["impurity"].as<std::string>();
  Cart<TableType> cart;
  cart.Init(reference_table.get(), leaf_size_in, impurity_in);

  // If the query set is provided, we need to do classification and
  // export the results.
  if (query_table != NULL) {

    // Allocate the results table.
    boost::shared_ptr<typename DataAccessType::template TableVector<index_t> > labels_out;
    std::string labels_out_file_name = vm["labels_out"].as<std::string>();
    data->Attach(labels_out_file_name,
                 std::vector<index_t>(1, 1),
                 std::vector<index_t>(),
                 query_table->n_entries() , &labels_out);
    fl::logger->Message() << "Classifying query data" << std::endl;
    cart.Classify(*query_table, labels_out.get());
    fl::logger->Message() << "Classification complete " << std::endl;
    //fl::logger->Message() << "Emiting results to " << labels_out;
    data->Purge(labels_out_file_name);
    data->Detach(labels_out_file_name);
  }

  // Output the tree, if requested.
  if( vm.count("tree_out") ) {
    std::string tree_string;
    cart.String(reference_table->get_tree(), 0, &tree_string);

    // Will have to decide how to export the tree string here.
  }

  // Free memory.
  return 0;
}

// bool fl::ml::Cart<boost::mpl::void_>::ConstructBoostVariableMap(
//   const std::vector<std::string> &args,
//   boost::program_options::variables_map *vm) {

//   boost::program_options::options_description desc("Available options");
//   desc.add_options()(
//     "help", "Print this information."
//   )(
//     "references_in",
//     boost::program_options::value<std::string>(),
//     "REQUIRED file containing reference data"
//   )(
//     "queries_in",
//     boost::program_options::value<std::string>()->default_value(""),
//     "OPTIONAL file containing query positions.  If omitted, CART computes "
//     "the leave-one-out result at each reference point."
//   )(
//     "labels_in",
//     boost::program_options::value<std::string>()->default_value(""),
//     "OPTIONAL file containing the labels of the reference data. If it is not "
//     "defined then it will try to find them in the references_in file"
//   )(
//     "labels_out",
//     boost::program_options::value<std::string>()->default_value(""),
//     "OPTIONAL file to store computed labels."
//   )(
//     "point",
//     boost::program_options::value<std::string>()->default_value("dense"),
//     "Point type used by CART.  One of:\n"
//     "  dense, sparse, dense_sparse, categorical, dense_categorical"
//   )(
//     "impurity",
//     boost::program_options::value<std::string>()->default_value("entropy"),
//     "The type of impurity used to generate the split. One of:\n"
//     "  entropy, gini, misclassification"
//   )(
//     "tree",
//     boost::program_options::value<std::string>()->default_value("balltree"),
//     "Tree structure used by CART."
//   )(
//     "leaf_size",
//     boost::program_options::value<index_t>()->default_value(20),
//     "Maximum number of points at a leaf of the tree.  More saves on tree "
//     "overhead but too much hurts asymptotic run-time."
//   )(
//     "tree_out",
//     boost::program_options::value<std::string>(),
//     "The output file for the tree."
//   )(
//     "log",
//     boost::program_options::value<std::string>()->default_value(""),
//     "A file to receive the log, or omit for stdout."
//   )(
//     "loglevel",
//     boost::program_options::value<std::string>()->default_value("debug"),
//     "Level of log detail.  One of:\n"
//     "  debug: log everything\n"
//     "  verbose: log messages and warnings\n"
//     "  warning: log only warnings\n"
//     "  silent: no logging"
//   );

//   boost::program_options::command_line_parser clp(args);
//   clp.style(boost::program_options::command_line_style::default_style
//      ^boost::program_options::command_line_style::allow_guessing );
//   boost::program_options::store(clp.options(desc).run(), *vm);
//   boost::program_options::notify(*vm);

//   if (vm->count("help")) {
//     std::cout << fl::DISCLAIMER << "\n";
//     std::cout << desc << "\n";
//     return true;
//   }
//   return false;
// }

template<class DataAccessType, typename BranchType>
int fl::ml::Cart<boost::mpl::void_>::Main(
  DataAccessType *data,
  const std::vector<std::string> &args
) {

  boost::program_options::variables_map vm;

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing reference data"
  )(
    "queries_in",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing query positions.  If omitted, CART computes "
    "the leave-one-out result at each reference point."
  )(
    "labels_in",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing the labels of the reference data. If it is not "
    "defined then it will try to find them in the references_in file"
  )(
    "labels_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file to store computed labels."
  )(
    "point",
    boost::program_options::value<std::string>()->default_value("dense"),
    "Point type used by CART.  One of:\n"
    "  dense, sparse, dense_sparse, categorical, dense_categorical"
  )(
    "impurity",
    boost::program_options::value<std::string>()->default_value("entropy"),
    "The type of impurity used to generate the split. One of:\n"
    "  entropy, gini, misclassification"
  )(
    "tree",
    boost::program_options::value<std::string>()->default_value("balltree"),
    "Tree structure used by CART."
  )(
    "leaf_size",
    boost::program_options::value<index_t>()->default_value(20),
    "Maximum number of points at a leaf of the tree.  More saves on tree "
    "overhead but too much hurts asymptotic run-time."
  )(
    "tree_out",
    boost::program_options::value<std::string>(),
    "The output file for the tree."
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

  boost::program_options::command_line_parser clp(args);
  clp.style(boost::program_options::command_line_style::default_style
     ^boost::program_options::command_line_style::allow_guessing );
  boost::program_options::store(clp.options(desc).run(), vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
    fl::logger->Message() << fl::DISCLAIMER << "\n";
    fl::logger->Message() << desc << "\n";
    return 1;
  }

  return BranchType::template BranchOnTable<Cart<boost::mpl::void_>, DataAccessType>(data, vm);
}
  
template<typename DataAccessType>
void fl::ml::Cart<boost::mpl::void_>::Run(
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

#endif
