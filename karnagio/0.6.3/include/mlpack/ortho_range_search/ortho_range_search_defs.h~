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
#ifndef MLPACK_ORTHO_RANGE_SEARCH_ORTHO_RANGE_SEARCH_DEFS_H
#define MLPACK_ORTHO_RANGE_SEARCH_ORTHO_RANGE_SEARCH_DEFS_H

#include <string>
#include "boost/program_options.hpp"
#include "boost/mpl/map.hpp"
#include "fastlib/dense/matrix.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/tree/tree.h"
#include "fastlib/table/table.h"
#include "ortho_range_search.h"
#include "fastlib/workspace/task.h"

namespace fl {
namespace ml {

template<typename DataAccessType, typename BranchType>
int OrthoRangeSearch<boost::mpl::void_>::Main(
  DataAccessType *data,
  const std::vector<std::string> &args) {

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////
  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Display help on orthogonal range search.")
  ("references_in", boost::program_options::value<std::string>(),
   "File containing the data points.")
  ("results_out", boost::program_options::value<std::string>(),
   "output_file")
  ("queries_in", boost::program_options::value<std::string>(),
   "File containing the search windows. Each line should be of the "
   "following: L_1, L_2, ... L_D H_1, H_2, ... H_D where L_i's is the "
   "lower bound for each dimension and H_i's the upper bound.")
  ("log",
   boost::program_options::value<std::string>()->default_value(""),
   "A file to receive the log, or omit for stdout.")
  (
    "point",
    boost::program_options::value<std::string>()->default_value("dense"),
    "Point type used by allkn.  One of:\n"
    "  dense, sparse, dense_sparse, categorical, dense_categorical")
  ("tree",
   boost::program_options::value<std::string>()->default_value("kdtree"),
   "Tree structure used by allkn.  One of:\n"
   "  kdtree, balltree")
  ("loglevel",
   boost::program_options::value<std::string>()->default_value("debug"),
   "Level of log detail.  One of:\n"
   "  debug: log everything\n"
   "  verbose: log messages and warnings\n"
   "  warning: log only warnings\n"
   "  silent: no logging");


  boost::program_options::variables_map vm;
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
  return BranchType::template BranchOnTable <
         OrthoRangeSearch<boost::mpl::void_>,
         DataAccessType
         > (data, vm);
}

template<typename TableType>
template<typename DataAccessType>
int OrthoRangeSearch<boost::mpl::void_>::Core<TableType>::Main(DataAccessType *data,
    boost::program_options::variables_map &vm) {

  // The window data file is a required parameter.
  if (!vm.count("queries_in")) {
    fl::logger->Die() << "You must specify the --queries_in option";
  }
  std::string window_file_name = vm["queries_in"].as<std::string>();
  boost::shared_ptr<QueryTable_t> query_table;
  fl::logger->Message() << "Loading query data from " << window_file_name;
  boost::shared_ptr<TableType> temp_query_table;
  data->Attach(window_file_name, &temp_query_table);
  data->CopyAndDestruct(temp_query_table, &query_table);
  fl::logger->Message() << "Completed loading";

  // The reference data file is a required parameter.
  if (!vm.count("references_in")) {
    fl::logger->Die() << "You must specify the --references_in option";
  }
  std::string references_file_name = vm["references_in"].as<std::string>();
  boost::shared_ptr<ReferenceTable_t> reference_table;
  fl::logger->Message() << "Loading Reference data from" << references_file_name;
  data->Attach(references_file_name, &reference_table);
  fl::logger->Message() << "Completed loading";

  // The search result.
  boost::shared_ptr<typename DataAccessType::DefaultSparseIntTable_t> output_table;
  if (!vm.count("results_out")) {
    fl::logger->Die() << "You must specify the --results_out option";
  }
  std::string results_out = vm["results_out"].as<std::string>();
  data->Attach(results_out,
               std::vector<index_t>(),
               std::vector<index_t>(1, reference_table->n_entries()),
               query_table->n_entries(),
               &output_table);
  fl::logger->Message() << "Computing orthogonal range search for " << query_table->n_entries()
  << " windows, over " << reference_table->n_entries() << " reference data points";
  fl::ml::OrthoRangeSearch<CoreOrthoArgs>::Compute(
    *query_table, 20, *reference_table, 20,
    output_table.get());
  fl::logger->Message() << "Finished search";
  // Write the result to the file.
  fl::logger->Message() << "Emitting results to " << results_out;
  data->Purge(results_out);
  fl::logger->Message() << "Finished emitting results";
  data->Detach(results_out);

  return 0;
}
template<typename DataAccessType>
void OrthoRangeSearch<boost::mpl::void_>::Run(
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
};
};

#endif
