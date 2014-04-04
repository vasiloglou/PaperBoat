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
#ifndef FL_LITE_MLPACK_NMF_BPP_NNLS_NMF_DEFS_H
#define FL_LITE_MLPACK_NMF_BPP_NNLS_NMF_DEFS_H

#include <iostream>
#include <vector>
#include "boost/program_options.hpp"
#include "boost/mpl/if.hpp"
#include "fastlib/base/base.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/table/table.h"
#include "fastlib/tree/similarity_tree.h"
#include "bpp_nnls_nmf.h"

namespace fl {
namespace ml {

template<typename DataAccessType, typename BranchType>
int BppNnlsNmf<boost::mpl::void_>::Main(DataAccessType *data,
                                        const std::vector<std::string> &args) {

  srand(time(NULL));
  ////////// READING PARAMETERS AND LOADING DATA /////////////////////
  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Help on non-negative matrix factorization")
  ("references_in", boost::program_options::value<std::string>(),
   "File containing the matrix")
  ("w_factor_out", boost::program_options::value<std::string>()->default_value("w_factor_out.txt"),
   "The file to which the realized weight vectors are written.")
  ("h_factor_out", boost::program_options::value<std::string>()->default_value("h_factor_out.txt"),
   "The file to which h factors are written.")
  ("k_rank", boost::program_options::value<index_t>(),
   "The number of columns to extract")
  ("iterations",
   boost::program_options::value<index_t>()->default_value(-1),
   "number of iterations for running the optimization problem")
  ("log",
   boost::program_options::value<std::string>()->default_value(""),
   "A file to receive the log, or omit for stdout.")
  ("loglevel",
   boost::program_options::value<std::string>()->default_value("debug"),
   "Level of log detail.  One of:\n"
   "  debug: log everything\n"
   "  verbose: log messages and warnings\n"
   "  warning: log only warnings\n"
   "  silent: no logging");

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::
                                command_line_parser(args).options(desc).run(), vm);
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout << fl::DISCLAIMER << "\n";
    std::cout << desc << "\n";
    return 1;
  }
  return BranchType::template BranchOnTable <
         BppNnlsNmf<boost::mpl::void_>, DataAccessType > (data, vm);

}

template<typename TableType>
template<typename DataAccessType>
int BppNnlsNmf<boost::mpl::void_>::Core<TableType>::Main(
  DataAccessType *data,
  boost::program_options::variables_map &vm) {

  // Start the computation.
  // Read in the dataset.

  boost::shared_ptr<TableType> table;

  // The reference data file is a required parameter.
  std::string references_file_name;
  try {
    references_file_name = vm["references_in"].as<std::string>();
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --references_in must be set to a string";
  }
  fl::logger->Message() << "Loading data from " <<
  vm["references_in"].as<std::string>() << std::endl;
  data->Attach(references_file_name, &table);
  fl::logger->Message() << "Data loaded" << std::endl;

  // The number of iterations.
  int num_iterations=0; 
  try {
    num_iterations = vm["iterations"].as<index_t>();
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --iterations must be set to an integer";
  }

  // The factors.
  fl::dense::Matrix<double, false> w_factor;
  fl::dense::Matrix<double, false> h_factor;
  
  index_t rank;
  try { 
    rank = vm["k_rank"].as<index_t>();
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --k_rank must be set to an integer";
  }
  fl::dense::Matrix<double, false> input_data;
  boost::mpl::if_ <
    boost::is_same <
      typename DataAccessType::DefaultTable_t::IsMatrixOnly_t,
      boost::mpl::bool_<true>
    > ,
    AliasMatrix,
    CopyMatrix >::type::Init(*table, &input_data);
  fl::ml::BppNnlsNmf<TableType>::Compute(
    input_data, rank, num_iterations, &w_factor, &h_factor);

  // Write the result to the file.
  boost::shared_ptr<typename DataAccessType::DefaultTable_t> w_factor_table;
  boost::shared_ptr<typename DataAccessType::DefaultTable_t> h_factor_table;
  std::string w_factor_out;
  try { 
    w_factor_out = vm["w_factor_out"].as<std::string>();
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --w_factor_out must be set to a string";
  }
  data->Attach(w_factor_out,
               std::vector<index_t>(1, rank),
               std::vector<index_t>(),
               input_data.n_rows(),
               &w_factor_table);
  std::string h_factor_out;
  try {
    h_factor_out = vm["h_factor_out"].as<std::string>();
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --h_factor_out must be set to a string";
  }
  data->Attach(h_factor_out,
               std::vector<index_t>(1, rank),
               std::vector<index_t>(),
               input_data.n_cols(),
               &h_factor_table);
  for (index_t i = 0; i < w_factor_table->n_entries(); ++i) {
    typename DataAccessType::DefaultTable_t::Point_t point;
    w_factor_table->get(i, &point);
    for (index_t j = 0; j < w_factor_table->n_attributes(); ++j) {
      point.set(j, w_factor.get(i, j));
    }
  }
  for (index_t i = 0; i < h_factor_table->n_entries(); ++i) {
    typename DataAccessType::DefaultTable_t::Point_t point;
    h_factor_table->get(i, &point);
    for (index_t j = 0; j < h_factor_table->n_attributes(); ++j) {
      point.set(j, h_factor.get(j, i));
    }
  }

  data->Purge(w_factor_out);
  data->Purge(h_factor_out);
  data->Detach(w_factor_out);
  data->Detach(h_factor_out);
  fl::logger->Message() << "Saved the w factor to " <<
  w_factor_out << ".";
  fl::logger->Message() << "Saved the h factor to " <<
  h_factor_out << ".";

  return 0;
}

}
}

#endif
