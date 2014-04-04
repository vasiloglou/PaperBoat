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

#ifndef FL_LITE_MLPACK_NMF_LBFGS_DEFS_H_
#define FL_LITE_MLPACK_NMF_LBFGS_DEFS_H_
#include "sparse_nmf.h"
#include "fastlib/util/timer.h"
namespace fl {
namespace ml {

template<typename TableType>
template<typename DataAccessType>
int SparseNmf<boost::mpl::void_>::Core<TableType>::Main(DataAccessType *data,
    boost::program_options::variables_map &vm) {
  boost::shared_ptr<TableType> table;
  boost::shared_ptr<typename DataAccessType::DefaultTable_t> w_factor;
  boost::shared_ptr<typename DataAccessType::DefaultTable_t> h_factor;
  index_t k_rank=0;
  try {
    k_rank = vm["k_rank"].as<index_t>();
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --k_rank must be set to an integer";
  }
  if (k_rank <=0) {
    fl::logger->Die() << "k_rank("<< k_rank <<") must be greater than zero";
  }
  try {
    fl::logger->Message() << "Loading data from "
        << vm["references_in"].as<std::string>() << std::endl;
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flags --references_in must be set to a string";
  }
  data->Attach(vm["references_in"].as<std::string>(), &table);
  fl::logger->Message() << "Data loaded" << std::endl;
  fl::logger->Message() << "Initializing w_factor" << std::endl;
  std::string w_factor_out;
  try {
    w_factor_out = vm["w_factor_out"].as<std::string>();
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --w_factor_out must be set to a string";
  }
  data->Attach(w_factor_out,
               std::vector<index_t>(1, k_rank),
               std::vector<index_t>(),
               table->n_entries(),
               &w_factor);
  fl::logger->Message() << "Initialized w_factor" << std::endl;
  std::string h_factor_out;
  try {
    h_factor_out = vm["h_factor_out"].as<std::string>();
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --h_factor_out must be set to a string";
  }
  fl::logger->Message() << "Initializing h_factor" << std::endl;
  data->Attach(h_factor_out,
               std::vector<index_t>(1, k_rank),
               std::vector<index_t>(),
               table->n_attributes(),
               &h_factor);
  fl::logger->Message() << "Initialized h_factor" << std::endl;
  typedef typename DataAccessType::DefaultTable_t FactorsTable_t;

  typedef fl::ml::SparseNmf<Args<FactorsTable_t> > NmfEngine_t;
  NmfEngine_t nmf;
  nmf.Init(table.get(),
           w_factor.get(),
           h_factor.get());
  nmf.set_rank(k_rank);
  // do some argument checking
  try {
    nmf.set_iterations(vm["iterations"].as<index_t>());
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --iterations must be set to an integer";
  }
  try {
    if (vm["w_sparsity_factor"].as<double>()>1 ||
        vm["w_sparsity_factor"].as<double>()<0) {
      fl::logger->Die()<<"--w_sparse_factor must be between 0 and 1";
    }
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --w_sparsity_factor must be set to a float";
  }
  nmf.set_w_sparsity_factor(vm["w_sparsity_factor"].as<double>());
  try {
    if (vm["h_sparsity_factor"].as<double>()>1 ||
        vm["h_sparsity_factor"].as<double>()<0) {
      fl::logger->Die()<<"--h_sparse_factor must be between 0 and 1";
    }
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --h_sparsity_factor must be set to a float";
  }
  nmf.set_h_sparsity_factor(vm["h_sparsity_factor"].as<double>());
  try {
    if (vm["lbfgs_rank"].as<index_t>()<=0) {
      fl::logger->Die() <<"--lbfgs_rank must be positive";
    } else {
      if (vm["lbfgs_rank"].as<index_t>()>10) {
        fl::logger->Warning()<<"--lbfgs_rank seems to be too high";
      }
    }
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --lbfgs_rank must be set to a float";
  }
  nmf.set_lbfgs_rank(vm["lbfgs_rank"].as<index_t>());
  if (vm["lbfgs_steps"].as<index_t>() <= 0) {
    fl::logger->Die()<< "--lbgs_steps must be postive";
  }
  try {
    nmf.set_lbfgs_steps(vm["lbfgs_steps"].as<index_t>());
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --lbfgs_steps must be set to an integer";
  }
  try {
    if (vm["step0"].as<double>()<=0) {
      fl::logger->Die() << "--step0 must be positive";
    }
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --step0 must be set to an integer";
  }
  try {
    nmf.set_step0(vm["step0"].as<double>());
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --step0 must be set to a float";
  }
  try {
    if (vm["epochs"].as<index_t>()<=0) {
      fl::logger->Die() << "--epochs must be positive";
    }
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die()<<"Flag --epochs must be set to an integer";
  }
  nmf.set_epochs(vm["epochs"].as<index_t>());
  if (vm["sparse_mode"].as<std::string>().find_first_of("stoc")==std::string::npos
      && vm["sparse_mode"].as<std::string>().find_first_of("lbfgs")==std::string::npos) {
    fl::logger->Die()<<"--sparse_mode must contain a combination of the strings stoc, lbfgs";
  }
  fl::logger->Message() << "Started training NMF for k=" <<k_rank<<std::endl;
  fl::util::Timer timer;
  timer.Start();
  try {
    nmf.Train(vm["sparse_mode"].as<std::string>());
  }
  catch(const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --sparse_mode must be set to a string";
  }
  timer.End();
  fl::logger->Message() << "Finished training after " 
    << timer.GetTotalElapsedTime() <<"sec" << std::endl;
  fl::logger->Message() << "Emmiting results" << std::endl;
  data->Purge(w_factor_out);
  data->Purge(h_factor_out);
  data->Detach(w_factor_out);
  data->Detach(h_factor_out);
  return 0;
}

template<typename DataAccessType, typename BranchType>
int SparseNmf<boost::mpl::void_>::Main(DataAccessType *data,
                                      const std::vector<std::string> &args) {

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////
  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Help on non-negative matrix factorization")
  ("references_in", boost::program_options::value<std::string>(),
   "File containing the matrix")
  ("w_sparsity_factor",
   boost::program_options::value<double>()->default_value(0.5),
   "The sparsity level for each column of the W factor."
  )
  ("h_sparsity_factor",
   boost::program_options::value<double>()->default_value(0.5),
   "The sparsity level for each row of the H factor."
  )
  ("w_factor_out", boost::program_options::value<std::string>(),
   "w_factor output")
  ("h_factor_out", boost::program_options::value<std::string>(),
   "h_factor output")
  ("k_rank", boost::program_options::value<index_t>(),
   "The number of columns to extract")
  ("iterations",
   boost::program_options::value<index_t>()->default_value(100),
   "number of iterations for running the optimization problem")
  ("error", boost::program_options::value<double>()->default_value(0.1),
   "The desired approximation relative error (norm of the error versus the norm of the reference matrix)"
   "that we would like to achieve "
   "within the iterations")
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
         SparseNmf<boost::mpl::void_>, DataAccessType > (data, vm);

}


}
} // namespaces

#endif
