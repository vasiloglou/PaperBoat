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

#ifndef FL_LITE_MLPACK_NMF_NMF_DEFS_H_
#define FL_LITE_MLPACK_NMF_NMF_DEFS_H_
#include "boost/mpl/if.hpp"
#include "mlpack/nmf/nmf_dev.h"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "boost/lexical_cast.hpp"
#include "fastlib/workspace/task.h"

namespace fl {
namespace ml {
template<typename TableType>
template<typename DataAccessType>
int Nmf::Core<TableType>::Main(DataAccessType *data,
                               boost::program_options::variables_map &vm) {
  FL_SCOPED_LOG(Nmf);
  std::string run_mode=vm["run_mode"].as<std::string>();
  std::string w_factor_in=vm["w_factor_in"].as<std::string>();
  std::string h_factor_in=vm["h_factor_in"].as<std::string>();
  std::string row_str=vm["row"].as<std::string>();
  std::string col_str=vm["col"].as<std::string>();
  std::string v_out=vm["v_out"].as<std::string>();

  if (run_mode=="train") {
    return boost::mpl::if_ <
           typename TableType::Dataset_t::IsDenseOnly_t,
           BppNnlsNmf<boost::mpl::void_>::Core<TableType>,
           SparseNmf<boost::mpl::void_>::Core<TableType>
           >::type::Main(data, vm);
  } else {
    if (run_mode=="eval") {
      if (w_factor_in=="") {
        fl::logger->Die() << "In the eval mode you need to specify --w_factor_in";
      } 
      if (h_factor_in=="") {
        fl::logger->Die() << "In the eval mode you need to specify --h_factor_in";
      }
      if (row_str=="") {
        fl::logger->Die()<< "In the eval mode you need to specify --row";
      }
      if (col_str=="") {
        fl::logger->Die()<< "In the eval mode you need to specify --col";
      } 

      std::vector<int> rows;
      std::vector<int> cols;
      boost::shared_ptr<typename DataAccessType::template TableVector<int> > rows_table;
      boost::shared_ptr<typename DataAccessType::template TableVector<int> > cols_table;
      bool inds_are_tables=false;
      try {
        std::vector<std::string> temp_tokens;
        boost::algorithm::split(temp_tokens, row_str, boost::algorithm::is_any_of(","));
        for(int i=0; i<temp_tokens.size(); ++i) {
          rows.push_back(boost::lexical_cast<int>(temp_tokens[i]));
        }
        boost::algorithm::split(temp_tokens, col_str, boost::algorithm::is_any_of(","));
        for(int i=0; i<temp_tokens.size(); ++i) {
          cols.push_back(boost::lexical_cast<int>(temp_tokens[i]));
        }
      }
      catch(const boost::bad_lexical_cast &e) {
        // read from a file 
        data->Attach(row_str, &rows_table);
        data->Attach(col_str, &cols_table);
        if (rows_table->n_entries()!=cols_table->n_entries()) {
          fl::logger->Die()<<"The lists of --rows --cols must be of the same size";
        }
        inds_are_tables=true;
      }
      if (rows.size()!=cols.size()) {
        fl::logger->Die() << "The lists of --rows --cols must be of the same size";
      }
      boost::shared_ptr<typename DataAccessType::DefaultTable_t> w_table;
      data->Attach(w_factor_in, &w_table);
      boost::shared_ptr<typename DataAccessType::DefaultTable_t> h_table;
      data->Attach(h_factor_in, &h_table);
      typename DataAccessType::DefaultTable_t::Point_t p1, p2;
      boost::shared_ptr<typename DataAccessType::template TableVector<double> > new_table_v;
      if (v_out!="") {
        data->Attach(v_out,
            std::vector<index_t>(1,1),
            std::vector<index_t>(),
            inds_are_tables?rows_table->size():rows.size(),
            &new_table_v);
      }
      if (inds_are_tables==false) {
        for(index_t i=0; i<rows.size(); ++i) {
          w_table->get(rows[i], &p1);
          h_table->get(cols[i], &p2);
          double val=fl::la::Dot(p1, p2);
          if (v_out=="") {
            fl::logger->Message()<<"("<<rows[i]<<","<<cols[i]<<","<<val<<")"<<std::endl;
          } else {
            
          }
        }
      } else {
        for(index_t i=0; i<rows_table->size(); ++i) {
          w_table->get((*rows_table)[i], &p1);
          h_table->get((*cols_table)[i], &p2);
          double val=fl::la::Dot(p1, p2);
          if (v_out!="") {
            new_table_v->set(i,  val);
          }
        }    
      }
      if (v_out!="") {
        data->Purge(v_out);
        data->Detach(v_out);
      }
    } else {
      fl::logger->Die() << "This option (" << run_mode <<") for the"
        " --run_mode flag is not supported";
    }
  }
  return 0;
}

template<typename DataAccessType, typename BranchType>
int Nmf::Main(DataAccessType *data,
              const std::vector<std::string> &args) {
  ////////// READING PARAMETERS AND LOADING DATA /////////////////////
  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Help on non-negative matrix factorization")
  ("references_in", boost::program_options::value<std::string>()->default_value(""),
   "File containing the matrix to be factored. If the matrix is dense then"
   "the algorithm will automatically use the Kim-Park fast algorithm. "
   "If the matrix is sparse, it will pick a combination of "
   "stochastic gradiient descent and LBFGS. "
   "The zeros of the sparse matrix will be treated as missing values")
  ("w_factor_out", boost::program_options::value<std::string>()->default_value("w_factor_out.txt"),
   "The file to which the W factor will be saved.")
  ("h_factor_out", boost::program_options::value<std::string>()->default_value("h_factor_out.txt"),
   "The file to which H transposed factor will be saved.")
  ("w_factor_in", boost::program_options::value<std::string>()->default_value(""),
   "The file of the W factor.")
  ("h_factor_in", boost::program_options::value<std::string>()->default_value(""),
   "The file of the transposed H factor.")
  ("v_out", boost::program_options::value<std::string>()->default_value(""),
   "The file that the predicted values of V will be saved.")
  ("k_rank", boost::program_options::value<index_t>(),
   "The rank of the factorization, if the initial matrix is NxM, then "
   "the resulting matrices will be W: Nxk and H: kxM")
  ("lbfgs_rank", boost::program_options::value<index_t>()->default_value(3),
   "The memory of LBFGS")
  ("lbfgs_steps", boost::program_options::value<index_t>()->default_value(3),
   "The algorithm optimizes W and H alternatively using LBFGS. This parameter "
   " controls the number of steps we need to take to optimize W(H) before switching "
   "to the other matrix")
  ("error", boost::program_options::value<double>()->default_value(0.1),
   "The desired approximation relative error (norm of the error versus the norm of the reference matrix)"
   "that we would like to achieve "
   "within the iterations, currently disabled as a feature.")
  ("w_sparsity_factor", boost::program_options::value<double>()->default_value(0),
   "The sparsity of the W factor, it should be between 0 and 1, 0 means dense, 1 means super sparse")
  ("h_sparsity_factor", boost::program_options::value<double>()->default_value(0),
   "The sparsity of the H factor, it should be between 0 and 1, 0 means dense, 1 means super sparse")
  ("iterations", boost::program_options::value<index_t>()->default_value(-1),
   "number of iterations for running the optimization problem")
  ("epochs", boost::program_options::value<index_t>()->default_value(10),
   "If you run NMF on stochastic gradient descent mode (also know as online mode) then you "
   "should set epochs to a positive number.")
  ("step0", boost::program_options::value<double>()->default_value(1),
   "if you run NMF in stochastic gradient descent mode (also know as online mode) then you "
   "need to set step0 to a positive number. step0 is the step in the first epoch of gradient "
   "descent")
  ("sparse_mode", boost::program_options::value<std::string>()->default_value("stoc_lbfgs"),
   "When you run NMF with sparse data then you have the following options:\n"
   "stoc       : it uses stochastic gradient descent (also known as online)\n"
   "lbfgs      : it uses LBFGS gradient descent\n"
   "stoc_lbfgs : it uses stochastic gradient descent first and then continues with LBFGS\n"
   "If your dataset has a lot of redundancy then stochastic gradient descent will converge faster,"
   "otherwise LBFGS will be more effective.")
  ("log",
   boost::program_options::value<std::string>()->default_value(""),
   "A file to receive the log, or omit for stdout.")
  ("run_mode", 
   boost::program_options::value<std::string>()->default_value("train"),
   " As every machine learning algorithm, NMF has a train mode where the W and H"
   " matrices are computed. In the evaluation mode the V_{ij} element is computed"
   " given i,j. If you want to compute W,H set the --run_mode=train. If you want"
   " to compute V_{ij}, set the --run_mode=eval")
  ("row",
   boost::program_options::value<std::string>()->default_value(""),
   " In the evaluation mode (--run_mode=eval) you need to specify the element of V"
   " that you want to recinstruct. The flag --row is a comma separated list that "
   " specifies the rows of V to be evaluated. For example if we need to evaluate"
   " V_{1,2}, V_{1,55}, V_{4,33}, V_{4,55}, then you should use the following syntax"
   " --row=1,1,4,4 --col=2,55,33,55 . Notice that both lists must have the same size."
   " You can also specify a file name that has the indices instead of a list.")
  ("col",
   boost::program_options::value<std::string>()->default_value(""),
   " In the evaluation mode (--run_mode=eval) you need to specify the element of V"
   " that you want to recinstruct. The flag --col is a comma separated list that "
   " specifies the columns of V to be evaluated. For example if we need to evaluate"
   " V_{1,2}, V_{1,55}, V_{4,33}, V_{4,55}, then you should use the following syntax"
   " --row=1,1,4,4 --col=2,55,33,55 . Notice that both lists must have the same size."
   " You can also specify a file name that has the indices, instead of a list.");

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
    fl::logger->Message() << fl::DISCLAIMER << "\n";
    fl::logger->Message() << desc << "\n";
    return 1;
  }

  // Check argument sanity here.
  if (vm["run_mode"].as<std::string>() != "eval" && vm["references_in"].as<std::string>() == "") {
    fl::logger->Die() << "You need to provide the --references_in argument.\n";
  }
  if (vm["run_mode"].as<std::string>() != "eval" && vm.count("k_rank") == 0) {
    fl::logger->Die() << "You need to provide the --k_rank argument.\n";
  }

  return BranchType::template BranchOnTable<Nmf, DataAccessType>(data, vm);
}

template<typename DataAccessType>
void Nmf::Run(
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

}
} // namespaces
#endif
