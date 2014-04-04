/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
LLC, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE ISMION INC "AS IS" AND ANY
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

#ifndef PAPERBOAT_MLPACK_LOGISTIC_TENSOR_FACTORIZATION_DEFS_H_
#define PAPERBOAT_MLPACK_LOGISTIC_TENSOR_FACTORIZATION_DEFS_H_
#include "logistic_tensor_factorization.h"
#include "fastlib/optimization/lbfgs/lbfgs_dev.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/workspace/based_on_table_run.h"


namespace fl { namespace ml {
  template<typename ArgsType>
  void LogisticTensorFactorization<ArgsType>::BatchTrainer::Init(index_t num_dimensions) {
    num_dimensions_=num_dimensions;
  } 
 
  template<typename ArgsType>
  double LogisticTensorFactorization<ArgsType>::BatchTrainer::Evaluate(const fl::data::MonolithicPoint<double> &model){
    std::unordered_map<index_t, Matrix_t> r_x_a_cache;
    Deserialize(model, &a_mat_, &r_mat_);
    Point_t point;
    double error=0;
    Matrix_t row_vec, col_vec;
    for(index_t i=0; i<references_->n_entries(); ++i) {
      references_->get(i, &point);
      for(auto it=point.begin(); it!=point.end(); ++it) {
        a_mat_.MakeColumnVector(i, &i_vec);
        if (r_x_a_cache.count(it.attribute())>0) {
          col_vec.Alias(r_x_a_cache[it.attribute()]);
        } else {
          r_x_a_cache[it.attribute()]=Matrix_t();
          Matrix_t vec;
          a_mat_.MakeColumVector(it.attribute(), &vec);
          fl::dense::ops::Mul<fl::la::Init>(r_mat_, vec, &r_x_a_cache[it.attribute()]);
          col_vec.Alias(r_x_a_cache[it.attribute()]);
        }
        value=fl::dense::ops::Dot(row_vec, col_vec);
        value=Objective_t::Transform(value);
        error+=fl::math::Pow<double,2,1>(value-it.value());
      }
    }
    return error;
  }
 
  template<typename ArgsType>
  double LogisticTensorFactorization<ArgsType>::BatchTrainer::Gradient(const fl::data::MonolithicPoint<double> &model,
      fl::data::MonolithicPoint<double> *gradient) {
    // objective
    // -\sum x_{ijk} log(logit) + (1-x_{ijk})log(1-logit)
    // derrivative
    // -\sum x_{ijk} (-1)/logit *logit' + (1-x_{ijk})*(logit')/(1-logit)
    std::unordered_map<index_t, Matrix_t> r_x_a_cache;
    Deserialize(model, &a_mat_, &r_mat_);
    gradient->SetAll(0);
    Point_t point;
    double error=0;
    Matrix_t row_vec, col_vec;
    for(index_t i=0; i<references_->n_entries(); ++i) {
      references_->get(i, &point);
      for(auto it=point.begin(); it!=point.end(); ++it) {
        a_mat_.MakeColumnVector(i, &i_vec);
        if (r_x_a_cache.count(it.attribute())>0) {
          col_vec.Alias(r_x_a_cache[it.attribute()]);
        } else {
          r_x_a_cache[it.attribute()]=Matrix_t();
          Matrix_t vec;
          a_mat_.MakeColumVector(it.attribute(), &vec);
          fl::dense::ops::Mul<fl::la::Init>(r_mat_, vec, &r_x_a_cache[it.attribute()]);
          col_vec.Alias(r_x_a_cache[it.attribute()]);
        }
        value=fl::dense::ops::Dot(row_vec, col_vec);
        auto logit=1.0/(1.0+exp(-value));
        auto one_minus_logit=1-logit;
        auto term1 = 1.0/logit * (logit * logit)
        auto term2 = one_minus_logit * 1.0/one_minus_logit

      }
    }

  }

  template<typename WorkSpaceType>
  template<typename TableType>
  void LogisticTensorFactorization<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
    FL_SCOPED_LOG(LogisitcTensorFactorization);
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_prefix_in",
      boost::program_options::value<std::string>(),
      "the reference data prefix"
    )(
      "references_num_in",
      boost::program_options::value<int32>(),
      "number of references file with the prefix defined above"
    )(
      "references_in",
       boost::program_options::value<std::string>(),
      "the reference data comma separated. Each file is a time series"
      " eg --references_in=myfile,yourfile,hisfile"
    )(
      "training_objective",
      boost::program_options::value<std::string>()->default_value("parafac"),
      "The method for factoring a 3 dimensional tensor, it can be \n"
      "  least_square\n"
      "  logistic\n"
    )(
      "a_factor_out",
      boost::program_options::value<std::string>(),
      "The a factor of the RESCAL  factorization X_k=AR_kA^T" 
    )(
      "r_factor_out",
      boost::program_options::value<std::string>(),
      "The r factor of the PARAFAC factorization X_=AR_kA^T"
    )(
      "a_regularization",
      boost::program_options::value<double>()->default_value(0.0),
      "The regularization for the a matrix"  
    )(
      "r_regularization",
      boost::program_options::value<double>()->default_value(0.0),
      "The regularization for the r matrix"
    )(
      "rank",
      boost::program_options::value<int32>()->default_value(5),
      "The factorization rank"
    );

    boost::program_options::variables_map vm;
    std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args_, "");
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
      return ;
    }

    fl::ws::RequiredArgValues(vm, "method:parafac", 
        "algorithm:cpwopt_lbfgs,"
        "algorithm:cpwopt_sgd,algorithm:cpwopt_sgd_lbfgs");
    fl::ws::RequiredArgValues(vm, "method:dedicom", 
        "algorithm:dedicom_lbfgs");

    std::vector<boost::shared_ptr<TableType> > tensor;
    std::vector<std::string> filenames=fl::ws::GetFileSequence("references", vm);

    tensor.resize(filenames.size());
    fl::logger->Message()<<"Loaded tensor"<<std::endl;
    for(size_t i=0; i<tensor.size(); ++i) {
      ws_->Attach(filenames[i],
          &tensor[i]);
    }
    fl::ws::RequiredArgs(vm, "rank");
    int32 rank=vm["rank"].as<int32>();
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> a_table;
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> b_table;
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> c_table;
  }

  template<typename WorkSpaceType>
  int LogisticTensorFactorization<boost::mpl::void_>::Run(
      WorkSpaceType *ws,
      const std::vector<std::string> &args) {

    bool found=false;
    std::string references_in;
    for(size_t i=0; i<args.size(); ++i) {
      if (fl::StringStartsWith(args[i],"--references_prefix_in=")) {
        found=true;
        std::vector<std::string> tokens=fl::SplitString(args[i], "=");
        if (tokens.size()!=2) {
          fl::logger->Die()<<"Something is wrong with the --references_in flag";
        }
        references_in=ws->GiveFilenameFromSequence(tokens[1], 0);
        break;
      }
      if (fl::StringStartsWith(args[i],"--references_in=")) {
        found=true;
        std::vector<std::string> tokens=fl::SplitString(args[i], "=");
        if (tokens.size()!=2) {
          fl::logger->Die()<<"Something is wrong with the --references_in flag";
        }
        std::vector<std::string> filenames=fl::SplitString(tokens[1], ":,"); 
        references_in=filenames[0];
        break;
      }
    }

    if (found==false) {
      Core<WorkSpaceType> core(ws, args);
      typename WorkSpaceType::DefaultTable_t t;
      core(t);
      return 1;
    }

    Core<WorkSpaceType> core(ws, args);
    fl::ws::BasedOnTableRun(ws, references_in, core);
    return 0;
  }

  template<typename WorkSpaceType>
  LogisticTensorFactorization<boost::mpl::void_>::Core<WorkSpaceType>::Core(
     WorkSpaceType *ws, const std::vector<std::string> &args) :
   ws_(ws), args_(args)  {}




}}
