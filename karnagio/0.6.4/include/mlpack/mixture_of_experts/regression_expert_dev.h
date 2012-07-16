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

#ifndef FL_LITE_INCLUDE_MLPACK_MIXTURE_OF_EXPERTS_REGRESSION_EXPERT_DEV_H_
#define FL_LITE_INCLUDE_MLPACK_MIXTURE_OF_EXPERTS_REGRESSION_EXPERT_DEV_H_
#include "regression_expert.h"
#include "mlpack/regression/linear_regression.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/util/string_utils.h"
#include "boost/lexical_cast.hpp"

namespace fl {
  namespace ml {
    template<typename TableType, typename WorkSpaceType>
    void RegressionExpert<TableType, WorkSpaceType>::Build() {
      FL_SCOPED_LOG(Expert);
      fl::ws::WorkSpace ws;
      ws.set_schedule_mode(2);
      if (references_->n_entries()==0) {
        fl::logger->Die()<<"Cluster is empty";
      }
      ws.LoadTable("references", references_);
      if (log_==false) {
        logger->SuspendLogging();
      }
      LinearRegression<boost::mpl::void_>::Main<fl::ws::WorkSpace, typename fl::ws::WorkSpace::Branch_t>(&ws, arguments_); 
      if (log_==false) {
        logger->ResumeLogging();
      }
      //boost::shared_ptr<typename fl::ws::WorkSpace::DefaultTable_t > standard_errors_table;
      //ws.Attach("standard_errors", &standard_errors_table);
      boost::shared_ptr<typename fl::ws::WorkSpace::DefaultTable_t> sigma_table;
      ws.Attach("sigma", &sigma_table);
      score_=sigma_table->get(0, index_t(0));//+standard_errors_table->operator[](coeff_index_);
      ws.Attach("coefficients", &coefficients_table_); 
    }
     
    template<typename TableType, typename WorkSpaceType>
    double RegressionExpert<TableType, WorkSpaceType>::Evaluate(const Point_t &point) {
      if (references_->n_entries()==0) {
        return std::numeric_limits<double>::max();
      }
      double prediction=0;
      for(typename Point_t::iterator it=point.begin();
          it!=point.end(); ++it) {
        if (coeff_index_!=it.attribute()) {
          prediction+=coefficients_table_->get(it.attribute(), index_t(0))*it.value();
        }
      }
      prediction+=coefficients_table_->get(coeff_index_, index_t(0));
      double error=fabs(prediction-point[coeff_index_]);
      return error;
    }   
      
    template<typename TableType, typename WorkSpaceType>
    double RegressionExpert<TableType, WorkSpaceType>::score() {
      return score_;
    }

    template<typename TableType, typename WorkSpaceType>
    void RegressionExpert<TableType, WorkSpaceType>::set(boost::shared_ptr<Table_t> &table) {
     /* 
      references_.reset(new TableType());
      references_->Init("",
          table->dense_sizes(),
          table->sparse_sizes(),
          0);
      typename TableType::Point_t point;
      for(index_t i=0; i<table->n_entries(); ++i) {
        table->get(i, &point);
        references_->push_back(point);
      }
      references_->labels()=table->labels(); 
      */
      references_=table;
      //table->CloneDataOnly(references_.get());
    }
    
    template<typename TableType, typename WorkSpaceType>
    void RegressionExpert<TableType, WorkSpaceType>::set_coeff_index(
        index_t coeff_index) {
      coeff_index_=coeff_index; 
    }

    
    template<typename TableType, typename WorkSpaceType>
    void RegressionExpert<TableType, WorkSpaceType>::set_args(const std::vector<std::string> &arguments) {
      arguments_=fl::ws::MakeArgsFromPrefix(arguments, "");
      std::map<std::string, std::string> argmap=fl::ws::GetArgumentPairs(arguments_);
      if (argmap.count("--prediction_index_prefix")==0) {
        fl::logger->Die()<<"You should set the --prediction_index_prefix "
         "for the regressor expert"; 
      }
      coeff_index_=boost::lexical_cast<index_t>(argmap["--prediction_index_prefix"]);
      if (argmap.count("--references_in=references")) {
        fl::logger->Die()<<"You are not allowed to set "
          "--references_in in the expert arguments";     
      }
      arguments_.push_back("--references_in=references");
      if (argmap.count("--standard_errors_out")) {
        fl::logger->Die()<<"You are not allowed to set "
          "--standard_errors_out in the expert arguments";     
      }
      arguments_.push_back("--standard_errors_out=standard_errors");    
      if (argmap.count("--coeffs_out")) {
        fl::logger->Die()<<"You are not allowed to set "
          "--coeffs_out in the expert arguments";     
      }
      arguments_.push_back("--coeffs_out=coefficients");
      if (argmap.count("--sigma_out")) {
        fl::logger->Die()<<"You are not allowed to set "
          "--sigma_out in the expert arguments";     
      }

      arguments_.push_back("--sigma_out=sigma");
      if (argmap.count("--check_columns")) {
        fl::logger->Die()<<"You are not allowed to set "
          "--check_columns in the expert arguments";     
      }
      arguments_.push_back("--check_columns=1");
    }
   
    template<typename TableType, typename WorkSpaceType>
    void RegressionExpert<TableType, WorkSpaceType>::set_log(bool log) {
      log_=log;
    }
    
    template<typename TableType, typename WorkSpaceType>
    index_t RegressionExpert<TableType, WorkSpaceType>::cardinality() const {
      return references_->n_entries();
    }

  }
}


#endif
