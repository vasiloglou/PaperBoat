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

#ifndef PAPERBOAT_FASTLIB_ERROR_ENGINE_ERROR_ENGINE_H_
#define PAPERBOAT_FASTLIB_ERROR_ENGINE_ERROR_ENGINE_H_
#include <sstream>

class ErrorEngine {
  public:
    ErrorEngine() {
      Reset();
    }
    void Reset() {
      l1_norm_ = 0;
      l2_norm_ = 0;
      mape_ = 0;
      count_ = 0;
      mape_count_ = 0;
      square_error_ = 0;
      abs_error_= 0;
      max_error_ = 0;
      smape_norm_ = 0;
      positive_error_=0;
      negative_error_=0;
      negative_forecasts_=0;
      positive_forecasts_=0;
    }
    void AddErrorMetric(const std::string& error_metric);
    void AddPrediction(double actual_value, double prediction) {
      l1_norm_ += fabs(actual_value);
      l2_norm_ += actual_value * actual_value;
      double diff=actual_value-prediction;
      double abs_diff=fabs(diff);
      if (actual_value!=0) {
        mape_ += abs_diff/fabs(actual_value);
        mape_count_+=1;
      }
      count_+=1;
      square_error_+=fl::math::Pow<double,2,1>(abs_diff);
      abs_error_+=abs_diff;
      max_error_=std::max(max_error_, abs_diff);
      smape_norm_+=actual_value+prediction;
      if (diff<0) {
        positive_error_+=abs_diff;
        positive_forecasts_+=1;
      } else {
        if (diff>0) {
          negative_error_+=abs_diff;
          negative_forecasts_+=1;
        }
      }
    }
    
    template<typename WorkSpaceType, typename TableType, typename PredictorType>
    void AddQueries(WorkSpaceType *ws, const std::vector<std::string> &table_names, 
        PredictorType &predictor) {
      for(const auto &table_name : table_names) {
        AddQueries<WorkSpaceType, TableType, PredictorType>(
            ws, table_name, predictor);
      }
    }
    
    template<typename WorkSpaceType, typename TableType, typename PredictorType>
    void AddQueries(WorkSpaceType *ws, const std::string &table_name, 
        PredictorType &predictor) {  
      boost::shared_ptr<TableType> table;
      typename TableType::Point_t point;
      ws->Attach(table_name, &table);
      for(index_t i=0; i<table->n_entries(); ++i) {
        table->get(i, &point);
        AddPrediction(point.meta_data().template get<1>(), predictor.Evaluate(point));
      }
      ws->Purge(table_name);
      ws->Detach(table_name);
    }

    std::unordered_map<std::string, double> GetErrors() {
      FL_SCOPED_LOG(ErrorEngine);
      std::unordered_map<std::string, double> errors;
      if (l2_norm_==0) {
        fl::logger->Warning()<<"l2_norm is zero";
      }
      errors["MSE"]=100*fl::math::Pow<double,1,2>(square_error_/l2_norm_);
      if (mape_count_==0) {
        fl::logger->Message()<<"mape count is zero";    
      }
      errors["MAPE"]=100*mape_/mape_count_;
      errors["SMAPE"]=100*abs_error_/smape_norm_;
      if (l1_norm_==0) {
        fl::logger->Message()<<"l1_norm is zero";
      }
      errors["AdjMAPE"]=100*abs_error_/l1_norm_;
      errors["negative_error"]=100*negative_error_/abs_error_;
      errors["negative_forecasts"]=100.0*negative_forecasts_/(negative_forecasts_+positive_forecasts_);
      errors["positive_error"]=100*positive_error_/abs_error_;
      errors["positive_forecasts"]=100.0*positive_forecasts_/(negative_forecasts_+positive_forecasts_);
      return errors;
    }

    std::string ReportError() {
      // std::cout<<l1_norm_/count_<<std::endl;
      std::string message;
      message+="RMSPE="+Truncate(100*fl::math::Pow<double,1,2>(square_error_/l2_norm_))+"%"
        +", MAPE="+Truncate(100*mape_/mape_count_)+"%"
        +", SMAPE="+Truncate(100*abs_error_/smape_norm_)+"%"
        +", Adj MAPE="+Truncate(100*abs_error_/l1_norm_)+"%"
        +", negative_error="+Truncate(100*negative_error_/abs_error_)+"%"
        +", negative_forecasts="+Truncate(100.0*negative_forecasts_/(negative_forecasts_+positive_forecasts_))+"%"
        +", positive_error="+Truncate(100*positive_error_/abs_error_)+"%"
        +", positive_forecasts="+Truncate(100.0*positive_forecasts_/(negative_forecasts_+positive_forecasts_))+"%";
      return message;
    }
  
  protected:
    double smape_norm_;
    double l1_norm_;
    double l2_norm_;
    double mape_;
    index_t count_;
    index_t mape_count_;
    double square_error_;
    double abs_error_;
    double max_error_;
    double positive_error_;
    double negative_error_;
    index_t negative_forecasts_;
    index_t positive_forecasts_;

    std::string Truncate(double x) {
      std::ostringstream ss;
      ss <<std::setprecision(4)<< x;
      return ss.str();
    }
};

#endif

