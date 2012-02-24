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
#ifndef FL_LITE_MLPACK_REGRESSION_MODEL_STATISTICS_H
#define FL_LITE_MLPACK_REGRESSION_MODEL_STATISTICS_H
#include <vector>

namespace fl {namespace ml {
  struct ModelStatistics {
    std::vector<double> standard_errors;
    std::vector<double> t_statistics;
    std::vector<double> confidence_interval_los; 
    std::vector<double> confidence_interval_his; 
    std::vector<double> p_values; 
    double adjusted_r_squared;
    double f_statistic;
    double r_squared;
    double sigma;

    template<typename TableType2>
    void Export(index_t prediction_index,
                TableType2 *standard_errors_table,
                TableType2 *confidence_interval_los_table,
                TableType2 *confidence_interval_his_table,
                TableType2 *t_statistics_table,
                TableType2 *p_values_table,
                TableType2 *adjusted_r_squared_table,
                TableType2 *f_statistic_table,
                TableType2 *r_squared_table,
                TableType2 *sigma_table) {
      index_t attribute=0;
      for(size_t i=0; i<(standard_errors.size()+(prediction_index<0?0:1)); ++i) {
        if (i!=prediction_index) {
          if (standard_errors_table!=NULL) {
            standard_errors_table->set(i, standard_errors[attribute]);
          }
          if (confidence_interval_los_table!=NULL) {
            confidence_interval_los_table->set(i, confidence_interval_los[attribute]);
          }
          if (confidence_interval_his_table!=NULL) {
            confidence_interval_his_table->set(i, confidence_interval_his[attribute]);
          }
          if (t_statistics_table!=NULL) {
            t_statistics_table->set(i, t_statistics[attribute]);
          }
          if (p_values_table!=NULL) {
            p_values_table->set(i, p_values[attribute]);
          }
          attribute++;
        } else {
          if (standard_errors_table!=NULL) {
            standard_errors_table->set(i, 0);
          }
          if (confidence_interval_los_table!=NULL) {
            confidence_interval_los_table->set(i, 0);
          }
          if (confidence_interval_his_table!=NULL) {
            confidence_interval_his_table->set(i, 0);
          }
          if (t_statistics_table!=NULL) {
            t_statistics_table->set(i, 0);
          }
          if (p_values_table!=NULL)
          p_values_table->set(i, 0);

        }
      }
      if (adjusted_r_squared_table!=NULL) {
        adjusted_r_squared_table->set(0, adjusted_r_squared);
      }
      if (f_statistic_table!=NULL) {
        f_statistic_table->set(0, f_statistic);
      }
      if (r_squared_table!=NULL) {
        r_squared_table->set(0, r_squared);
      }
      if (sigma_table!=NULL) {
        sigma_table->set(0, sigma);
      }
    }
  };
}}
#endif
