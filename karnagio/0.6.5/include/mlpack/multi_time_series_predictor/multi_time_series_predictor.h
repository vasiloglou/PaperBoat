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

#ifndef PAPERBOAT_KARNAGIO_INCLUDE_MLPACK_MULTI_TIME_SERIES_PREDICTION_H_
#define PAPERBOAT_KARNAGIO_INCLUDE_MLPACK_MULTI_TIME_SERIES_PREDICTION_H_
#include "fastlib/base/base.h"

namespace fl { namespace ml {

  /**
   * @brief Given a table that each column is a time series we develop a model 
   *        using svd and npr so that we can predict the values of the next time 
   *        stamps
   */
  template<typename WorkSpaceType>
  class MultiTsPredictor {
    public:
      template<typename TableType>
      static void PrepareNprData(
          WorkSpaceType *ws,
          TableType &references_table,
          const std::string &new_table,
          typename WorkSpaceType::DefaultTable_t &low_dim_references_table,
          typename WorkSpaceType::DefaultTable_t &augment_table, 
          index_t timestamp_attribute,
          int32 time_lag,
          std::vector<index_t> *timestamp_breakpoints,
          std::vector<index_t> *reference_breaks) ;
      
      template<typename TableType>
      static void PrepareNprQueryData(
          WorkSpaceType *ws,
          TableType &references_table,
          const std::vector<index_t> reference_breaks,
          typename WorkSpaceType::DefaultTable_t &low_dim_references_table,
          typename WorkSpaceType::DefaultTable_t &augment_table, 
          index_t timestamp_attribute,
          int32 time_lag,
          const std::string &query_table_name,
          boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *query_table);
 

  };
 
  template<>
  class MultiTsPredictor<boost::mpl::void_> {
    public:
      template<typename WorkSpaceType>
      struct Core {
        public:
          Core(WorkSpaceType *ws, const std::vector<std::string> &args);
          template<typename TableType>
          void operator()(TableType&);
        private:
          WorkSpaceType *ws_;
          std::vector<std::string> args_;
      };
    
      template <typename WorkSpaceType, typename BranchType>
      static int Main(WorkSpaceType *ws, const std::vector<std::string> &args);
    

      template<typename WorkSpaceType>
      static int Run(WorkSpaceType *data,
          const std::vector<std::string> &args);
 };

}}

#endif
