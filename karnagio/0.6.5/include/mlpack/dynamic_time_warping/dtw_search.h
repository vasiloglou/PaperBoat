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

#ifndef PAPERBOAT_MLPACK_DYNAMIC_TIME_WARPING_DTW_SEARCH_H_
#define PAPERBOAT_MLPACK_DYNAMIC_TIME_WARPING_DTW_SEARCH_H_


#include "boost/program_options.hpp"
#include "boost/mpl/void.hpp"
#include "boost/shared_ptr.hpp"
#include "fastlib/workspace/workspace.h"
#include "fastlib/workspace/based_on_table_run.h"

namespace fl {namespace ml{
  template<typename>
  class DtwSearch;

  template<>
  class DtwSearch<boost::mpl::void_> {
    public:
      template<typename WorkSpaceType1>
      struct Core {
        public:
          Core(WorkSpaceType1 *ws, const std::vector<std::string> &args);
          template<typename TableType>
          void operator()(TableType&);
         
         template<typename TableType>
         void  TransformQueriesReferences(
              TableType &references_table,
              TableType &queries_table, 
              std::vector<std::vector<double> > *references, 
              std::vector<std::vector<double> > *queries);

          template<typename TableType, 
                   typename IndicesTableType,
                   typename DistancesTableType>
          static void ComputeNearestNeighbors(
            TableType &references_table, 
            TableType &query_table, 
            int32 k_neighbors,
            double scaling_factor,
            index_t horizon,
            IndicesTableType *indices, 
            DistancesTableType *distances);
          template<typename TableType>
          static double DistanceLB(
              TableType &queries, 
              index_t i, 
              TableType &references, 
              index_t j,
              index_t scaling_factor,
              index_t horizon);

          static double DistanceLB(
              std::vector<std::vector<double> > &queries, 
              index_t i, 
              std::vector<std::vector<double> > &references, 
              index_t j,
              index_t scaling_factor,
              index_t horizon);

          template<typename TableType>
          static double Distance(
              TableType &queries, 
              index_t i, 
              TableType &references, 
              index_t j, 
              double scaling_factor,
              index_t horizon);

          static double Distance(
              std::vector<std::vector<double> > &queries, 
              index_t i, 
              std::vector<std::vector<double> > &references, 
              index_t j, 
              double scaling_factor, 
              index_t horizon);

          static index_t GetEntries(const std::vector<std::vector<double> > &x);

          template<typename TableType>
          static index_t GetEntries(const TableType &x);

        private:
          WorkSpaceType1 *ws_;
          std::vector<std::string> args_;
      };
    
      template<typename WorkSpaceType>
      static int Run(WorkSpaceType *data,
          const std::vector<std::string> &args);

  };
}}
#endif
