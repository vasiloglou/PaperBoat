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

#ifndef PAPERBOAT_MLPACK_OASIS_OASIS_H_
#define PAPERBOAT_MLPACK_OASIS_OASIS_H_
#include <unordered_map>
#include <string>
#include <vector>
#include "fastlib/base/base.h"
#include "boost/mpl/void.hpp"
#include "boost/program_options.hpp"

namespace fl { namespace ml {

  template<typename TableType>
  class Oasis {
    public:
      template<typename MatrixTableType>
      static void Update( 
          const typename TableType::Point_t &point,
          const typename TableType::Point_t &point_plus,
          const typename TableType::Point_t &point_minus, 
          double regularizer_c,
          bool is_nonnegative,
          MatrixTableType *metric_matrix);

      template<typename MatrixTableType>
      static void RegularizeAndProject( 
          double eta,
          bool is_nonnegative,
          double lambda_regularization,
          MatrixTableType *metric_matrix);

      template<typename MatrixTableType>
      static double L_w(const typename TableType::Point_t &point, 
          const typename TableType::Point_t &point_plus, 
          const typename TableType::Point_t &point_minus, 
          const MatrixTableType &w_matrix);
      template<typename MatrixTableType>
      static bool IsItCorrect(const typename TableType::Point_t &point, 
          const typename TableType::Point_t &point_plus, 
          const typename TableType::Point_t &point_minus, 
          const MatrixTableType &w_matrix,
          typename MatrixTableType::Point_t *score_point);
      template<typename MatrixTableType, typename IntegerTableType>
      static void ComputeTrainPrecision(
        IntegerTableType &triplets_table, 
        TableType &reference_table,
        MatrixTableType &weight_matrix,
        double *score,
        index_t *num_of_triplets,
        MatrixTableType *score_table);
  };
 
  template<typename TableType>
  class OasisLowRank {
    public:
      template<typename MatrixTableType>
      static void Update( 
          const typename TableType::Point_t &point,
          const typename TableType::Point_t &point_plus,
          const typename TableType::Point_t &point_minus, 
          double regularizer_c,
          bool is_nonnegative,
          MatrixTableType *metric_matrix);
 
      template<typename MatrixTableType>
      static void RegularizeAndProject( 
          double eta,
          bool is_nonnegative,
          double lambda_regularization,
          MatrixTableType *metric_matrix);
     
      template<typename MatrixTableType>
      static double L_w(const typename TableType::Point_t &point, 
          const typename TableType::Point_t &point_plus, 
          const typename TableType::Point_t &point_minus, 
          const MatrixTableType &w_matrix);
      template<typename MatrixTableType>
      static bool IsItCorrect(const typename TableType::Point_t &point, 
          const typename TableType::Point_t &point_plus, 
          const typename TableType::Point_t &point_minus, 
          const MatrixTableType &w_matrix,
          typename MatrixTableType::Point_t *score_point);
      template<typename MatrixTableType, typename IntegerTableType>
      static void ComputeTrainPrecision(
        IntegerTableType &triplets_table, 
        TableType &reference_table,
        MatrixTableType &weight_matrix,
        double *score,
        index_t *num_of_triplets,
        MatrixTableType *score_table);
  };
 
  template<>
  class Oasis<boost::mpl::void_> {
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
    
      template<typename WorkSpaceType>
      static int Run(WorkSpaceType *data,
          const std::vector<std::string> &args);
  };
}}
#endif
