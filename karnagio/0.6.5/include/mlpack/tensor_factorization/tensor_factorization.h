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

#ifndef PAPERBOAT_INCLUDE_MLPACK_TENSOR_FACTORIZATION_TENSOR_FACTORIZATION_H_
#define PAPERBOAT_INCLUDE_MLPACK_TENSOR_FACTORIZATION_TENSOR_FACTORIZATION_H_
#include "boost/program_options.hpp"
#include "boost/mpl/void.hpp"
#include "boost/shared_ptr.hpp"
#include "fastlib/workspace/workspace.h"


namespace fl { namespace ml {
  template<typename WorkSpaceType>
  class TensorFactorization {
    public:
      class Cpwopt {
        public:
          template<typename TableType>
          static void ComputeSGD(
              std::vector<boost::shared_ptr<TableType> > &tensor, 
              int32 rank,
              double a_regularization,
              double b_regularization,
              double c_regularization,
              double step0,
              int32 epochs,
              int32 num_iterations,
              boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *a_table,
              boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *b_table,
              boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *c_table
          );
          template<typename TableType>
          static void ComputeCpwopt(
            std::vector<boost::shared_ptr<TableType> > &tensor,
            int32 parafac_rank,
            double a_regularization,
            double b_regularization,
            double c_regularization,
            int32 num_basis,
            int32 max_num_line_searches_in,
            int32 num_iterations,
            boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *a_table,
            boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *b_table,
            boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *c_table);
          template<typename TableType, typename FunType, typename EngineType>
          static bool Optimizer(
              std::vector<boost::shared_ptr<TableType> > &tensor, 
              FunType &fun,
              EngineType &engine,
              int32 parafac_rank,
              double a_regularization,
              double b_regularization,
              double c_regularization,
              int32 num_iterations,
              int32 max_num_line_searched_in,
              int32 num_basis,
              fl::data::MonolithicPoint<double> *iterate);
          template<typename TableType>
          static void AssignMats(fl::data::MonolithicPoint<double> &variable,
                  std::vector<boost::shared_ptr<TableType> > &tensor,
                  int32 k_rank,
                  fl::dense::Matrix<double> *a_mat,
                  fl::dense::Matrix<double> *b_mat,
                  fl::dense::Matrix<double> *c_mat);
    
          template<typename TableType,
                   bool A_REGULARIZATION,
                   bool B_REGULARIZATION,
                   bool C_REGULARIZATION,
                   bool L2_REGULARIZATION
                   > 
          class LbfgsFun {
            public:
              void Init(std::vector<boost::shared_ptr<TableType> > *tensor,
                   int32 factorization_rank,
                   double a_regularization,
                   double b_regularization,
                   double c_regularization);
              void Gradient(const fl::data::MonolithicPoint<double> &variable,
                            fl::data::MonolithicPoint<double> *gradient);
              double Evaluate(const fl::data::MonolithicPoint<double> &variable);
              const index_t num_dimensions() const; 
              void AssignMats(fl::data::MonolithicPoint<double> &variable,
                  fl::dense::Matrix<double> *a_mat,
                  fl::dense::Matrix<double> *b_mat,
                  fl::dense::Matrix<double> *c_mat);
              double tensor_sq_norm() const ;
              double eval_result() const;
              double factorization_error() const;
    
            private:
              std::vector<boost::shared_ptr<TableType> > *tensor_;
              int32 rank_;          
              index_t num_dimensions_;
              double a_regularization_;
              double b_regularization_;
              double c_regularization_;
              double tensor_sq_norm_;
              double eval_result_;
              double factorization_error_;// error without the regularization
          };
      };
      class Dedicom {
        public:
          template<typename TableType>
          static void ComputeSGD(
              std::vector<boost::shared_ptr<TableType> > &tensor, 
              int32 rank,
              double a_regularization,
              double b_regularization,
              double c_regularization,
              double step0,
              int32 epochs,
              int32 num_iterations,
              boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *a_table,
              boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *b_table,
              boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *c_table
          );
          template<typename TableType>
          static void ComputeCpwopt(
            std::vector<boost::shared_ptr<TableType> > &tensor,
            int32 parafac_rank,
            double a_regularization,
            double b_regularization,
            double c_regularization,
            int32 num_basis,
            int32 max_num_line_searches_in,
            int32 num_iterations,
            boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *a_table,
            boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *b_table,
            boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *c_table);
          template<typename TableType, typename FunType, typename EngineType>
          static bool Optimizer(
              std::vector<boost::shared_ptr<TableType> > &tensor, 
              FunType &fun,
              EngineType &engine,
              int32 parafac_rank,
              double a_regularization,
              double b_regularization,
              double c_regularization,
              int32 num_iterations,
              int32 max_num_line_searched_in,
              int32 num_basis,
              fl::data::MonolithicPoint<double> *iterate);
          template<typename TableType>
          static void AssignMats(fl::data::MonolithicPoint<double> &variable,
                  std::vector<boost::shared_ptr<TableType> > &tensor,
                  int32 k_rank,
                  fl::dense::Matrix<double> *a_mat,
                  fl::dense::Matrix<double> *b_mat,
                  fl::dense::Matrix<double> *c_mat);
    
          template<typename TableType,
                   bool A_REGULARIZATION,
                   bool B_REGULARIZATION,
                   bool C_REGULARIZATION,
                   bool L2_REGULARIZATION
                   > 
          class LbfgsFun {
            public:
              void Init(std::vector<boost::shared_ptr<TableType> > *tensor,
                   int32 factorization_rank,
                   double a_regularization,
                   double b_regularization,
                   double c_regularization);
              void Gradient(const fl::data::MonolithicPoint<double> &variable,
                            fl::data::MonolithicPoint<double> *gradient);
              double Evaluate(const fl::data::MonolithicPoint<double> &variable);
              const index_t num_dimensions() const; 
              void AssignMats(fl::data::MonolithicPoint<double> &variable,
                  fl::dense::Matrix<double> *a_mat,
                  fl::dense::Matrix<double> *b_mat,
                  fl::dense::Matrix<double> *c_mat);
              double tensor_sq_norm() const ;
              double eval_result() const;
              double factorization_error() const;
    
            private:
              std::vector<boost::shared_ptr<TableType> > *tensor_;
              int32 rank_;          
              index_t num_dimensions_;
              double a_regularization_;
              double b_regularization_;
              double c_regularization_;
              double tensor_sq_norm_;
              double eval_result_;
              double factorization_error_;// error without the regularization
          };
      };
  };

  template<>
  class TensorFactorization<boost::mpl::void_> {
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
