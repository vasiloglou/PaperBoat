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

#ifdef PAPERBOAT_MLPACK_LOGISTIC_TENSOR_FACTORIZATION_H_
#define PAPERBOAT_MLPACK_LOGISTIC_TENSOR_FACTORIZATION_H_
#include "boost/program_options.hpp"
#include "boost/mpl/void.hpp"
#include "boost/shared_ptr.hpp"
#include "fastlib/workspace/workspace.h"


namespace fl { namespace ml {
  template<typename ArgsType>
  class LogisticTensorFactorization {
    public:
      typedef typename ArgsType::Table_t Table_t;
      typedef typename Table_t::Point_t Point_t;
      typedef typename ArgsType::Objective_t Objective_t;
      typedef typename ArgsType::Regularizer_t Regularizer_t;
      
      struct BatchTrainer {
        public:
        
          void Init(index_t num_dimensions);
          double Evaluate(const fl::data::MonolithicPoint<double> &model);
          void Gradient(fl::data::MonolithicPoint<double> &model,
              fl::data::MonolithicPoint<double> *gradient);
          index_t num_dimensions();
          void set_references(Table_t *references);
          void set_a_regularization(double regularization);
          void set_r_regularization(double regularization);
          double ComputeReferenceNorm();
          void DeserializeModel(fl::data::MonolithicPoint<double> &data,
              Matrix_t *a_mat, Matrix_t *r_mat);
          void SerializeModel(Matrix_t &a_mat, Matrix_t b_mat, 
              fl::data::MonolithicPoint<double> *data);

        protected:
          Table_t *references_; 
          index_t num_of_dimensions_;
          Matrix_t a_mat_;
          Matrix_t r_mat_;
          double a_regularization_;
          double r_regularization_;

      };
 
      struct PointTrainer : public BatchTrainer{
        public:
          typedef typename Table_t::Point_t Point_t;
          std::unordered_map<index_t, double> Gradient(fl::data::MonolithicPoint<double> &model,
              const Point_t &point);
          double LocalError(fl::data::MonolithicPoint<double> &model,
              const Point_t &point);
          double Evaluate(fl::data::MonolithicPoint<double> &data);
        protected:
          std::vector<Matrix_t> model_;
          fl::data::MonolithicPoint<double> data_;
          FactorizationMachine machine_;
       };
    
      const std::vector<Matrix_t> &model() const ;

      const fl::data::MonolithicPoint<double> &data() const;
      
      template<typename WorkSpaceType>
      void Optimize(
         WorkSpaceType *ws,
         const std::vector<std::string> &references_names,
         const std::string &optimizer, 
         const std::vector<double> &regularization_parameter,
         int32 iterations,
         const std::vector<std::string> &args);

      template<typename WorkSpaceType>
      void ExportModel(const std::string &model_prefix,
                       int32 model_size,
                       WorkSpaceType *ws);

    private:
      int32 model_order_;
      std::vector<int32> ranks_;
      std::vector<Matrix_t> model_;
      fl::data::MonolithicPoint<double> data_;

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
