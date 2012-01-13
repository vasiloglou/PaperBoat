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

#ifndef FL_LITE_MLPACK_SVM_SVM_H_
#define FL_LITE_MLPACK_SVM_SVM_H_
#include "fastlib/metric_kernel/lmetric.h"
#include "fastlib/metric_kernel/gaussian_dotprod.h"
#include "mlpack/allkn/allkn.h"

namespace fl { namespace ml{

  template<typename TableType>
  class Svm {
    public:
      typedef TableType Table_t;
      typedef typename Table_t::CalcPrecision_t CalcPrecision_t;
      typedef typename Table_t::Point_t Point_t;
      template<typename KernelType>
      class Trainer;
      template<typename KernelType>
      class Predictor;
        
      template<typename GeometryType>
      class Trainer<fl::math::GaussianDotProduct<
          typename TableType::CalcPrecision_t, GeometryType>  > { 
        public:
          typedef  fl::math::GaussianDotProduct<
              typename TableType::CalcPrecision_t, GeometryType> Kernel_t;
          struct PointInfo {
            double a; // the alphas in optimization
            signed char y;    // the label of the point as defined in the SMO
            int u;    // the svm prediction
            std::vector<index_t> neighbors;
            std::vector<double> kernel_distances;
            std::vector<double> distances;
          };
          Trainer();
          void set_reference_table(Table_t *reference_table);
          void set_labels(std::vector<index_t> *labels);
          void set_regularization(double regularization);
          void set_iterations(index_t iterations);
          void set_accuracy(double accuracy);
          void set_kernel(Kernel_t &kernel);
          void set_bias(bool bias);
          void set_bandwidth_overload_factor(double bandwidth_overload_factor);
          void Train(std::map<index_t, double> *support_vectors_full,
                     boost::shared_ptr<Table_t> *reduced_table);
          void Sample(Table_t &table_in, double bandwidth_overload_factor, Table_t *table_out, index_t *suggested_knn);
          void InitSmo(Table_t &references,
                       const index_t knn,
                       std::vector<PointInfo> *point_info);
          void RunSmo(Table_t &references,
                      const double accuracy,
                      const double regularization,
                      const index_t iterations, 
                      std::vector<PointInfo> *point_info,
                      std::map<index_t, double>  *nonzero_alphas);
          double EvaluateSvm(const Point_t &point,
                           Table_t &references,
                           const std::map<index_t, double> &nnz_support_vectors);
          double EvaluateSvm(index_t p1,
                           Table_t &references,
                           const std::map<index_t, double> &nnz_support_vectors);
          void PickPoint(index_t i1, 
                         std::vector<PointInfo> &point_info, 
                         Table_t &references,
                         std::map<index_t, double> &nnz_support_vectors,
                         index_t *i2, double *step);
          void ComputeTrainingError(Table_t &references, 
                                    Table_t &queries,
                                    std::map<index_t, double> &nnz_support_vectors,
                                    double *error); 
        protected:
          Table_t *references_;
          std::vector<index_t> *labels_;
          double regularization_;
          index_t iterations_;
          double accuracy_;
          Kernel_t kernel_;
          bool bias_;
          double bandwidth_overload_factor_;
          std::map<index_t, double> kernel_cache_;

          void SampleRecursion(
              typename Table_t::Tree_t *node,
              const double diameter,
              Table_t &table_in, 
              Table_t *table_out,
              index_t *total_knns,
              index_t *total_nodes);
          index_t get_cache_id(index_t i, index_t j, index_t n);
      };
     
      template<typename GeometryType>
      class Predictor<fl::math::GaussianDotProduct<
          typename TableType::CalcPrecision_t, GeometryType>  > {
        public:
          typedef TableType Table_t;
          typedef typename Table_t::Point_t Point_t;
          typedef  fl::math::GaussianDotProduct<
              typename TableType::CalcPrecision_t, GeometryType> Kernel_t;
          void set_query_table(Table_t *query_table);
          void set_kernel(Kernel_t &kernel);
          void set_support_vectors(Table_t *support_vectors);
          void set_alphas(std::vector<double> *alphas);
          void Predict(std::vector<double> *margins, double *prediction_accuracy);      
          void Predict(const Point_t &point, double *margin);

        protected:
          Table_t *query_table_;
          std::vector<double> *labels_;
          Kernel_t kernel_;
          Table_t *support_vectors_;
          std::vector<double> *alphas_;
      };
  };

  template<>
  class Svm<boost::mpl::void_> {
    public:
      template<typename TableType>
      struct Core {
        template<typename WorkSpaceType>
        static int Main(WorkSpaceType *data,
                        boost::program_options::variables_map &vm);
      };
      template <typename WorkSpaceType, typename BranchType>
      static int Main(WorkSpaceType *ws, const std::vector<std::string> &args);
    
    template<typename DataAccessType>
    static void Run(DataAccessType *data,
        const std::vector<std::string> &args);

  };

}}
#endif
