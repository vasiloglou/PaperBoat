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
#ifndef PAPER_BOAT_INCLUDE_MLPACK_NONPARAMETRIC_REGRESSION_H_
#define PAPER_BOAT_INCLUDE_MLPACK_NONPARAMETRIC_REGRESSION_H_
#include "fastlib/workspace/workspace_defs.h"
#include "fastlib/table/default/dense/labeled/kdtree/table.h"
#include "nonparametric_regression_stat.h"
#include "mlpack/kde/kde.h"

namespace fl {namespace ml {
  
  template<typename TableType1>
  class NonParametricRegression { 
    public:
      typedef TableType1 Table_t;
      class Trainer {
        public:
          typedef fl::data::MonolithicPoint<double> WPoint_t;
          typedef TableType1 Table_t;
          typedef typename Table_t::Point_t Point_t;
        
          Trainer();
          void InitBandwidthsFromData();
          double LbfgsTrain();
          double StochasticTrain();
          void Project(WPoint_t *point) {}
          void Gradient(WPoint_t &x, WPoint_t *df_dx);
          double Evaluate(const WPoint_t &x);
          index_t num_dimensions();
 
          WPoint_t get_bandwidths();
          double get_error();
          void set_references(Table_t *references);
          void set_queries(Table_t *queries);
          void set_lbfgs_rank(int lbfgs_rank);
          void set_num_of_line_searches(int num_of_line_searches);
          void set_iterations(int iterations);
          void set_iteration_chunks(int iteration_chunks);
          void set_bandwidths(fl::data::MonolithicPoint<double> &bandwidths);
          void set_eta0(double eta);

        private:
          WPoint_t bandwidths_;
          Table_t *references_;
          Table_t *queries_;
          int lbfgs_rank_;
          double eta0_;
          int num_of_line_searches_;
          int iterations_;
          int iteration_chunks_;
          double error_;

          void ComputeGradient(Table_t &refereces, Table_t &queries,
              WPoint_t &x, WPoint_t *df_dx);
      };
  
      class Predictor {
        public:
          typedef TableType1 Table_t;

          template<typename TemplateArgs>
          struct NprStructArgs {
            typedef NprDelta<double> Delta_t;
            typedef NprGlobal<TemplateArgs> Global_t;
            typedef NprPostponed<double> Postponed_t;
            typedef NprResult< std::vector<double> > Result_t;
            typedef NprStatistic<double> Statistic_t;
            typedef KdeSummary<double> Summary_t;
          };

          struct NprArgs {
            typedef fl::math::GaussianKernel<double> KernelType;
            typedef Table_t TableType;
            typedef fl::math::LMetric<2> MetricType;
            typedef fl::table::TimeFilter PointFilterType;
            typedef double CalcPrecision_t;
            struct TemplateArgs {
              typedef fl::math::GaussianKernel<double> KernelType;
              typedef NprStatistic<double> Statistic_t;
              typedef Table_t TableType;
              typedef double CalcPrecision_t;
            };
            typedef NprStructArgs<TemplateArgs> ComputationType;
          };

          typedef fl::ml::Kde<NprArgs>  Npr_t;
          Predictor();
          void set_references(Table_t *references);
          void set_relative_error(double relative_error);
          void set_probability(double probability);
          void Predict(Table_t *queries,
                       NprResult<std::vector<double> > *result);
        
        private:
          Table_t *references_;
          double relative_error_;
          double probability_;
      };
  };

  template<>
  class NonParametricRegression<boost::mpl::void_> {
    public:
      template<typename TableType>
      struct Core {
        static void SplitTable(boost::shared_ptr<TableType> table,
                    index_t n_splits, 
                    index_t new_table_max_size,
                    std::vector<boost::shared_ptr<TableType> > *tables);

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

