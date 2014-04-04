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

#ifndef PAPERBOAT_INCLUDE_MLPACK_HMM_H_
#define PAPERBOAT_INCLUDE_MLPACK_HMM_H_
#include "boost/program_options.hpp"
#include "boost/mpl/void.hpp"
#include "boost/shared_ptr.hpp"
#include "fastlib/workspace/workspace.h"
#include "discrete_distribution.h"
#include "gmm_distribution.h"
#include "kde_distribution.h"
#include "fastlib/table/default_sparse_double_table.h"
#include "fastlib/workspace/based_on_table_run.h"


namespace fl { namespace ml {
  template<typename HmmArgsType>
  class Hmm {
    public:
      typedef typename HmmArgsType::WorkSpaceType WorkSpace_t;
      typedef typename HmmArgsType::TransitionTableType TransitionTable_t;
      typedef typename HmmArgsType::DistributionType Distribution_t;
      typedef typename Distribution_t::Table_t Table_t;
      typedef typename Table_t::Point_t Point_t;
      typedef std::vector<std::pair<index_t, double> > StateSequence_t;

      void Init(index_t n_states,
               const std::vector<std::string> &distribution_args,
               std::vector<
               boost::shared_ptr<Table_t> 
               > &references);
      void Train(int32 iterations,
          std::vector<
            boost::shared_ptr<Table_t> 
          > &references);
      double Eval(Table_t &table);
      void Update(std::vector<
                 boost::shared_ptr<
                   Table_t
                 > 
               > &references, 
               std::vector<StateSequence_t> &state_sequences);
      StateSequence_t ComputeMostLikelyStateSequence(Table_t &table);

      boost::shared_ptr<TransitionTable_t> transition_matrix();
      void set_transition_matrix(
          boost::shared_ptr<TransitionTable_t> transition_matrix);
      std::map<index_t, double> &initial_probabilities();
      template<typename PointType>
      void set_initial_probabilities(PointType &initial_prob);
      std::vector<Distribution_t> &distributions();
      void set_distributions(std::vector<Distribution_t> &distributions);
      int32 n_states();
    private:
      struct LogTransform {
        LogTransform(double sum) {
          sum_=sum;
        }
        template<typename T>
        void operator()(index_t ind, T *element) {
          if (*element!=-std::numeric_limits<double>::max()) {
            *element=log(*element/sum_);
          }
        }
        private:
          double sum_;
      };

      boost::shared_ptr<TransitionTable_t> transition_matrix_;
      std::map<index_t, double> initial_probabilities_;
      std::vector<Distribution_t> distributions_;
      index_t n_states_;
  };

  template<>
  class Hmm<boost::mpl::void_> {
    public:
      template<typename WorkSpaceType1>
      struct Core {
        public:
          template<typename TableType>
          struct HmmArgsDiscrete1 {
            typedef WorkSpaceType1 WorkSpaceType;
            typedef typename WorkSpaceType1::DefaultSparseDoubleTable_t TransitionTableType;
            typedef DiscreteDistribution<TableType> DistributionType;
          };
          template<typename TableType>
          struct HmmArgsDiscrete2 {
            typedef WorkSpaceType1 WorkSpaceType;
            typedef typename WorkSpaceType1::DefaultTable_t TransitionTableType;
            typedef DiscreteDistribution<TableType> DistributionType;
      
          };
          template<typename TableType>
          struct HmmArgsKde1 {
            typedef WorkSpaceType1 WorkSpaceType;
            typedef typename WorkSpaceType1::DefaultSparseDoubleTable_t TransitionTableType;
            typedef KdeDistribution<TableType> DistributionType;
      
          };
          template<typename TableType>
          struct HmmArgsKde2 {
            typedef WorkSpaceType1 WorkSpaceType;
            typedef typename WorkSpaceType1::DefaultTable_t TransitionTableType;
            typedef KdeDistribution<TableType> DistributionType;
      
          };
          template<typename TableType>
          struct HmmArgsGmm1 {
            typedef WorkSpaceType1 WorkSpaceType;
            typedef typename WorkSpaceType1::DefaultSparseDoubleTable_t TransitionTableType;
            typedef GmmDistribution<TableType, true> DistributionType;
      
          };
          template<typename TableType>
          struct HmmArgsGmm2 {
            typedef WorkSpaceType1 WorkSpaceType;
            typedef typename WorkSpaceType1::DefaultTable_t TransitionTableType;
            typedef GmmDistribution<TableType, true> DistributionType;
          };
        
          Core(WorkSpaceType1 *ws, const std::vector<std::string> &args);
          template<typename FullTransitionTableType,
            typename SparseTransitionTableType,
            typename Engine1Type,
            typename Engine2Type,
            typename InitialProbType>
          void LoadHmmParams(
              FullTransitionTableType full_transition_matrix,
              SparseTransitionTableType ff_transition_matrix,
              InitialProbType &initial_probabilities_point,
              Engine1Type &engine1,
              Engine2Type &engine2,
              int32 n_states,
              const std::vector<std::string> &init_args,
              const std::vector<std::string> &exec_args);
          template<typename TableType>
          void operator()(TableType&);

          template<typename EngineType>
          void Export(
            boost::program_options::variables_map &vm,
            EngineType &engine);

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
