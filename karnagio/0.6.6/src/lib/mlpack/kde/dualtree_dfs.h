/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
Inc, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE Ismion Inc "AS IS" AND ANY
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

#ifndef FL_LITE_MLPACK_KDE_DUALTREE_DFS_H
#define FL_LITE_MLPACK_KDE_DUALTREE_DFS_H

#include "mlpack/kde/dualtree_trace.h"

namespace fl {
namespace ml {

template<typename ProblemType>
class DualtreeDfs {

  public:

    typedef typename ProblemType::Table_t Table_t;
    typedef typename Table_t::Tree_t Tree_t;
    typedef typename ProblemType::Result_t Result_t;
    typedef typename ProblemType::PointFilter_t PointFilter_t;

  public:
    template<typename MetricType>
    class iterator {
      private:
        class IteratorArgType {

          private:

            Tree_t *qnode_;

            Tree_t *rnode_;

            GenRange<double> squared_distance_range_;

          public:

            IteratorArgType();

            IteratorArgType(const IteratorArgType &arg_in);

            IteratorArgType(
              const MetricType &metric_in,
              Table_t *query_table_in, Tree_t *qnode_in,
              Table_t *reference_table_in,
              Tree_t *rnode_in);

            IteratorArgType(
              const MetricType &metric_in,
              Table_t *query_table_in, Tree_t *qnode_in,
              Table_t *reference_table_in,
              Tree_t *rnode_in,
              const GenRange<double> &squared_distance_range_in);

            Tree_t *qnode();

            Tree_t *qnode() const;

            Tree_t *rnode();

            Tree_t *rnode() const;

            const GenRange<double> &squared_distance_range() const;
        };

      private:

        Table_t *query_table_;

        Table_t *reference_table_;

        DualtreeDfs<ProblemType> *engine_;

        const MetricType &metric_;

        Result_t *query_results_;

        fl::ml::DualtreeTrace<IteratorArgType> trace_;

      public:

        iterator(const MetricType &metric_in,
                 DualtreeDfs<ProblemType> &engine_in,
                 Result_t *query_results_in);

        void operator++();

        Result_t &operator*();

        const Result_t &operator*() const;

        void Finalize();
    };

  private:

    ProblemType *problem_;

    Table_t *query_table_;

    Table_t *reference_table_;

    boost::shared_ptr<std::vector<typename ProblemType::Statistic_t> > 
      reference_statistics_;

    boost::shared_ptr<std::vector<typename ProblemType::Statistic_t> > 
      query_statistics_;

    PointFilter_t filter_;
  private:

    void ResetStatisticRecursion_(Tree_t *node, 
        Table_t * table,
        std::vector<typename ProblemType::Statistic_t> *statistics);

    void PreProcessReferenceTree_(Tree_t *rnode);

    void PreProcess_(Tree_t *qnode);

    template<typename MetricType>
    void DualtreeBase_(MetricType &metric,
                       Tree_t *qnode,
                       Tree_t *rnode,
                       Result_t *result);

    bool CanSummarize_(Tree_t *qnode,
                       Tree_t *rnode,
                       const typename ProblemType::Delta_t &delta,
                       typename ProblemType::Result_t *query_results);

    template<typename MetricType>
    bool CanProbabilisticSummarize_(
      const MetricType &metric,
      Tree_t *qnode,
      Tree_t *rnode,
      double failure_probability,
      typename ProblemType::Delta_t &delta,
      typename ProblemType::Result_t *query_results);

    void Summarize_(Tree_t *qnode,
                    const typename ProblemType::Delta_t &delta,
                    typename ProblemType::Result_t *query_results);

    template<typename GlobalType>
    void ProbabilisticSummarize_(
      const GlobalType &global,
      typename ProblemType::Table_t::Tree_t *qnode,
      double failure_probability,
      const typename ProblemType::Delta_t &delta,
      typename ProblemType::Result_t *query_results);

    template<typename MetricType>
    void Heuristic_(
      const MetricType &metric,
      Tree_t *node,
      Table_t *node_table,
      Tree_t *first_candidate,
      Tree_t *second_candidate,
      Table_t *candidate_table,
      Tree_t **first_partner,
      GenRange<double> &first_squared_distance_range,
      Tree_t **second_partner,
      GenRange<double> &second_squared_distance_range);

    template<typename MetricType>
    bool DualtreeCanonical_(
      MetricType &metric,
      Tree_t *qnode,
      Tree_t *rnode,
      double failure_probability,
      const GenRange<double> &squared_distance_range,
      Result_t *query_results);

    template<typename MetricType>
    void PostProcess_(
      const MetricType &metric, Tree_t *qnode, Result_t *query_results);

  public:
    DualtreeDfs() {
      problem_ = NULL;
      query_table_ = NULL;
      reference_table_ = NULL;
    }

    void set_filter(const PointFilter_t &filter) {
      filter_=filter;
    }

    ProblemType *problem();

    Table_t *query_table();

    Table_t *reference_table();

    template<typename MetricType>
    typename DualtreeDfs<ProblemType>::template iterator<MetricType> get_iterator(
      const MetricType &metric_in, Result_t *query_results_in);

    void ResetStatistic();

    void Init(ProblemType &problem_in);

    template<typename MetricType>
    void Compute(const MetricType &metric,
                 typename ProblemType::Result_t *query_results);
};
};
};

#endif
