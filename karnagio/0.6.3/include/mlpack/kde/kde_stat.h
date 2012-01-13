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
#ifndef FL_LITE_MLPACK_KDE_KDE_STAT_H
#define FL_LITE_MLPACK_KDE_KDE_STAT_H

#include "boost/math/distributions/normal.hpp"
#include "fastlib/base/base.h"
#include "fastlib/dense/matrix.h"
#include "fastlib/tree/abstract_statistic.h"
#include "mlpack/kde/mean_variance_pair.h"


namespace fl {
namespace ml {

template<typename CalcPrecision_t>
class KdePostponed {

  public:

    CalcPrecision_t densities_l_;

    CalcPrecision_t densities_u_;

    CalcPrecision_t pruned_;

    KdePostponed() {
    }

    ~KdePostponed() {
    }

    void Init() {
      SetZero();
    }

    void Init(index_t rnode_count) {
      densities_l_ = densities_u_ = 0;
      pruned_ = (CalcPrecision_t) rnode_count;
    }

    template<typename KdeDelta, typename ResultType>
    void ApplyDelta(const KdeDelta &delta_in,
                    ResultType *query_results) {
      densities_l_ = ((CalcPrecision_t) densities_l_ +
                      (CalcPrecision_t) delta_in.densities_l_);
      densities_u_ = ((CalcPrecision_t) densities_u_ +
                      (CalcPrecision_t) delta_in.densities_u_);
      pruned_ = ((CalcPrecision_t) pruned_ +
                 (CalcPrecision_t) delta_in.pruned_);
    }

    void ApplyPostponed(const KdePostponed &other_postponed) {

      densities_l_ = ((CalcPrecision_t) densities_l_ +
                      (CalcPrecision_t) other_postponed.densities_l_);
      densities_u_ = ((CalcPrecision_t) densities_u_ +
                      (CalcPrecision_t) other_postponed.densities_u_);
      pruned_ = ((CalcPrecision_t) pruned_ +
                 (CalcPrecision_t) other_postponed.pruned_);
    }

    template<typename GlobalType, typename MetricType, typename PointType>
    void ApplyContribution(const GlobalType &global,
                           const MetricType &metric,
                           const PointType &query_point,
                           const PointType &reference_point) {
      double distsq = metric.DistanceSq(query_point, reference_point);
      double density_incoming = global.kernel().EvalUnnormOnSq(distsq);
      densities_l_ = ((CalcPrecision_t) densities_l_ + density_incoming);
      densities_u_ = ((CalcPrecision_t) densities_u_ + density_incoming);
    }

    void SetZero() {
      densities_l_ = 0;
      densities_u_ = 0;
      pruned_ = 0;
    }
};

template<typename TemplateMap>
class KdeGlobal {

  public:

    typedef typename TemplateMap::TableType Table_t;

    typedef typename TemplateMap::KernelType Kernel_t;

    typedef typename TemplateMap::TableType::CalcPrecision_t CalcPrecision_t;

    typedef typename TemplateMap::Statistic_t Statistic_t;

  protected:

    double relative_error_;

    double probability_;

    Kernel_t kernel_;

    CalcPrecision_t mult_const_;

    Table_t *query_table_;

    Table_t *reference_table_;

    boost::math::normal normal_dist_;

    std::vector< fl::ml::MeanVariancePair > mean_variance_pair_;

    std::vector<Statistic_t> *reference_statistics_;

  public:

    std::vector< fl::ml::MeanVariancePair > *mean_variance_pair() {
      return &mean_variance_pair_;
    }

    double compute_quantile(double tail_mass) const {
      double mass = 1 - 0.5 * tail_mass;
      if (mass > 0.999) {
        return 3;
      }
      else {
        return boost::math::quantile(normal_dist_, mass);
      }
    }

    Table_t *query_table() {
      return query_table_;
    }

    const Table_t *query_table() const {
      return query_table_;
    }

    Table_t *reference_table() {
      return reference_table_;
    }

    const Table_t *reference_table() const {
      return reference_table_;
    }

    double relative_error() const {
      return relative_error_;
    }

    double probability() const {
      return probability_;
    }

    void set_bandwidth(double bandwidth_in) {
      kernel_.Init(bandwidth_in);
    }

    const Kernel_t &kernel() const {
      return kernel_;
    }

    std::vector<Statistic_t>* &reference_statistic() {
      return reference_statistics_;
    }

    const std::vector<Statistic_t>* reference_statistic() const {
      return reference_statistics_;
    }

    void Init(Table_t *reference_table_in, Table_t *query_table_in,
              std::vector<Statistic_t> *references_stats, 
              double bandwidth_in, const bool is_monochromatic,
              double relative_error_in, double probability_in) {
      reference_statistics_=references_stats;
      index_t effective_num_points =
        (is_monochromatic) ?
        (reference_table_in->n_entries() - 1) :
        reference_table_in->n_entries();
      kernel_.Init(bandwidth_in);
      mult_const_ = 1.0 /
                    (kernel_.CalcNormConstant(
                       reference_table_in->n_attributes()) *
                     ((CalcPrecision_t) effective_num_points));
      relative_error_ = relative_error_in;
      probability_ = probability_in;
      query_table_ = query_table_in;
      reference_table_ = reference_table_in;

      // Initialize the temporary vector for storing the Monte Carlo
      // results.
      mean_variance_pair_.resize(query_table_->n_entries());
    }

    CalcPrecision_t get_mult_const() const {
      return mult_const_;
    }
};

template<typename ContainerType>
class KdeResult {
  public:
    typedef typename ContainerType::value_type CalcPrecision_t;
    ContainerType densities_l_;
    ContainerType densities_;
    ContainerType densities_u_;
    ContainerType pruned_;

    KdeResult() {
    }

    ~KdeResult() {
    }

    template<typename MetricType, typename GlobalType>
    void PostProcess(const MetricType &metric,
                     index_t q_index,
                     const GlobalType &global,
                     const bool is_monochromatic) {
      if (is_monochromatic) {
        densities_l_[q_index] = (CalcPrecision_t)(densities_l_[q_index] -
                                1.0);
        densities_u_[q_index] = (CalcPrecision_t)(densities_u_[q_index] -
                                1.0);
      }

      densities_[q_index] = 0.5 * (densities_l_[q_index] +
                                   densities_u_[q_index]);
      densities_l_[q_index] *= global.get_mult_const();
      densities_[q_index] *= global.get_mult_const();
      densities_u_[q_index] *= global.get_mult_const();
    }

    void PrintDebug(const std::string &file_name) {
      FILE *file_output = fopen(file_name.c_str(), "w+");
      for (unsigned int i = 0; i < densities_.size(); i++) {
        fprintf(file_output, "%g %g %g %g\n", densities_l_[i],
                densities_[i], densities_u_[i], pruned_[i]);
      }
      fclose(file_output);
    }

    template<int mode, typename TableType>
    void GetDensities(TableType* table) {
      if (mode == 0) {
        table->Init("", std::vector<index_t>(1, 1), std::vector<index_t>(),
                    densities_.size());
      }
      for (int i = 0; i < densities_.size(); i++) {
        typename TableType::Point_t point;
        table->get(i, &point);
        point.set(0, densities_[i]);
      }
    }

    void Init(int num_points) {
      densities_l_.resize(num_points);
      densities_.resize(num_points);
      densities_u_.resize(num_points);
      pruned_.resize(num_points);

      SetZero();
    }

    void SetZero() {
      for (int i = 0; i < static_cast<int>(densities_l_.size()); i++) {
        densities_l_[i] = 0;
        densities_[i] = 0;
        densities_u_[i] = 0;
        pruned_[i] = 0;
      }
    }

    template<typename GlobalType, typename TreeType, typename DeltaType>
    void ApplyProbabilisticDelta(const GlobalType &global,
                                 TreeType *qnode,
                                 double failure_probability,
                                 const DeltaType &delta_in) {

      // Get the iterator for the query node.
      typename GlobalType::Table_t::TreeIterator qnode_it =
        global.query_table()->get_node_iterator(qnode);
      typename GlobalType::Table_t::Dataset_t::Point_t qpoint;
      index_t qpoint_index;

      // Look up the number of standard deviations.
      double num_standard_deviations = global.compute_quantile(
                                         failure_probability);

      do {
        // Get each query point.
        qnode_it.Next(&qpoint, &qpoint_index);
        GenRange<double> contribution;
        (*delta_in.mean_variance_pair_)[qpoint_index].scaled_interval(
          delta_in.pruned_, num_standard_deviations, &contribution);
        densities_l_[qpoint_index] += contribution.lo;
        densities_u_[qpoint_index] += contribution.hi;
        pruned_[qpoint_index] += delta_in.pruned_;
      }
      while (qnode_it.HasNext());
    }

    void ApplyPostponed(int q_index,
                        const KdePostponed<CalcPrecision_t> &postponed_in) {

      densities_l_[q_index] = ((CalcPrecision_t) densities_l_[q_index] +
                               (CalcPrecision_t) postponed_in.densities_l_);
      densities_u_[q_index] = ((CalcPrecision_t) densities_u_[q_index] +
                               (CalcPrecision_t) postponed_in.densities_u_);
      pruned_[q_index] = ((CalcPrecision_t) pruned_[q_index] +
                          (CalcPrecision_t) postponed_in.pruned_);
    }
};

template<typename CalcPrecision_t>
class KdeDelta {

  public:

    CalcPrecision_t densities_l_;

    CalcPrecision_t densities_u_;

    CalcPrecision_t pruned_;

    std::vector< fl::ml::MeanVariancePair > *mean_variance_pair_;

    KdeDelta() {
      SetZero();
    }

    ~KdeDelta() {
    }

    void SetZero() {
      densities_l_ = densities_u_ = pruned_ = 0;
      mean_variance_pair_ = NULL;
    }

    template<typename MetricType, typename GlobalType, typename TreeType>
    void DeterministicCompute(
      const MetricType &metric,
      const GlobalType &global, TreeType *qnode, TreeType *rnode,
      const GenRange<CalcPrecision_t> &squared_distance_range) {

      index_t rnode_count = global.reference_table()->get_node_count(rnode);
      densities_l_ = rnode_count *
                     global.kernel().EvalUnnormOnSq(squared_distance_range.hi);
      densities_u_ = rnode_count *
                     global.kernel().EvalUnnormOnSq(squared_distance_range.lo);
      pruned_ = (CalcPrecision_t) rnode_count;
    }
};

template<typename CalcPrecision_t>
class KdeSummary {

  public:

    CalcPrecision_t densities_l_;

    CalcPrecision_t densities_u_;

    CalcPrecision_t pruned_l_;

    KdeSummary() {
      SetZero();
    }

    ~KdeSummary() {
    }

    KdeSummary(const KdeSummary &summary_in) {
      densities_l_ = summary_in.densities_l_;
      densities_u_ = summary_in.densities_u_;
      pruned_l_ = summary_in.pruned_l_;
    }

    template < typename MetricType, typename GlobalType, typename DeltaType,
    typename TreeType, typename ResultType >
    bool CanProbabilisticSummarize(
      const MetricType &metric, const GlobalType &global,
      DeltaType &delta, TreeType *qnode, TreeType *rnode,
      double failure_probability, ResultType *query_results) const {

      const int speedup_factor = 10;
      int num_samples = global.reference_table()->get_node_count(rnode) /
                        speedup_factor;

      if (num_samples > global.reference_table()->get_node_count(rnode)) {
        return false;
      }

      // Get the iterator for the query node.
      typename GlobalType::Table_t::TreeIterator qnode_it =
        global.query_table()->get_node_iterator(qnode);
      typename GlobalType::Table_t::Dataset_t::Point_t qpoint;
      index_t qpoint_index;

      // Get the iterator for the reference node.
      typename GlobalType::Table_t::TreeIterator rnode_it =
        global.reference_table()->get_node_iterator(rnode);
      typename GlobalType::Table_t::Dataset_t::Point_t rpoint;
      index_t rpoint_index;

      // Interval for the pivot query point.
      double num_standard_deviations = global.compute_quantile(
                                         failure_probability);
      delta.mean_variance_pair_ = ((GlobalType &) global).mean_variance_pair();

      // The flag saying whether the pruning is a success.
      bool prunable = true;

      // The min kernel value determined by the bounding box.
      double min_kernel_value = delta.densities_l_ /
                                ((double) global.reference_table()->
                                 get_node_count(rnode));

      index_t prev_qpoint_index = -1;
      double bandwidth = sqrt(global.kernel().bandwidth_sq());
      double movement_threshold = 0.05 * bandwidth;
      int movement_count = 0;
      do {

        // Get each query point.
        qnode_it.Next(&qpoint, &qpoint_index);
        bool skip = false;
        if (prev_qpoint_index >= 0) {
          typename GlobalType::Table_t::Dataset_t::Point_t prev_qpoint;
          global.query_table()->get(prev_qpoint_index, &prev_qpoint);
          double dist = sqrt(metric.DistanceSq(qpoint, prev_qpoint));
          if (dist <= movement_threshold && movement_count < 5) {
            (*delta.mean_variance_pair_)[qpoint_index].Copy(
              (*delta.mean_variance_pair_)[prev_qpoint_index]);
            skip = true;
            movement_count++;
          }
          else {
            movement_count = 0;
          }
        }

        // Clear the sample mean variance pair for the current query.
        if (skip == false) {
          (*delta.mean_variance_pair_)[qpoint_index].SetZero();

          for (int i = 0; i < num_samples; i++) {

            // Pick a random reference point and compute the kernel
            // difference.
            rnode_it.RandomPick(&rpoint, &rpoint_index);
            double squared_dist = metric.DistanceSq(qpoint, rpoint);
            //global.set_bw_for_point(rpoint_index);
            double kernel_value =
              global.kernel().EvalUnnormOnSq(squared_dist);
            double new_sample = kernel_value - min_kernel_value;

            // Accumulate the sample.
            (*delta.mean_variance_pair_)[qpoint_index].push_back(new_sample);
          }
        }

        // Add the correction.
        GenRange<double> correction;
        (*delta.mean_variance_pair_)[qpoint_index].scaled_interval(
          delta.pruned_, num_standard_deviations, &correction);
        correction.lo = std::max(correction.lo, 0.0);
        correction += delta.densities_l_;

        // Take the middle estimate, though technically it is not correct.
        double modified_densities_l =
          query_results->densities_l_[qpoint_index] + correction.lo;
        double left_hand_side = correction.width() * 0.5;
        double right_hand_side =
          global.reference_table()->get_node_count(rnode) *
          global.relative_error() * modified_densities_l /
          static_cast<double>(global.reference_table()->n_entries());

        prunable = (left_hand_side <= right_hand_side);

        prev_qpoint_index = qpoint_index;
      }
      while (qnode_it.HasNext() && prunable);
      return prunable;
    }

    template < typename GlobalType, typename DeltaType, typename TreeType,
    typename ResultType >
    bool CanSummarize(
      const GlobalType &global, const DeltaType &delta,
      TreeType *qnode, TreeType *rnode, ResultType *query_results) const {

      double left_hand_side =
        0.5 * (delta.densities_u_ - delta.densities_l_);
      double right_hand_side =
        global.reference_table()->get_node_count(rnode) *
        global.relative_error() * densities_l_ /
        static_cast<double>(global.reference_table()->n_entries());

      return left_hand_side <= right_hand_side;
    }

    void SetZero() {
      densities_l_ = densities_u_ = pruned_l_ = 0;
    }

    void Init() {
      SetZero();
    }

    void StartReaccumulate() {
      densities_l_ = std::numeric_limits<CalcPrecision_t>::max();
      densities_u_ = 0;
      pruned_l_ = densities_l_;
    }

    template<typename ResultType>
    void Accumulate(const ResultType &results, index_t q_index) {
      densities_l_ = std::min(densities_l_, results.densities_l_[q_index]);
      densities_u_ = std::max(densities_u_, results.densities_u_[q_index]);
      pruned_l_ = std::min(pruned_l_, results.pruned_[q_index]);
    }

    void Accumulate(const KdeSummary<CalcPrecision_t> &summary_in) {
      densities_l_ = std::min(densities_l_, summary_in.densities_l_);
      densities_u_ = std::max(densities_u_, summary_in.densities_u_);
      pruned_l_ = std::min(pruned_l_, summary_in.pruned_l_);
    }

    void Accumulate(const KdeSummary<CalcPrecision_t> &summary_in,
                    const KdePostponed<CalcPrecision_t> &postponed_in) {

      densities_l_ = std::min(densities_l_, summary_in.densities_l_ +
                              postponed_in.densities_l_);
      densities_u_ = std::max(densities_u_, summary_in.densities_u_ +
                              postponed_in.densities_u_);
      pruned_l_ = std::min(pruned_l_, summary_in.pruned_l_ +
                           postponed_in.pruned_);
    }

    void ApplyDelta(const KdeDelta<CalcPrecision_t> &delta_in) {
      densities_l_ = ((CalcPrecision_t) densities_l_ +
                      (CalcPrecision_t) delta_in.densities_l_);
      densities_u_ = ((CalcPrecision_t) densities_u_ +
                      (CalcPrecision_t) delta_in.densities_u_);
    }

    void ApplyPostponed(const KdePostponed<CalcPrecision_t>
                        &postponed_in) {

      densities_l_ = ((CalcPrecision_t) densities_l_ +
                      (CalcPrecision_t) postponed_in.densities_l_);
      densities_u_ = ((CalcPrecision_t) densities_u_ +
                      (CalcPrecision_t) postponed_in.densities_u_);
      pruned_l_ = ((CalcPrecision_t) pruned_l_ +
                   (CalcPrecision_t) postponed_in.pruned_);
    }
};

template<typename CalcPrecision_t>
class KdeStatistic : public fl::tree::AbstractStatistic {

  public:

    fl::ml::KdePostponed<CalcPrecision_t> postponed;

    fl::ml::KdeSummary<CalcPrecision_t> summary;

    KdeStatistic() {
    }

    ~KdeStatistic() {
    }

    void SetZero() {
      postponed.SetZero();
      summary.SetZero();
    }

    /**
     * Initializes by taking statistics on raw data.
     */
    template<typename TreeIterator>
    void Init(TreeIterator &it) {
      SetZero();
    }

    /**
     * Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    template<typename TreeIterator>
    void Init(TreeIterator &it,
              const KdeStatistic& left_stat, const KdeStatistic& right_stat) {
      SetZero();
    }
};
};
};

#endif
