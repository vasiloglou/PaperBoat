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

#ifndef PAPER_BOAT_INCLUDE_MLPACK_NONPARAMETRIC_REGRESSION_STAT_H_
#define PAPER_BOAT_INCLUDE_MLPACK_NONPARAMETRIC_REGRESSION_STAT_H_
#include "mlpack/kde/kde_stat.h"
#include "boost/math/special_functions/fpclassify.hpp" 
namespace fl {
namespace ml {

template<typename CalcPrecision_t>
class NprPostponed : public KdePostponed<CalcPrecision_t> {
  public:
    CalcPrecision_t weighted_densities_l_;
    CalcPrecision_t weighted_densities_u_;
    
    void Init(index_t rnode_count) {
      weighted_densities_l_ = weighted_densities_u_ = 0;
      KdePostponed<CalcPrecision_t>::Init(rnode_count);
    }

    template<typename NprDelta, typename ResultType>
    void ApplyDelta(const NprDelta &delta_in,
                    ResultType *query_results) {
      KdePostponed<CalcPrecision_t>::ApplyDelta(delta_in, query_results);
      weighted_densities_l_ = ((CalcPrecision_t) weighted_densities_l_ +
                      (CalcPrecision_t) delta_in.weighted_densities_l_);
      weighted_densities_u_ = ((CalcPrecision_t) weighted_densities_u_ +
                      (CalcPrecision_t) delta_in.weighted_densities_u_);
    }

    void ApplyPostponed(const NprPostponed &other_postponed)  {
      KdePostponed<CalcPrecision_t>::ApplyPostponed(static_cast<const KdePostponed<CalcPrecision_t>&>(other_postponed));
      weighted_densities_l_ = ((CalcPrecision_t) weighted_densities_l_ +
                      (CalcPrecision_t) other_postponed.weighted_densities_l_);
      weighted_densities_u_ = ((CalcPrecision_t) weighted_densities_u_ +
                      (CalcPrecision_t) other_postponed.weighted_densities_u_);
    }

    template<typename GlobalType, typename MetricType, typename PointType>
    void ApplyContribution(const GlobalType &global,
                           const MetricType &metric,
                           const PointType &query_point,
                           const PointType &reference_point) {
      double distsq = metric.DistanceSq(query_point, reference_point);
      double density_incoming = global.kernel().EvalUnnormOnSq(distsq);
      this->densities_l_ = ((CalcPrecision_t) this->densities_l_ + density_incoming);
      this->densities_u_ = ((CalcPrecision_t) this->densities_u_ + density_incoming);
      double weighted_density_incoming = density_incoming*reference_point.meta_data().template get<1>();
      weighted_densities_l_=((CalcPrecision_t) weighted_densities_l_ + weighted_density_incoming);
      weighted_densities_u_=((CalcPrecision_t) weighted_densities_u_ + weighted_density_incoming);
    }

    void SetZero() {
      KdePostponed<CalcPrecision_t>::SetZero();
      weighted_densities_l_ = 0;
      weighted_densities_u_ = 0;
    }
};

template<typename TemplateMap>
class NprGlobal : public KdeGlobal<TemplateMap> {

  public:

    typedef typename TemplateMap::TableType Table_t;

    typedef typename TemplateMap::KernelType Kernel_t;

    typedef typename TemplateMap::TableType::CalcPrecision_t CalcPrecision_t;
    

};

template<typename ContainerType>
class NprResult : public KdeResult<ContainerType> {
  public:
    typedef typename ContainerType::value_type CalcPrecision_t;
    ContainerType weighted_densities_l_;
    ContainerType weighted_densities_;
    ContainerType weighted_densities_u_;
    ContainerType predictions_;

    NprResult() {
    }

    ~NprResult() {
    }

    template<typename MetricType, typename GlobalType>
    void PostProcess(const MetricType &metric,
                     index_t q_index,
                     const GlobalType &global,
                     const bool is_monochromatic) {
      // we need to review this and see if we can add the known value at q_index
      if (is_monochromatic) {
        this-> densities_l_[q_index] = (CalcPrecision_t)(this->densities_l_[q_index] -
                                1);
        this->densities_u_[q_index] = (CalcPrecision_t)(this->densities_u_[q_index] -
                                1);

        // weighted_densities_l_[q_index] = (CalcPrecision_t)(weighted_densities_l_[q_index] -
        //                        1);
        //  weighted_densities_u_[q_index] = (CalcPrecision_t)(weighted_densities_u_[q_index] -
        //                        1);
      }

      weighted_densities_[q_index] = 0.5 * (weighted_densities_l_[q_index] +
                                   weighted_densities_u_[q_index]);
      this->densities_[q_index] = 0.5 * (this->densities_l_[q_index] +
                                   this->densities_u_[q_index]);
      double val=predictions_[q_index] = weighted_densities_[q_index]/this->densities_[q_index];
      if (boost::math::isnan(val) || boost::math::isinf(val)) {
        predictions_[q_index]=0;
      }
      this->densities_l_[q_index] *= global.get_mult_const();
      this->densities_[q_index] *= global.get_mult_const();
      this->densities_u_[q_index] *= global.get_mult_const();

    }

    void PrintDebug(const std::string &file_name) {
      FILE *file_output = fopen(file_name.c_str(), "w+");
      for (unsigned int i = 0; i < this->densities_.size(); i++) {
        fprintf(file_output, "%g %g %g %g %g %g %g\n", this->densities_l_[i],
                this->densities_[i], this->densities_u_[i], weighted_densities_l_, weighted_densities_, 
                weighted_densities_u_, this->pruned_[i]);
      }
      fclose(file_output);
    }

    template<int mode, typename TableType>
    void GetPredictions(TableType* table) {
      if (mode == 0) {
        table->Init("", std::vector<index_t>(1, 1), std::vector<index_t>(),
                    this->densities_.size());
      }
      for (int i = 0; i < this->densities_.size(); i++) {
        typename TableType::Point_t point;
        table->get(i, &point);
        point.set(0, predictions_[i]);
      }
    }

    void Init(int num_points) {
      KdeResult<ContainerType>::Init(num_points);
      weighted_densities_l_.resize(num_points);
      weighted_densities_.resize(num_points);
      weighted_densities_u_.resize(num_points);
      predictions_.resize(num_points);
      SetZero();
    }

    void SetZero() {
      KdeResult<ContainerType>::SetZero();
      for (int i = 0; i < static_cast<int>(this->densities_l_.size()); i++) {
        weighted_densities_l_[i] = 0;
        weighted_densities_[i] = 0;
        weighted_densities_u_[i] = 0;
        predictions_[i]=0;
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
        this->densities_l_[qpoint_index] += contribution.lo;
        this->densities_u_[qpoint_index] += contribution.hi;
        this->pruned_[qpoint_index] += delta_in.pruned_;
      }
      while (qnode_it.HasNext());
    }

    void ApplyPostponed(int q_index,
                        const NprPostponed<CalcPrecision_t> &postponed_in) {

      KdeResult<ContainerType>::ApplyPostponed(q_index, static_cast<const KdePostponed<CalcPrecision_t>&>(postponed_in));
      weighted_densities_l_[q_index] = ((CalcPrecision_t) weighted_densities_l_[q_index] +
                               (CalcPrecision_t) postponed_in.weighted_densities_l_);
      weighted_densities_u_[q_index] = ((CalcPrecision_t) weighted_densities_u_[q_index] +
                               (CalcPrecision_t) postponed_in.weighted_densities_u_);
    }
};

template<typename CalcPrecision_t>
class NprDelta : public KdeDelta<CalcPrecision_t> {

  public:

    CalcPrecision_t weighted_densities_l_;

    CalcPrecision_t weighted_densities_u_;


    NprDelta()  {
      SetZero();
    }

    ~NprDelta() {
    }

    void SetZero() {
      weighted_densities_l_ = weighted_densities_u_ = 0;
    }

    // we need to find a way to introduce the hi and the low in the values
    template<typename MetricType, typename GlobalType, typename TreeType>
    void DeterministicCompute(
      const MetricType &metric,
      const GlobalType &global, TreeType *qnode, TreeType *rnode,
      const GenRange<CalcPrecision_t> &squared_distance_range) {
      
      KdeDelta<CalcPrecision_t>::DeterministicCompute(metric, global, qnode, rnode, squared_distance_range);
      index_t rnode_count = global.reference_table()->get_node_count(rnode);
      index_t rnode_id=global.reference_table()->get_node_id(rnode);
      const typename GlobalType::Statistic_t &stat= global.reference_statistic()->at(
         rnode_id);
      CalcPrecision_t value_lo=stat.prediction_value_lo;
      CalcPrecision_t value_hi=stat.prediction_value_hi;
      weighted_densities_l_ = rnode_count *
                     global.kernel().EvalUnnormOnSq(squared_distance_range.hi) * value_lo;
      weighted_densities_u_ = rnode_count *
                     global.kernel().EvalUnnormOnSq(squared_distance_range.lo) * value_hi;
    }
};

template<typename CalcPrecision_t>
class NprSummary : public KdeSummary<CalcPrecision_t> {

  public:

    CalcPrecision_t weighted_densities_l_;

    CalcPrecision_t weighted_densities_u_;


    NprSummary() {
      SetZero();
    }

    ~NprSummary() {
    }

    NprSummary(const NprSummary &summary_in) : KdeSummary<CalcPrecision_t>(summary_in) {
      weighted_densities_l_ = summary_in.weighted_densities_l_;
      weighted_densities_u_ = summary_in.weighted_densities_u_;
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
        global.relative_error() * this->densities_l_ /
        static_cast<double>(global.reference_table()->n_entries());

      return left_hand_side <= right_hand_side;
    }

    void SetZero() {
      KdeSummary<CalcPrecision_t>::SetZero();
      weighted_densities_l_ = weighted_densities_u_ = 0;
    }

    void Init() {
      KdeSummary<CalcPrecision_t>::SetZero();
      SetZero();
    }

    void StartReaccumulate() {
      KdeSummary<CalcPrecision_t>::StartReaccumulate();
      weighted_densities_l_ = std::numeric_limits<CalcPrecision_t>::max();
      weighted_densities_u_ = -std::numeric_limits<CalcPrecision_t>::max();
    }

    template<typename ResultType>
    void Accumulate(const ResultType &results, index_t q_index) {
      KdeSummary<CalcPrecision_t>::Accumulate(results, q_index);
      weighted_densities_l_ = std::min(weighted_densities_l_, results.weighted_densities_l_[q_index]);
      weighted_densities_u_ = std::max(weighted_densities_u_, results.weighted_densities_u_[q_index]);
    }

    void Accumulate(const NprSummary<CalcPrecision_t> &summary_in) {
      KdeSummary<CalcPrecision_t>::Accumulate(summary_in);
      weighted_densities_l_ = std::min(weighted_densities_l_, summary_in.weighted_densities_l_);
      weighted_densities_u_ = std::max(weighted_densities_u_, summary_in.weighted_densities_u_);
    }

    void Accumulate(const NprSummary<CalcPrecision_t> &summary_in,
                    const NprPostponed<CalcPrecision_t> &postponed_in) {
      KdeSummary<CalcPrecision_t>::Accumulate(summary_in, postponed_in);
      weighted_densities_l_ = std::min(weighted_densities_l_, summary_in.weighted_densities_l_ +
                              postponed_in.weighted_densities_l_);
      weighted_densities_u_ = std::max(weighted_densities_u_, summary_in.weighted_densities_u_ +
                              postponed_in.weighted_densities_u_);
    }

    void ApplyDelta(const NprDelta<CalcPrecision_t> &delta_in) {
      KdeSummary<CalcPrecision_t>::ApplyDelta(delta_in);
      weighted_densities_l_ = ((CalcPrecision_t) weighted_densities_l_ +
                      (CalcPrecision_t) delta_in.weighted_densities_l_);
      weighted_densities_u_ = ((CalcPrecision_t) weighted_densities_u_ +
                      (CalcPrecision_t) delta_in.weighted_densities_u_);
    }

    void ApplyPostponed(const NprPostponed<CalcPrecision_t>
                        &postponed_in) {

      KdeSummary<CalcPrecision_t>::ApplyPostponed(postponed_in);

      weighted_densities_l_ = ((CalcPrecision_t) weighted_densities_l_ +
                      (CalcPrecision_t) postponed_in.weighted_densities_l_);
      weighted_densities_u_ = ((CalcPrecision_t) weighted_densities_u_ +
                      (CalcPrecision_t) postponed_in.weighted_densities_u_);
    }
};

template<typename CalcPrecision_t>
class NprStatistic : public fl::tree::AbstractStatistic {

  public:

    fl::ml::NprPostponed<CalcPrecision_t> postponed;

    fl::ml::NprSummary<CalcPrecision_t> summary;
 
    CalcPrecision_t prediction_value_lo;

    CalcPrecision_t prediction_value_hi;

    NprStatistic() {
    }

    ~NprStatistic() {
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
      it.Reset();
      typename TreeIterator::Point_t point;
      while(it.HasNext()) {
        index_t point_id;
        it.Next(&point, &point_id);
        prediction_value_lo=std::min(prediction_value_lo, 
            point.meta_data().template get<1>());
        prediction_value_hi=std::max(prediction_value_hi, 
            point.meta_data().template get<1>());

      }
    }

    /**
     * Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    template<typename TreeIterator>
    void Init(TreeIterator &it,
              const NprStatistic& left_stat, const NprStatistic& right_stat) {
      SetZero();
      prediction_value_lo=std::min(left_stat.prediction_value_lo,
          right_stat.prediction_value_lo);
      prediction_value_hi=std::max(left_stat.prediction_value_hi,
          right_stat.prediction_value_hi);
    }
};


}} // namespaces 
#endif
