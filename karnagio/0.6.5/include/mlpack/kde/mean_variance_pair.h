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

#ifndef FL_LITE_MLPACK_KDE_MEAN_VARIANCE_PAIR_H
#define FL_LITE_MLPACK_KDE_MEAN_VARIANCE_PAIR_H

namespace fl {
namespace ml {
class MeanVariancePair {

  private:
    int num_samples_;

    double sample_mean_;

    double sample_variance_;

  public:

    MeanVariancePair() {
      SetZero();
    }

    void Copy(const MeanVariancePair &pair_in) {
      num_samples_ = pair_in.num_samples();
      sample_mean_ = pair_in.sample_mean();
      sample_variance_ = pair_in.sample_variance();
    }

    int num_samples() const {
      return num_samples_;
    }

    double sample_mean() const {
      return sample_mean_;
    }

    double sample_mean_variance() const {
      return sample_variance_ / ((double) num_samples_);
    }

    double sample_variance() const {
      return sample_variance_;
    }

    void scaled_interval(double scale_in, double standard_deviation_factor,
                         GenRange<double> *interval_out) const {
      // Compute the sample mean variance.
      double sample_mean_variance = this->sample_mean_variance();
      double error = standard_deviation_factor * sqrt(sample_mean_variance);

      // In case no sample has been collected, then we need to set the
      // error to zero (since the variance will be infinite).
      if (num_samples_ == 0) {
        error = 0;
      }

      // Compute the interval.
      interval_out->lo = scale_in * (sample_mean_ - error);
      interval_out->hi = scale_in * (sample_mean_ + error);
    }

    void SetZero() {
      num_samples_ = 0;
      sample_mean_ = 0;
      sample_variance_ = 0;
    }

    void Add(const MeanVariancePair &mv_pair_in) {
      sample_mean_ += mv_pair_in.sample_mean();
      sample_variance_ += mv_pair_in.sample_variance();
    }

    void push_back(double sample) {

      // Update the number of samples.
      num_samples_++;
      double delta = sample - sample_mean_;

      // Update the sample mean.
      sample_mean_ = sample_mean_ + delta / ((double) num_samples_);

      // Update the sample variance.
      sample_variance_ = ((num_samples_ - 1) * sample_variance_ +
                          delta * (sample - sample_mean_)) /
                         ((double) num_samples_);
    }
};
};
};

#endif
