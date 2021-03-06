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
#ifndef FASTLIB_METRIC_KERNEL_WEIGHTED_LMETRIC_H_
#define FASTLIB_METRIC_KERNEL_WEIGHTED_LMETRIC_H_

#include "fastlib/math/fl_math.h"
#include "fastlib/dense/matrix.h"
#include "fastlib/data/monolithic_point.h"
#include "fastlib/la/linear_algebra.h"
#include "abstract_metric.h"

namespace fl {
namespace math {
/**
  * A Weighted L_p metric for vector spaces. It is actually supposed to work
  *  with mixed categorical data too
  *
  * A generic Metric class should simply compute the distance between
  * two points.  An LMetric operates for integer powers on Vector spaces.
  */
template < int t_pow, typename PointType1, typename PointType2,
typename WeightingContainerType >
class WeightedLMetric : AbstractMetric<PointType1, PointType2> {
  public:
    WeightedLMetric();

    WeightedLMetric(WeightingContainerType &weights);

    virtual typename PointType1::CalcPrecision_t Distance(
      const PointType1 &a,
      const PointType2 &b) const ;

    virtual typename PointType1::CalcPrecision_t DistanceSq(
      const PointType1 &a,
      const PointType2 &b) const ;

    virtual typename PointType1::CalcPrecision_t DistanceIneq(
      const PointType1 &a,
      const PointType2 &b) const ;

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {

    }

    void set_weights(WeightingContainerType &weights) {
      weights_ = weights;
    }

  private:
    WeightingContainerType weights_;
};

/**
 *  Spcial Case for a dense point that scales all dimensions    */
template<typename T, typename PointType1, typename PointType2>
class WeightedLMetric<2, PointType1, PointType2, fl::data::MonolithicPoint<T>  > :
      public AbstractMetric<PointType1, PointType2> {
  public:
    WeightedLMetric();
    WeightedLMetric(fl::data::MonolithicPoint<T> &weights);

    virtual typename PointType1::CalcPrecision_t Distance(
      const PointType1 &a,
      const PointType2 &b) const ;

    virtual typename PointType1::CalcPrecision_t DistanceSq(
      const PointType1 &a,
      const PointType2 &b) const ;

    virtual typename PointType1::CalcPrecision_t DistanceIneq(
      const PointType1 &a,
      const PointType2 &b) const ;

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {

    }

    void set_weights(fl::data::MonolithicPoint<T> &weights) {
      weights_.Alias(weights);
    }

  private:
    fl::data::MonolithicPoint<T> weights_;
};

/**
 *  Spcial Case for mixed point of 2 types one dense and one sparse,
 *  We will use this specialization for mixed continuous categorical
 *  We have two bandwidths (weights), one for dense and one for sparse
 */
template<typename PointType1, typename PointType2>
class WeightedLMetric < 2, PointType1, PointType2,
      std::pair < typename PointType1::CalcPrecision_t,
      typename PointType1::CalcPrecision_t > > :
      public AbstractMetric<PointType1, PointType2> {
  public:
    typedef std::pair < typename PointType1::CalcPrecision_t,
    typename PointType1::CalcPrecision_t > WeightingContainer_t;

    WeightedLMetric(WeightingContainer_t &weights);

    WeightedLMetric();

    virtual typename PointType1::CalcPrecision_t Distance(
      const PointType1 &a,
      const PointType2 &b) const ;

    virtual typename PointType1::CalcPrecision_t DistanceSq(
      const PointType1 &a,
      const PointType2 &b) const ;


    virtual typename PointType1::CalcPrecision_t DistanceIneq(
      const PointType1 &a,
      const PointType2 &b) const ;

    void set_weights(const WeightingContainer_t &weights) {
      weights_ = weights;
    }


    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {

    }
  private:
    WeightingContainer_t weights_;
};

/**
 *  Spcial Case for mixed point of 2 types one dense and one sparse,
 *  We will use this specialization for mixed continuous categorical
 *  We have two vector as weights.
 */
template<typename PointType1, typename PointType2>
class WeightedLMetric < 2, PointType1, PointType2,
      std::pair <
      fl::dense::Matrix<typename PointType1::CalcPrecision_t, true>,
      fl::dense::Matrix<typename PointType1::CalcPrecision_t, true> > > :
      public AbstractMetric<PointType1, PointType2> {



};

} // namespace math
}   // namespace fl


#endif
