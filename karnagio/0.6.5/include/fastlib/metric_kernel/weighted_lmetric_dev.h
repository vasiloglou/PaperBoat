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
#ifndef FASTLIB_METRIC_KERNEL_WEIGHTED_LMETRIC_DEV_H_
#define FASTLIB_METRIC_KERNEL_WEIGHTED_LMETRIC_DEV_H_

#include "boost/mpl/front.hpp"
#include "weighted_lmetric.h"
#include "fastlib/dense/linear_algebra.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/data/sparse_point.h"
#include "fastlib/data/monolithic_point.h"

namespace fl {
namespace math {

template<typename PrecisionType1, typename PrecisionType2>
WeightedLMetric<2, std::pair<PrecisionType1, PrecisionType2> >::WeightedLMetric(
  typename WeightedLMetric < 2, std::pair < PrecisionType1,
  PrecisionType2 > >::WeightingContainer_t &weights) : weights_(weights) {
}

template<typename PrecisionType1, typename PrecisionType2>
WeightedLMetric<2, std::pair<PrecisionType1, PrecisionType2> >::WeightedLMetric(
) {
  weights_.first = 1;
  weights_.second = 1;
}

template<typename PrecisionType1, typename PrecisionType2>
template<typename PointType1, typename PointType2>
typename PointType1::CalcPrecision_t  WeightedLMetric < 2, std::pair < PrecisionType1,
PrecisionType2 > >::Distance(const PointType1 &a, const PointType2 &b) const {
  return ::fl::math::Pow<typename PointType1::CalcPrecision_t, 1, 2>(
           DistanceIneq(a, b));

}

template<typename PrecisionType1, typename PrecisionType2>
template<typename PointType1, typename PointType2>
typename PointType1::CalcPrecision_t  WeightedLMetric < 2, std::pair < PrecisionType1,
PrecisionType2 > >::DistanceSq(const PointType1 &a, const PointType2 &b) const {
  //  we need to assure that it is a mixed point with one
  //  dense one sparse and the precisions match
  // BOOST_MPL_ASSERT();
  typename PointType1::CalcPrecision_t s1;
  typename PointType2::CalcPrecision_t s2;
  typedef typename boost::mpl::front<typename PointType1::DenseTypeList_t>::type DensePrecision;
  typedef typename boost::mpl::front<typename PointType1::SparseTypeList_t>::type SparsePrecision;

  typename ::fl::la::template RawLMetric<2>(a.template dense_point<DensePrecision>(),
                                          b.template dense_point<DensePrecision>(), &s1);
  typename ::fl::la::template RawLMetric<2>(a.template sparse_point<SparsePrecision>(),
                                          b.template sparse_point<SparsePrecision>(), &s2);
  return (weights_.first*s1 + weights_.second*s2);

}

template<typename PrecisionType1, typename PrecisionType2>
template<typename PointType1, typename PointType2>
typename PointType1::CalcPrecision_t  WeightedLMetric < 2, std::pair < PrecisionType1,
PrecisionType2 > >::DistanceIneq(const PointType1 &a, const PointType2 &b) const {
  return DistanceSq(a, b);
}

/**
 *  Spcial Case for a dense point that scales all dimensions    */
template<typename T>
WeightedLMetric<2, ::fl::data::MonolithicPoint<T> >::WeightedLMetric(
  ::fl::data::MonolithicPoint<T> &weights) {
  weights_.Copy(weights);
  // do a test on the wieights to see if they are all positive
  for (index_t i = 0; i < weights_.size(); i++) {
    if (weights_[i] < 0) {
      ::fl::logger->Die() << "The " << i << "th  element of the weight vector "
      << "is negative " << weights_[i];
    }
  }
}

template<typename T>
template<typename PointType1, typename PointType2>
typename PointType1::CalcPrecision_t
WeightedLMetric<2, ::fl::data::MonolithicPoint<T> >::Distance(
  const PointType1 &a,
  const PointType2 &b) const {
  return ::fl::math::Pow<typename PointType1::CalcPrecision_t, 1, 2>(
           DistanceSq(a, b));
}

template<typename T>
template<typename PointType1, typename PointType2>
typename PointType1::CalcPrecision_t
WeightedLMetric<2, ::fl::data::MonolithicPoint<T> >::DistanceSq(
  const PointType1 &a,
  const PointType2 &b) const {
  typename PointType1::CalcPrecision_t result;
  typename ::fl::la::template RawLMetric<2>(weights_, a, b, &result);
  return result;

}

template<typename T>
template<typename PointType1, typename PointType2>
typename PointType1::CalcPrecision_t
WeightedLMetric<2, ::fl::data::MonolithicPoint<T> >::DistanceIneq(
  const PointType1 &a,
  const PointType2 &b) const {
  DistanceSq(a, b);
}

}} // namespaces

#endif
