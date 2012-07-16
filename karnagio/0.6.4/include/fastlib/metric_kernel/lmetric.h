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
#ifndef FL_LITE_INCLUDE_FASTLIB_METRIC_KERNEL_METRIC_H_
#define FL_LITE_INCLUDE_FASTLIB_METRIC_KERNEL_METRIC_H_

#include "fastlib/math/fl_math.h"
#include "fastlib/dense/matrix.h"
#include "fastlib/la/linear_algebra.h"

namespace fl {
namespace math {
/**
   * An L_p metric for vector spaces.
   *
   * A generic Metric class should simply compute the distance between
   * two points.  An LMetric operates for integer powers on Vector spaces.
   */
template<int t_pow>
class LMetric {
  public:

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t Distance(const PointType1 &a,
        const PointType2 &b) {
      return fl::math::Pow<typename PointType1::CalcPrecision_t, 1, t_pow>(
               DistanceIneq(a, b));
    }

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t DistanceSq(
      const PointType1 &a, const PointType2 &b) {
      return fl::math::Pow<typename PointType1::CalcPrecision_t, 2, 1>(
               Distance(a, b));
    }

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t DistanceIneq(
      const PointType1 &a, const PointType2 &b) {
      return fl::la::RawLMetric<t_pow>(a, b);
    }

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {

    }
};

template<>
class LMetric<2> {
  public:

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t Distance(const PointType1 &a,
        const PointType2 &b) const {
      return fl::math::Pow<typename PointType1::CalcPrecision_t, 1, 2>(
               DistanceIneq(a, b));
    }

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t DistanceSq(
      const PointType1 &a, const PointType2 &b) const {
      typename PointType1::CalcPrecision_t result;
      fl::la::RawLMetric<2>(a, b, &result);
      return result;
    }

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t DistanceIneq(
      const PointType1 &a, const PointType2 &b) const {
      return DistanceSq(a, b);
    }

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {

    }
};
}
}

#endif
