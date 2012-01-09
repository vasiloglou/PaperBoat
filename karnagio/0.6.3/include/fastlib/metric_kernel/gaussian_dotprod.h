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

#ifndef FL_LITE_FASTLIB_METRIC_KERNEL_GAUSSIAN_DOT_PRODUCT_H_
#define FL_LITE_FASTLIB_METRIC_KERNEL_GAUSSIAN_DOT_PRODUCT_H_

#include "fastlib/math/fl_math.h"
#include "boost/utility.hpp"

namespace fl {
namespace math {
/**
 * The Gaussian Kernel for vector spaces.
 */
template<typename CalcPrecision_t, typename GeometryType>
class GaussianDotProduct  {
  private:
    fl::math::GaussianKernel<CalcPrecision_t> kernel_;
    double bandwidth_;
    GeometryType geometry_;
  public:
    GaussianDotProduct() {
      bandwidth_=0;
    }
    template<typename Precision>
    void Init(Precision bandwidth_in, GeometryType &geometry) {
      geometry_=geometry;
      kernel_.Init(bandwidth_in);
      bandwidth_ = bandwidth_in;
    }

    void set(CalcPrecision_t bandwidth) {
      kernel_.Init(bandwidth);
      bandwidth_=bandwidth;
    }
    const GeometryType geometry() const {
      return geometry_;
    } 
    
    const CalcPrecision_t bandwidth() const {
      return bandwidth_;
    }

    /**
     * Computes the distance metric between two points.
     */
    template<typename PointType1, typename PointType2>
    const CalcPrecision_t Dot(
      const PointType1 &a,
      const PointType2 &b) const {

      CalcPrecision_t squared_distance =
        geometry_.DistanceSq(a, b);
      return kernel_.EvalUnnormOnSq(squared_distance);
    }
    
    template<typename PrecisionType>
    const double Dot(PrecisionType squared_distance) {
      return kernel_.EvalUnnormOnSq(squared_distance);
    }

    template<typename Point_t>
    const CalcPrecision_t NormSq(const Point_t &a) const {
      return 1.0;
    }
};
}
}

#endif
