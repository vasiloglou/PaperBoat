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

#ifndef FL_LITE_INLCLUDE_METRIC_KERNEL_DIVERGENCE_H
#define FL_LITE_INLCLUDE_METRIC_KERNEL_DIVERGENCE_H

#include "fastlib/data/monolithic_point.h"
#include "fastlib/math/fl_math.h"

namespace fl {
namespace math {
class KLDivergence {
  public:

    template<typename PointType1, typename PointType2>
    double Divergence(const PointType1 &x, const PointType2 &y) const {
      double divergence = 0;
      for (int i = 0; i < x.length(); i++) {
        if (x[i] > std::numeric_limits<double>::min()) {
          // Prevent numerical blow-ups.
          if (y[i] <= std::numeric_limits<double>::min()) {
            divergence = std::numeric_limits<double>::max();
            break;
	  }
          divergence += (x[i] * log(x[i] / y[i]) + y[i] - x[i]);
        }
      }
      return divergence;
    }

    template<typename PointType1, typename PointType2>
    double DistanceSq(const PointType1 &x, const PointType2 &y) const {
      return Divergence(x, y);
    }

    template<typename PointType1, typename PointType2>
    double Distance(const PointType1 &x, const PointType2 &y) const {
      return fl::math::Pow<double, 1,2>(Divergence(x, y));
    }

    template<typename PointType1, typename PointType2>
    void Gradient(
      const PointType1 &x, PointType2 *gradient) const {

      for (int i = 0; i < x.length(); i++) {
        gradient->set(i, log(x[i]) + 1.0);
      }
    }

    template<typename PointType1, typename PointType2>
    void GradientConvexConjugate(
      const PointType1 &x, PointType2 *gradient) const {

      for (int i = 0; i < x.length(); i++) {
        gradient->set(i, exp(x[i] - 1.0));
      }
    }
};
};
};

#endif
