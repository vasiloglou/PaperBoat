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
#ifndef FASTLIB_METRIC_KERNEL_ABSTRACT_METRIC_H_
#define FASTLIB_METRIC_KERNEL_ABSTRACT_METRIC_H_

namespace fl {
namespace math {
/**
  * An abstract metric. Every metric should inherit from this one
  */
template<typename PointType1, typename PointType2>
class AbstractMetric {
  public:

    virtual typename PointType1::CalcPrecision_t Distance(const PointType1 &a,
        const PointType2 &b) const = 0;

    virtual typename PointType1::CalcPrecision_t DistanceSq(
      const PointType1 &a, const PointType2 &b) const = 0;

    virtual typename PointType1::CalcPrecision_t DistanceIneq(
      const PointType1 &a, const PointType2 &b) const = 0;

};
}
}

#endif
