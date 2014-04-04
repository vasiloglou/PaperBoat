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
#ifndef FL_LITE_FASTLIB_METRIC_KERNEL_COSINE_PREMETRIC_H_
#define FL_LITE_FASTLIB_METRIC_KERNEL_COSINE_PREMETRIC_H_
namespace fl {
namespace math {

class CosinePreMetric {
  public:
    /**
     * Computes the distance metric between two points.
     */

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t Distance(const PointType1 &a,
        const PointType2 &b) {

      typedef typename PointType1::CalcPrecision_t CalcPrecision_t;
      CalcPrecision_t a_dot_b = fabs(fl::la::Dot(a, b));
      CalcPrecision_t a_length = sqrt(fl::la::Dot(a, a));
      CalcPrecision_t b_length = sqrt(fl::la::Dot(b, b));
      CalcPrecision_t div_factor;
      if (a_length == 0 || b_length == 0) {
        div_factor = 1;
      }
      else {
        div_factor = a_length * b_length;
      }
      return a_dot_b / div_factor;
    }
};
}
}
#endif
