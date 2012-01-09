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

#ifndef FL_LITE_FASTLIB_TREE_CART_IMPURITY_H
#define FL_LITE_FASTLIB_TREE_CART_IMPURITY_H

namespace fl {
namespace tree {
class EntropyImpurity {
  public:

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t Distance(const PointType1 &a,
        const PointType2 &b) const {
      return 0;
    }

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t DistanceSq(
      const PointType1 &a, const PointType2 &b) const {
      return 0;
    }

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t DistanceIneq(
      const PointType1 &a, const PointType2 &b) const {
      return 0;
    }

    template<typename T>
    double Compute(const std::map<T, int> &class_counts, int total_counts,
                   T majority_class_label) const {

      double impurity = 0;
      typename std::map<T, int>::const_iterator it = class_counts.begin();

      for (; it != class_counts.end(); it++) {
        double probability = ((double) it->second) /
                             ((double) total_counts);
        impurity += (-probability * (log(probability) / log(2)));
      }
      return impurity;
    }
};

class GiniImpurity {
  public:

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t Distance(const PointType1 &a,
        const PointType2 &b) const {
      return 0;
    }

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t DistanceSq(
      const PointType1 &a, const PointType2 &b) const {
      return 0;
    }

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t DistanceIneq(
      const PointType1 &a, const PointType2 &b) const {
      return 0;
    }

    template<typename T>
    double Compute(const std::map<T, int> &class_counts, int total_counts,
                   T majority_class_label) const {

      double impurity = 0;
      typename std::map<T, int>::const_iterator it = class_counts.begin();

      for (; it != class_counts.end(); it++) {
        double probability = ((double) it->second) /
                             ((double) total_counts);
        impurity += fl::math::Sqr(probability);
      }
      impurity = 1.0 - impurity;
      impurity *= 0.5;
      return impurity;
    }
};

class MisclassificationImpurity {
  public:

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t Distance(const PointType1 &a,
        const PointType2 &b) const {
      return 0;
    }

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t DistanceSq(
      const PointType1 &a, const PointType2 &b) const {
      return 0;
    }

    template<typename PointType1, typename PointType2>
    typename PointType1::CalcPrecision_t DistanceIneq(
      const PointType1 &a, const PointType2 &b) const {
      return 0;
    }

    template<typename T>
    double Compute(const std::map<T, int> &class_counts, int total_counts,
                   T majority_class_label) const {

      return 1.0 -
             ((double)(class_counts.find(majority_class_label)->second)) /
             ((double) total_counts);
    }
};
};
};

#endif
