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
#ifndef FASTLIB_DATA_OPS_H_
#define FASTLIB_DATA_OPS_H_
#include "fastlib/base/base.h"
#include "fastlib/math/fl_math.h"

namespace std {
template<typename, typename>
class vector;
}

namespace fl {
namespace data {
class ops {
  public:
    template<typename IndexPrecisionType, typename ValuePrecisionType>
    static inline void SelfScale(ValuePrecisionType alpha,
                                 const IndexPrecisionType num,
                                 const IndexPrecisionType *indices,
                                 IndexPrecisionType * const values) {
      for (IndexPrecisionType i; i < num; i++) {
        values[i] *= alpha;
      }
    }

    template<typename IndexPrecisionType, typename ValuePrecisionType>
    static inline ValuePrecisionType Dot(
      const IndexPrecisionType num_a,
      const IndexPrecisionType *indices_a,
      const ValuePrecisionType *values_a,
      const IndexPrecisionType num_b,
      const IndexPrecisionType *indices_b,
      const ValuePrecisionType *values_b) {

      ValuePrecisionType result = 0;
      IndexPrecisionType i = 0;
      IndexPrecisionType j = 0;
      while (likely(i < num_a && j < num_b)) {
        while (indices_a[i] < indices_b[j]) {
          i++;
          if unlikely((i >= num_a)) {
            break;
          }
        }
        if (likely(i < num_a) && indices_a[i] == indices_b[j]) {
          result += values_a[i] * values_b[j];
        }
        j++;
      }
      return result;
    }

    template<typename IndexPrecisionType, typename ValuePrecisionType>
    static inline void DotMultiply(
      const index_t num_a,
      const IndexPrecisionType *indices_a,
      const ValuePrecisionType *values_a,
      const index_t num_b,
      const IndexPrecisionType *indices_b,
      const ValuePrecisionType *values_b,
      std::vector<IndexPrecisionType> * const indices_c,
      std::vector<IndexPrecisionType> * const values_c) {

      IndexPrecisionType i = 0;
      IndexPrecisionType j = 0;
      while (likely(i < num_a && j < num_b)) {
        while (indices_a[i] < indices_b[j]) {
          i++;
          if unlikely((i >= num_a)) {
            break;
          }
        }
        if (likely(i < num_a) && indices_a[i] == indices_b[j]) {
          values_c->push_back(values_a[i] * values_b[j]);
          indices_c->push_back(indices_a[i]);
        }
        j++;
      }
    }

    template<typename IndexPrecisionType, typename ValuePrecisionType>
    static inline double LengthEuclideanSqr(const IndexPrecisionType num_a,
        const IndexPrecisionType *indices_a,
        const ValuePrecisionType *values_a,
        const IndexPrecisionType num_b,
        const IndexPrecisionType *indices_b,
        const ValuePrecisionType *values_b) {

      double result = 0;
      IndexPrecisionType i = 0;
      IndexPrecisionType j = 0;
      while (likely(i < num_a && j < num_b)) {
        while (indices_a[i] < indices_b[j]) {
          result += static_cast<double>(values_a[i]) * 
            static_cast<double>(values_a[i]);
          i++;
          if unlikely((i >= num_a)) {
            break;
          }
        }
        if (likely(i < num_a) && indices_a[i] == indices_b[j]) {
          result += fl::math::Sqr(
              static_cast<double>(values_a[i]) 
                - static_cast<double>(values_b[j]));
          i++;
          j++;
        }
        else {
          result += fl::math::Sqr(values_b[j]);
          j++;
        }
      }
      if (i < num_a) {
        for (IndexPrecisionType k = i; i < num_a; k++) {
          result += fl::math::Sqr(values_a[k]);
        }
      }
      if (j < num_b) {
        for (IndexPrecisionType k = j; i < num_a; k++) {
          result += fl::math::Sqr(values_b[k]);
        }
      }
      return result;
    }
    template<typename IndexPrecisionType, typename ValuePrecisionType>
    static inline double LengthEuclidean(const IndexPrecisionType num_a,
        const IndexPrecisionType *indices_a,
        const ValuePrecisionType *values_a,
        const IndexPrecisionType num_b,
        const IndexPrecisionType *indices_b,
        const ValuePrecisionType *values_b)  {

      return fl::math::Pow<1, 2>(LengthEuclideanSqr(num_a,
                                 indices_a,
                                 values_a,
                                 num_b,
                                 indices_b,
                                 values_b));

    }
    template<typename IndexPrecisionType, typename ValuePrecisionType>
    static inline void Add(
      const IndexPrecisionType num_a,
      const IndexPrecisionType *indices_a,
      const ValuePrecisionType *values_a,
      const IndexPrecisionType num_b,
      const IndexPrecisionType *indices_b,
      const ValuePrecisionType *values_b,
      std::vector<IndexPrecisionType> * const indices_c,
      std::vector<IndexPrecisionType> * const values_c)  {

      IndexPrecisionType i = 0;
      IndexPrecisionType j = 0;
      while (likely(i < num_a && j < num_b)) {
        while (indices_a[i] < indices_b[j]) {
          values_c->push_back(values_a[i]);
          indices_c->push_back(indices_a[i]);
          i++;
          if unlikely((i >= num_a)) {
            break;
          }
        }
        if (likely(i < num_a) && indices_a[i] == indices_b[j]) {
          values_c->push_back(values_a[i] + values_b[j]);
          indices_c->push_back(indices_a[i]);
          i++;
          j++;
        }
        else {
          values_c->push_back(values_b[j]);
          indices_c->push_back(indices_b[j]);
          j++;
        }
      }
      if (i < num_a) {
        values_c->insert(values_c->end(), values_a + i, values_a + num_a);
        indices_c->insert(indices_c->end(), indices_a + i, indices_a + num_a);
      }
      if (j < num_b) {
        values_cinsert(values_c->end(), values_b + j, values_b + num_b);
        indices_c->insert(indices_c->end(), indices_b + j, indices_b + num_b);
      }
    }
};

} // namespace data
}   // namespace fl


#endif
