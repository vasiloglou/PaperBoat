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

#ifndef FL_LITE_FASTLIB_TREE_HYPER_RECTANGLE_BOUND_H_
#define FL_LITE_FASTLIB_TREE_HYPER_RECTANGLE_BOUND_H_
#include "fastlib/data/monolithic_point.h"
namespace fl {
namespace tree {
template<typename CalcPrecisionType>
class HyperRectangleBound: public fl::data::MonolithicPoint<CalcPrecisionType> {
  public:
    typedef CalcPrecisionType CalcPrecision_t;
    void Init(int length) {
      fl::data::MonolithicPoint<CalcPrecision_t>::Init(length);
    }
    template<typename TableType>
    void Init(TableType &table) {
      fl::data::MonolithicPoint<CalcPrecision_t>::Init(table.n_attributes());
    }

    CalcPrecision_t MaxDistanceWithinBound() const {
      return std::numeric_limits<CalcPrecision_t>::max();
    }

    template<typename StreamType>
    void Print(StreamType &stream, const std::string &delim) const {
      stream << "*HyperRectangleBound*" << "\n";
      stream << "ranges:";
      for (index_t i = 0; i < this->size(); i++) {
         stream << this->get(i) << delim;
      }
      stream << "\n";
    }


    /** Calculates the midpoint of the range, cetroid must be initialized */
    template<typename PointType>
    void CalculateMidpoint(PointType *centroid) const {
      fl::logger->Die() << "Don't use this";
    }

    /** Calculates the midpoint of the range */
    template<typename PointType>
    void CalculateMidpointOverwrite(PointType *centroid) const {
      fl::logger->Die() << "Don't use this";
    }
};
}
}
#endif
