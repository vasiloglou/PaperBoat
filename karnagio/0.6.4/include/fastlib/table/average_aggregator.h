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
#ifndef FL_LITE_FASTLIB_TABLE_AVERAGE_AGGREGATOR_H_
#define FL_LITE_FASTLIB_TABLE_AVERAGE_AGGREGATOR_H_

#include <vector>
#include "fastlib/la/linear_algebra.h"

namespace fl {
namespace table {
  class AverageAggregator {
    public:
      /**
       * @brief Initializes the 
       */
      void Init(index_t num_of_points) {
         counter_.resize(num_of_points);
      }

      template<typename PointType>
      void Reset(index_t point_id, PointType * const point) {
        counter_[point_id]=0;
        point->SetAll(0.0);
      }
      template<typename PointType>
      void Aggregate(const index_t point_id, 
                     const PointType &new_point,
                     PointType * const old_point) {
        typename PointType::CalcPrecision_t N;
        counter_[point_id]++;
        N=counter_[point_id];
        fl::logger->Message() << "Counter " << N;
        fl::la::AddExpert(N, new_point, old_point);      
        fl::la::SelfScale(N/(N+1.0), old_point);
      }

    private:
      std::vector<index_t> counter_;
  };
}}

#endif
