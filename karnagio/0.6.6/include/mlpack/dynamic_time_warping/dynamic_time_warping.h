/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
LLC, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE ISMION INC "AS IS" AND ANY
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

#ifndef PAPERBOAT_MLPACK_DYNAMIC_TIME_WARPING_DYNAMIC_TIME_WARPING_H_
#define PAPERBOAT_MLPACK_DYNAMIC_TIME_WARPING_DYNAMIC_TIME_WARPING_H_
#include "fastlib/base/base.h"
#include "fastlib/data/monolithic_point.h"
#include "fastlib/data/sparse_point.h"

namespace fl { namespace ml {
  
  class DynamicTimeWarping {
    public:
      static double Compute(const std::vector<double> &query,
          const std::vector<double> &reference);
  
      template<typename PointType>
      static double Compute(const PointType &query,
          const PointType &reference, index_t q_index, index_t r_index);
  
      template<typename PointType>
      static double Compute(
          const PointType &query,
          const PointType &reference);
  
      template<typename IteratorType1, typename IteratorType2>
      static double Compute(
          IteratorType1 &query,
          IteratorType2 &reference, 
          IteratorType1 &q_end,
          IteratorType2 &r_end);
     
      template<typename PrecisionType1, typename PrecisionType2>
      static double ComputeBase(const PrecisionType1 &x, const PrecisionType2 &y);
  }; 


  class ConstrainedDynamicTimeWarping {
    public:
      static double Compute(
          std::vector<double> &query,
          std::vector<double> &reference, 
          index_t horizon);
    
      static double Compute(
          std::vector<double> &query,
          std::vector<double> &reference, 
          index_t q_index, 
          index_t r_index,
          index_t horizon); 
    
      template<typename PointType1, typename PointType2>
      static double Compute(
          PointType1 &query,
          PointType2 &reference, 
          index_t horizon); 
    
      template<typename IteratorType1, typename IteratorType2>
      static double Compute(
           IteratorType1 &query,
           IteratorType2 &reference, 
           IteratorType1 &q_end, 
           IteratorType2 &r_end,
          index_t horizon); 
  };

  class ScalingTimeWarping : public ConstrainedDynamicTimeWarping {
    public:
      template<typename PointType1, typename PointType2>
      static double Compute(PointType1 &query, PointType2 &reference,
          double scaling_factor, double horizon); 
  
      static index_t Size(const std::vector<double> &point); 
       
      template<typename PointType>
      static index_t Size(const PointType &point);   
  
      static void UniformScale(const std::vector<double> &point1, 
          index_t prefix_size,
          index_t new_size,
          std::vector<double> *point2);   
  
      template<typename PointType1>
      static void UniformScale(const PointType1 &point1, 
          index_t prefix_size,
          index_t new_size,
          fl::data::MonolithicPoint<double> *point2
          );   
      template<typename AggregatorType>
      static void BoundScalingTimeWarping(std::vector<double> &point,
          double scaling_factor, index_t horizon, 
          const AggregatorType &agg,
          std::vector<double> *bound_point);
  
      template<typename PointType, typename AggregatorType>
      static void BoundScalingTimeWarping(PointType &point,
          double scaling_factor, index_t horizon, 
          const AggregatorType &agg,
          fl::data::SparsePoint<double> *bound_point); 
    
      static double ScalingTimeWarpingLowerBound(std::vector<double> &query_point,
          std::vector<double> &reference_point,
          double scaling_factor, index_t horizon); 
    
      template<typename PointType>
      static double ScalingTimeWarpingLowerBound(PointType &query_point,
          PointType &reference_point,
          double scaling_factor, 
          index_t horizon); 
  };
  namespace dtw_private{
     template<typename PointType> 
      struct ScaledType {
        typedef fl::data::MonolithicPoint<double> type;
      };

      template<>
      struct ScaledType<std::vector<double> > {
        typedef std::vector<double> type;
      };
  }
}}


#endif
