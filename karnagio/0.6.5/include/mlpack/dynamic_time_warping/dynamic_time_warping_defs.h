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

#ifndef PAPERBOAT_MLPACK_DYNAMIC_TIME_WARPING_DYNAMIC_TIME_WARPING_DEFS_H_
#define PAPERBOAT_MLPACK_DYNAMIC_TIME_WARPING_DYNAMIC_TIME_WARPING_DEFS_H_
#include "dynamic_time_warping.h"
#include "fastlib/base/base.h"

namespace fl { 
  namespace ml {
  
    double DynamicTimeWarping::Compute(const std::vector<double> &query,
        const std::vector<double> &reference) {

      std::vector<std::vector<double> > dtw_matrix;
      if (query.size()==0 || reference.size()==0) {
        return std::numeric_limits<double>::max();
      }
      std::vector<double> *old_horizon=new std::vector<double>(
          reference.size(), 
          std::numeric_limits<double>::max());
      std::vector<double> *new_horizon=new std::vector<double>(
          reference.size(), 
          std::numeric_limits<double>::max());
      (*old_horizon)[0]=0;
      for(index_t i=0; i<query.size(); ++i) {
        for(size_t j=0; j<reference.size(); ++j) {
          size_t ref_j=j;
          size_t jj=std::min(std::max(size_t(0), size_t(j+1)), reference.size());
          double cost=DynamicTimeWarping::ComputeBase(query[i], reference[ref_j]);
          double gain=std::min(
              //dtw_matrix[i-1][j],
              (*old_horizon)[jj],
              std::min(
                //dtw_matrix[i][j-1],
                (*new_horizon)[jj-1],
                //dtw_matrix[i-1][j-1]
                (*old_horizon)[jj-1]
                ));
          (*new_horizon)[jj]=cost+gain;
        }
        std::swap(new_horizon, old_horizon);
        std::fill(new_horizon->begin(), new_horizon->end(), 
            std::numeric_limits<double>::max());
      }
      double result=old_horizon->back();
      delete new_horizon;
      delete old_horizon;
      return result; 
    }

    template<typename PointType>
    double DynamicTimeWarping::Compute(const PointType &query,
        const PointType &reference, index_t q_index, index_t r_index) {
      if (q_index==query.size() && r_index==reference.size()) {
        return 0;
      } 
      if (q_index==query.size() || r_index==reference.size()) {
        return std::numeric_limits<double>::max();
      }
      return ComputeBase(query[q_index], reference[r_index])+
        std::min(Compute(query, reference,q_index, r_index+1),
                 std::min(Compute(query, reference,q_index+1, r_index),
                          Compute(query, reference,q_index+1, r_index+1)));
    
    }
   
    template<typename PointType>
    double DynamicTimeWarping::Compute(
        const PointType &query,
        const PointType &reference) {

      std::vector<std::vector<double> > dtw_matrix;
      if (query.size()==0 || reference.size()==0) {
        return std::numeric_limits<double>::max();
      }
      std::vector<double> *old_horizon=new std::vector<double>(
          reference.size(), 
          std::numeric_limits<double>::max());
      std::vector<double> *new_horizon=new std::vector<double>(
          reference.size(), 
          std::numeric_limits<double>::max());
      (*old_horizon)[0]=0;
      typename PointType::iterator it1=query.begin();
      typename PointType::iterator it2=reference.begin();
      for(index_t i=0; i<query.size(); ++i) {
        for(size_t j=0; j<reference.size(); ++j) {
          size_t jj=std::min(std::max(size_t(0), size_t(j+1)), 
              reference.size());
          double cost=DynamicTimeWarping::ComputeBase(it1.value(), 
              it2.value());
          ++it2;
          double gain=std::min(
              //dtw_matrix[i-1][j],
              (*old_horizon)[jj],
              std::min(
                //dtw_matrix[i][j-1],
                (*new_horizon)[jj-1],
                //dtw_matrix[i-1][j-1]
                (*old_horizon)[jj-1]
                ));
          (*new_horizon)[jj]=cost+gain;
        }
        std::swap(new_horizon, old_horizon);
        std::fill(new_horizon->begin(), new_horizon->end(), 
            std::numeric_limits<double>::max());
      }
      ++it1;
      double result=old_horizon->back();
      delete new_horizon;
      delete old_horizon;
      return result; 
    }
  
    template<typename IteratorType1, typename IteratorType2>
    double DynamicTimeWarping::Compute(
         IteratorType1 &query,
         IteratorType2 &reference, 
         IteratorType1 &q_end,
         IteratorType2 &r_end) {
      if (query==q_end && reference==r_end) {
        return 0;
      } 
      if (query==q_end || reference==r_end) {
        return std::numeric_limits<double>::max();
      }
      double val1=ComputeBase(query.value(), reference.value());
      IteratorType2 reference_plus_one=reference; ++reference_plus_one;
      IteratorType1 query_plus_one=query; ++query_plus_one;
      double val2=Compute(query, reference_plus_one, q_end, r_end);
      double val3=Compute(query_plus_one, reference, q_end, r_end);
      double val4=Compute(query_plus_one, reference_plus_one, q_end, r_end);
      return val1+std::min(val2,std::min(val3,val4));
    }
   
    template<typename PrecisionType1, typename PrecisionType2>
    double DynamicTimeWarping::ComputeBase(const PrecisionType1 &x, const PrecisionType2 &y) {
      return fl::math::Pow<double, 2,1>(x-y);
    }


    double ConstrainedDynamicTimeWarping::Compute(
        std::vector<double> &query,
        std::vector<double> &reference, 
        index_t horizon) {
      if (horizon<=0) {
        fl::logger->Die() << "horizon must be greater than zero";
      }
        
      std::vector<std::vector<double> > dtw_matrix;
      if (query.size()==0 || reference.size()==0) {
        return std::numeric_limits<double>::max();
      }
      horizon=std::max(horizon, index_t(abs((index_t)query.size()-(index_t)reference.size())));
      std::vector<double> *old_horizon=new std::vector<double>(
          2*horizon, 
          std::numeric_limits<double>::max());
      std::vector<double> *new_horizon=new std::vector<double>(
          std::max(2*horizon, index_t(1)), 
          std::numeric_limits<double>::max());
      (*old_horizon)[0]=0;
      for(index_t i=0; i<query.size(); ++i) {
        for(int j=-horizon+1; j<horizon; ++j) {
          size_t ref_j=std::min(std::max(index_t(0), index_t(i+j)), 
             index_t(reference.size()-1));
          size_t jj=j+horizon;
          double cost=DynamicTimeWarping::ComputeBase(query[i], reference[ref_j]);
          double gain=std::min(
              //dtw_matrix[i-1][j],
              (*old_horizon)[jj],
              std::min(
                //dtw_matrix[i][j-1],
                (*new_horizon)[jj-1],
                //dtw_matrix[i-1][j-1]
                (*old_horizon)[jj-1]
                ));
          (*new_horizon)[jj]=cost+gain;
        }
        std::swap(new_horizon, old_horizon);
        std::fill(new_horizon->begin(), new_horizon->end(), 
            std::numeric_limits<double>::max());
      }
      double result=old_horizon->back();
      delete new_horizon;
      delete old_horizon;
      return result; 
    }
  
    double ConstrainedDynamicTimeWarping::Compute(
        std::vector<double> &query,
        std::vector<double> &reference, 
        index_t q_index, 
        index_t r_index,
        index_t horizon) {
   
      if (q_index==query.size() && r_index==reference.size()) {
        return 0;
      } 
      if (q_index==query.size() || r_index==reference.size()) {
        return std::numeric_limits<double>::max();
      } 
     
      if (abs(q_index-r_index)>horizon) {
        return std::numeric_limits<double>::max();
      } else {
        double val0=DynamicTimeWarping::ComputeBase(query[q_index], reference[r_index]);
        double val1=Compute(query, reference, q_index, r_index+1, horizon);
        double val2=Compute(query, reference, q_index+1, r_index, horizon);
        double val3=Compute(query, reference, q_index+1, r_index+1, horizon);
        return val0+std::min(val1, std::min(val2, val3));
      }
    }
  
    template<typename PointType1, typename PointType2>
    double ConstrainedDynamicTimeWarping::Compute(
        PointType1 &query,
        PointType2 &reference, 
        index_t horizon) {
      std::vector<std::vector<double> > dtw_matrix;
      if (query.size()==0 || reference.size()==0) {
        return std::numeric_limits<double>::max();
      }
      horizon=std::max(horizon, index_t(abs((index_t)query.size()-(index_t)reference.size())));
      std::vector<double> *old_horizon=new std::vector<double>(
          2*horizon, 
          std::numeric_limits<double>::max());
      std::vector<double> *new_horizon=new std::vector<double>(
          std::max(2*horizon, index_t(1)), 
          std::numeric_limits<double>::max());
      (*old_horizon)[0]=0;
      typename PointType1::iterator it1=query.begin();
      typename PointType2::iterator it2=reference.begin();
      for(index_t i=0; i<query.size(); ++i) {
        it2=reference.begin();
        for(index_t k=0; k<i-horizon; ++k) {
          ++it2;
        }
        for(int j=-horizon+1; j<horizon; ++j) {
          if ((i+j)>0 && (i+j)<reference.size()) {
            ++it2;
          }
          // size_t ref_j=std::min(std::max(index_t(0), index_t(i+j)), 
          //   index_t(reference.size()-1));

          size_t jj=j+horizon;
          double cost=DynamicTimeWarping::ComputeBase(it1.value(), 
              it2.value());
          double gain=std::min(
              //dtw_matrix[i-1][j],
              (*old_horizon)[jj],
              std::min(
                //dtw_matrix[i][j-1],
                (*new_horizon)[jj-1],
                //dtw_matrix[i-1][j-1]
                (*old_horizon)[jj-1]
                ));
          (*new_horizon)[jj]=cost+gain;
        }
        std::swap(new_horizon, old_horizon);
        std::fill(new_horizon->begin(), new_horizon->end(), 
            std::numeric_limits<double>::max());
        ++it1;
      }
      double result=old_horizon->back();
      delete new_horizon;
      delete old_horizon;
      return result; 
   
    }
  
    template<typename IteratorType1, typename IteratorType2>
    double ConstrainedDynamicTimeWarping::Compute(
         IteratorType1 &query,
         IteratorType2 &reference, 
         IteratorType1 &q_end, 
         IteratorType2 &r_end,
        index_t horizon) {
    
      if (!(query!=q_end) && !(reference!=r_end)) {
        return 0;
      } 
      if (!(query!=q_end) || !(reference!=r_end)) {
        return std::numeric_limits<double>::max();
      }
      double val1=DynamicTimeWarping::ComputeBase(query.value(), reference.value());
       IteratorType2 reference_plus_one=reference; ++reference_plus_one;
       IteratorType1 query_plus_one=query; ++query_plus_one;
      double val2=Compute(query, reference_plus_one, q_end, r_end, horizon);
      double val3=Compute(query_plus_one, reference, q_end, r_end, horizon);
      double val4=Compute(query_plus_one, reference_plus_one, q_end, r_end, horizon);
      return abs(query.attribute()-reference.attribute())<=horizon?(
            val1+std::min(val2,std::min(val3,val4))):
             std::numeric_limits<double>::max();
   
    }
 

  
    template<typename PointType1, typename PointType2>
    double ScalingTimeWarping::Compute(PointType1 &query, 
        PointType2 &reference,
        double scaling_factor, double horizon) {
      index_t m=Size(query);
      index_t n=Size(reference);
      index_t low=index_t(1.0*m/scaling_factor);
      index_t high=std::min(index_t(1.0*scaling_factor*m),n);
      double min_dist=std::numeric_limits<double>::max();
      for(index_t q=low; q<=high; ++q) {
        typename dtw_private::ScaledType<PointType2>::type scaled_reference;
        UniformScale(reference, q, m, &scaled_reference);
        double distance=ConstrainedDynamicTimeWarping::Compute(
            scaled_reference, query, 
            horizon);
        if (distance<min_dist) {
          min_dist=distance;
        }
      }
      return min_dist;
    }
  
    index_t ScalingTimeWarping::Size(const std::vector<double> &point) {
      return point.size();
    }
  
    template<typename PointType>
    index_t ScalingTimeWarping::Size(const PointType &point) {
      return point.nnz();
    }
  
    void ScalingTimeWarping::UniformScale(const std::vector<double> &point1, 
        index_t prefix_size,
        index_t new_size,
        std::vector<double> *point2) {
      
      point2->resize(new_size);
      for(index_t i=0; i<new_size; ++i) {
        (*point2)[i]=point1[index_t(i*prefix_size/new_size)];
      }
    }
  
    template<typename PointType1>
    void ScalingTimeWarping::UniformScale(const PointType1 &point1, 
        index_t prefix_size,
        index_t new_size,
        fl::data::MonolithicPoint<double> *point2) {
      
      point2->Init(point1.size());
      for(index_t i=0; i<new_size; ++i) {
        (*point2)[i]=point1[index_t(i*prefix_size/new_size)];
      }
    }
  
    template<typename AggregatorType>
    void ScalingTimeWarping::BoundScalingTimeWarping(std::vector<double> &point,
        double scaling_factor, index_t horizon, 
        const AggregatorType &agg,
        std::vector<double> *bound_point) {
      bound_point->resize(point.size());
      for(index_t i=0; i<bound_point->size(); ++i) {
        (*bound_point)[i]=point[i];
        for(index_t ind=std::max(index_t(1), index_t(1.0*i/scaling_factor)-horizon); 
            ind<std::min(size_t(i*scaling_factor+horizon), bound_point->size()); ++ind) {
          if (agg(point[ind], (*bound_point)[i])) {
            (*bound_point)[i]=point[ind];
          }
        }
      } 
    }
  
    template<typename PointType, typename AggregatorType>
    void ScalingTimeWarping::BoundScalingTimeWarping(PointType &point,
        double scaling_factor, index_t horizon, 
        const AggregatorType &agg,
        fl::data::SparsePoint<double> *bound_point) {
      bound_point->Init(point.size());
      std::vector<std::pair<index_t, double> > temp;
      for(typename PointType::iterator it=point.begin();
          it!=point.end(); ++it) {
        double bound=it.value();
        typename PointType::iterator it1;
        for(index_t ind=std::max(index_t(1), index_t(1.0*it.attribute()/scaling_factor)-horizon); 
            ind<std::min(index_t(it.attribute()*scaling_factor+horizon), bound_point->size()); ++ind) {
          it1=it;
          for(index_t ind1=0; ind1<ind; ++ind1) {
            ++it1;
          }
          if (agg(it1.value(), bound)) {
            bound=it1.value();
          }
        }
        temp.push_back(
          std::pair<index_t, double>(it.attribute(), bound));
      } 
      bound_point->Load(temp.begin(), temp.end());
    }
  
    double ScalingTimeWarping::ScalingTimeWarpingLowerBound(std::vector<double> &query_point,
        std::vector<double> &reference_point,
        double scaling_factor, index_t horizon) {
      double lower_bound=0;
      std::vector<double> lower_envelope;
      BoundScalingTimeWarping(reference_point,
         scaling_factor, 
         horizon,  
         std::less<double>(),
         &lower_envelope);
      std::vector<double> upper_envelope;
      BoundScalingTimeWarping(query_point,
         scaling_factor, 
         horizon,  
         std::greater<double>(),
         &upper_envelope);
      size_t i=0;
      for( std::vector<double>::const_iterator it=query_point.begin(); 
          it!=query_point.end(); ++it) {
        if (*it>upper_envelope[i]) {
          lower_bound+=fl::math::Pow<double,2,1>(
              *it-upper_envelope[i]);
        } else {
          if (*it<lower_envelope[i]) {
            lower_bound+=fl::math::Pow<double,2,1>(
                *it-lower_envelope[i]);
          }
        }
        i++;
      }
      return lower_bound;
    }
  
    template<typename PointType>
    double ScalingTimeWarping::ScalingTimeWarpingLowerBound(PointType &query_point,
        PointType &reference_point,
        double scaling_factor, 
        index_t horizon) {
      double lower_bound=0;
      fl::data::SparsePoint<double> lower_envelope;
      BoundScalingTimeWarping(reference_point,
         scaling_factor, 
         horizon,  
         std::less<double>(),
         &lower_envelope);
      fl::data::SparsePoint<double> upper_envelope;
      BoundScalingTimeWarping(query_point,
         scaling_factor, 
         horizon,  
         std::greater<double>(),
         &upper_envelope);
      for(index_t i=0; i<query_point.size(); ++i) {
        if (query_point[i]>upper_envelope[i]) {
          lower_bound+=fl::math::Pow<double,2,1>(
              query_point[i]-upper_envelope[i]);
        } else {
          if (query_point[i]<lower_envelope[i]) {
            lower_bound+=fl::math::Pow<double,2,1>(
                query_point[i]-lower_envelope[i]);
          }
        }
      }
      return lower_bound;
    }
  }
}


#endif
