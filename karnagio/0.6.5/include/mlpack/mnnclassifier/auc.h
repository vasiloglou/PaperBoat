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
#ifndef FL_LITE_MLPACK_CLASSIFIER_AUC_H_
#define FL_LITE_MLPACK_CLASSIFIER_AUC_H_
#include <vector>
#include "fastlib/base/base.h"

namespace fl {
namespace ml {
template < typename ContainerType>
void ComputeAUC(ContainerType &a_scores,
                ContainerType &b_scores, double *auc,
                std::vector<std::pair<double, double> > *roc) {
  BOOST_ASSERT(a_scores.size() > 0);
  BOOST_ASSERT(b_scores.size() > 0);
  *auc = 0;
  std::sort(b_scores.begin(), b_scores.end(),
      std::greater<typename ContainerType::value_type>());
  std::sort(a_scores.begin(), a_scores.end(), 
      std::greater<typename ContainerType::value_type>()); 
  typename ContainerType::iterator it1;
  typename ContainerType::iterator it2;
  typename ContainerType::iterator it=b_scores.begin();
  for (it1 = a_scores.begin(); it1 != a_scores.end(); ++it1) {
    if (*it1 >*it) {
      *auc+=b_scores.end()-it;
      continue;
    }
    for (it2 = it; it2 != b_scores.end(); ++it2) {
      if (*it2<*it1) {
        it=it2;
        *auc += b_scores.end()-it2;
        break;
      }
    }
  }
  *auc /= (a_scores.size() * b_scores.size());
  if (roc!=NULL) {
    double min=a_scores.back();
    double max=a_scores.front();
    int n_chunks=50;
    double chunk=(max-min)/n_chunks;
    for(int i=0; i<n_chunks+1; ++i) {
      it1=std::upper_bound(a_scores.begin(), a_scores.end(), max-i*chunk,
          std::greater<typename ContainerType::value_type>());
      double a=1.0*(it1-a_scores.begin())/a_scores.size();
      it1=std::upper_bound(b_scores.begin(), b_scores.end(), max-i*chunk,
          std::greater<typename ContainerType::value_type>());
      double b=1.0*(it1-b_scores.begin())/b_scores.size();
      roc->push_back(std::make_pair(a,b));
    }
  }
}

template < typename ContainerType>
void ComputeAUC(ContainerType &a_scores,
                ContainerType &b_scores, double *auc) {
  ComputeAUC(a_scores,
             b_scores, 
             auc, 
             NULL); 
} 

}
} // namespaces
#endif
