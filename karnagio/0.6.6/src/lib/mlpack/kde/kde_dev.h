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
#ifndef FL_LITE_MLPACK_KDE_KDE_DEV_H
#define FL_LITE_MLPACK_KDE_KDE_DEV_H

#include "mlpack/kde/kde.h"
#include "mlpack/kde/kde_defs.h"

template<typename TemplateMap>
typename fl::ml::Kde<TemplateMap>::Table_t *
fl::ml::Kde<TemplateMap>::query_table() {
  return query_table_;
}

template<typename TemplateMap>
typename fl::ml::Kde<TemplateMap>::Table_t *
fl::ml::Kde<TemplateMap>::reference_table() {
  return reference_table_;
}

template<typename TemplateMap>
typename fl::ml::Kde<TemplateMap>::Global_t &
fl::ml::Kde<TemplateMap>::global() {
  return global_;
}

template<typename TemplateMap>
bool fl::ml::Kde<TemplateMap>::is_monochromatic() const {
  return is_monochromatic_;
}

template<typename TemplateMap>
void fl::ml::Kde<TemplateMap>::Init(
  typename fl::ml::Kde<TemplateMap>::Table_t *reference_table,
  typename fl::ml::Kde<TemplateMap>::Table_t *query_table,
  double bandwidth_in,
  double relative_error_in,
  double probability_in) {

  if (reference_table == NULL) {
    fl::logger->Die()<<"Reference table cannot be NULL";
  }
  reference_table_ = reference_table;
  if (query_table == NULL) {
    is_monochromatic_ = true;
    query_table_ = reference_table;
  }
  else {
    is_monochromatic_ = false;
    query_table_ = query_table;
  }

  // Declare the global constants.
  global_.Init(reference_table_, query_table_, NULL, bandwidth_in, is_monochromatic_,
               relative_error_in, probability_in);
}

template<typename DensityTableType, typename QueryLabelsTableType>
void fl::ml::Kde<boost::mpl::void_>::ComputeScoresForAuc(
    const index_t references_size,
    const index_t queries_n_entries,
    const index_t auc_label,
    boost::shared_ptr<QueryLabelsTableType> &query_labels,
    int32_t overall_label,
    std::vector<DensityTableType> &densities,
    const std::vector<double> &priors,
    bool auc,
    bool compute_score,
    std::vector<double> *partial_scores,
    std::vector<index_t> *points_per_class,
    std::vector<double> *a_class_scores,
    std::vector<double> *b_class_scores,
    std::vector<index_t> *winning_classes,
    double *total_accuracy) {

  winning_classes->resize(queries_n_entries);
  std::fill(winning_classes->begin(), winning_classes->end(), 0);
  partial_scores->resize(references_size);
  std::fill(partial_scores->begin(), partial_scores->end(), 0);
  points_per_class->resize(references_size);
  std::fill(points_per_class->begin(), points_per_class->end(),0);
  for (index_t i = 0; i < queries_n_entries; ++i) {
    double score = -std::numeric_limits<double>::max();
    index_t winner = -1;
    double a_score=-std::numeric_limits<double>::max();
    double b_score=-std::numeric_limits<double>::max();
    int label=-1;
    if (compute_score==true) {
      if (query_labels.get()!=NULL) {
        typename QueryLabelsTableType::Point_t point;
        query_labels->get(i, &point);
        label=point[0];
      } else {
        label=overall_label;
      }
    } else {
      
    }
    
    double auc_kde_score=densities[auc_label].densities_[i]*priors[auc_label];
    for (index_t j = 0; j < densities.size(); j++) {
      double this_score = densities[j].densities_[i] * priors[j];
      if (this_score<0) {
        fl::logger->Warning()<<"Detected negative density value!!"<<std::endl;
      }
      if (boost::math::isnan(this_score)==true) {
        fl::logger->Warning()<<"Detected nan density value!!"<<std::endl;
      }
      if (boost::math::isinf(this_score)==true) {
        fl::logger->Warning()<<"Detected inf density value!!"<<std::endl;
      }
      if (this_score > score) {
        score = this_score;
        winner = j;
      } else {
        // We have to split ties equally so that we don't have artifacts
        if (this_score==score) {
          if (fl::math::Random(0.0, 1.0)<0.5) {
            score = this_score;
            winner = j;
          }
        }
      }
      if (auc==true && compute_score==true) {
        if(j==auc_label) {
          continue;
        }
        if (label==auc_label) {
          a_score=std::max(this_score, a_score);
        } else {
          b_score=std::max(this_score, b_score);
        }
      }
    }
    if (label==auc_label) {
      a_score=auc_kde_score-a_score;
    } else {
      b_score=auc_kde_score-b_score;
    }
    (*winning_classes)[i]=winner;
    if (compute_score==true) {
      (*points_per_class)[label]+=1;
      if (winner==label) {
        (*total_accuracy)+=1;
        (*partial_scores)[label]+=1;
      }
    }
    if (compute_score==true && auc==true) {
      if (label==auc_label) {
        a_class_scores->push_back(a_score);
      } else {
        b_class_scores->push_back(b_score);
      }
    }
  }
}

template<typename TemplateMap>
void fl::ml::Kde<TemplateMap>::set_bandwidth(double bandwidth_in) {
  global_.set_bandwidth(bandwidth_in);
}


#endif
