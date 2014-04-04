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

#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_LIC_DEV_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_LIC_DEV_H
#include "boost/math/distributions/students_t.hpp"
#include "fastlib/dense/matrix.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/math/fl_math.h"
#include "mlpack/regression/linear_regression_lic_defs.h"

namespace fl {
  namespace ml {
    void LinearRegressionLIC::ComputeQRNNLS(
                     const fl::dense::Matrix<double> &e_mat, 
                     const Vector_t &f_vec,
                     const fl::dense::Matrix<double> &r_mat,
                     const fl::dense::Matrix<double, true> &q_trans_times_f_vec,
                     fl::dense::Matrix<double, true> *x_vec,
                     std::set<index_t> *p_set,
                     ModelStatistics *stats) {
      index_t dimension=std::min(r_mat.n_cols(), r_mat.n_rows());
      Vector_t z_vec;
      z_vec.Init(r_mat.n_cols());
      std::set<index_t> z_set;
      for(index_t i=0; i<dimension; ++i) {
        z_set.insert(i);
      }
      x_vec->Init(r_mat.n_cols());
      x_vec->SetAll(0.0);

      Vector_t w_vec,w_vec1;
      w_vec.Init(r_mat.n_cols());
      w_vec.SetAll(0.0);
      w_vec1.Init(e_mat.n_rows());
      w_vec1.SetAll(0.0);
      while(true) {
/*
        // We comput E^{t}(f-Ex), but we do it in the QR domain, We have
        // R^{t}Q^{t}(f-QRx) -> R^{t}(Q^{t}f-Rx)
        for(index_t i=0; i<r_mat.n_rows(); ++i) {
          w_vec1[i]=q_trans_times_f_vec[i];
          // R is upper triangular  so when we 
          // multiply a vector we don't have to 
          // do it with the zeros
          for(index_t j=i; j<r_mat.n_cols(); ++j) {
            w_vec1[i]-=r_mat.get(i,j)*(*x_vec)[j];
          }
        }
        for(index_t i=0; i<r_mat.n_cols(); ++i) {
          // R is upper triangular  so when we 
          // multiply a vector we don't have to 
          // do it with the zeros
          w_vec[i]=0;
          for(index_t j=0; j<=std::min(i, r_mat.n_rows()-1); ++j) {
            w_vec[i]+=r_mat.get(j,i)*w_vec1[j];
          }
        }
*/        
        for(index_t i=0; i<e_mat.n_rows(); ++i) {
          w_vec1[i]=f_vec[i];
          for(index_t j=0; j<e_mat.n_cols(); ++j) {
            w_vec1[i]-=e_mat.get(i, j)*(*x_vec)[j];
          }
        }
        for(index_t i=0; i<e_mat.n_cols(); ++i) {
          w_vec[i]=0;
          for(index_t j=0; j<e_mat.n_rows(); ++j) {
            w_vec[i]+=e_mat.get(j, i)*w_vec1[j];
          }
        }

        // check if all w[i]<0 for i in z_set
        bool all_negative=true;
        for(std::set<index_t>::const_iterator it=z_set.begin();
            it!=z_set.end(); ++it) {
          if (w_vec[*it]>0) {
            all_negative=false;
            break;
          }
        }

        if (z_set.size()==0 || all_negative) {
          break; 
        }
        
        // find the maximum w by drawing indices from the z_set
        index_t arg_max_w_in_z=*(z_set.begin());
        for(std::set<index_t>::const_iterator it=z_set.begin(); 
            it !=z_set.end(); ++it) {
          if (w_vec[*it]>w_vec[arg_max_w_in_z]) {
            arg_max_w_in_z=*it;
          }
        }
        p_set->insert(arg_max_w_in_z);
        z_set.erase(arg_max_w_in_z);
        // Solve Rz=Q^Tf, only for p indices
        while(true) {
          if (p_set->size()!=0) {
            Solve(r_mat,
                  q_trans_times_f_vec,
                  *p_set, 
                  &z_vec);
          }
          bool all_p_positive=true;
          for(std::set<index_t>::const_iterator it=p_set->begin();
              it!=p_set->end(); ++it) {
            if (z_vec[*it]<=0) {
              all_p_positive=false;
              break;
            }
          }
          if (all_p_positive) {
            for(index_t i=0; i<x_vec->size(); ++i) {
              (*x_vec)[i]=z_vec[i];
            }
            break;
          }
       
          // find the argmin of x/(x-z)  
          double best_ratio=1;
          for(std::set<index_t>::const_iterator it=p_set->begin();
              it!=p_set->end(); ++it) {
            double ratio=(*x_vec)[*it]/
                ((*x_vec)[*it]-z_vec[*it]);        
            if (z_vec[*it]<=0 && ratio<best_ratio) {
              best_ratio=ratio;
            }
          }
          for(index_t i=0; i<x_vec->size(); ++i) {
            (*x_vec)[i]+=best_ratio*(z_vec[i]-(*x_vec)[i]);
          }
          std::set<index_t> p_to_erase;
          for(std::set<index_t>::const_iterator it=p_set->begin();
              it!=p_set->end(); ++it) {
            if (fabs((*x_vec)[*it])<=tolerance_) {
              z_set.insert(*it);
              p_to_erase.insert(*it);
            }
          }
          for(std::set<index_t>::const_iterator it=p_to_erase.begin(); 
              it!=p_to_erase.end(); ++it) {
            p_set->erase(*it);
          }
        }
      }
      if (stats!=NULL) {
        ComputeModelStatistics(
              e_mat,
              f_vec,
              r_mat,
              *x_vec,
              *p_set,
              conf_prob_,
              stats);
      }
    }

    void LinearRegressionLIC::ComputeNNLS(const fl::dense::Matrix<double> &e_mat,
                     const fl::dense::Matrix<double, true> &f_vec,
                     fl::dense::Matrix<double, true> *x_vec,
                     std::set<index_t> *p_set,
                     ModelStatistics *stats) {

      FL_SCOPED_LOG(RegressionNNLS);
      success_t success;
      fl::dense::Matrix<double> q_mat;
      fl::dense::Matrix<double> r_mat;
      fl::dense::ops::QR<fl::la::Init>(e_mat,
         &q_mat,
         &r_mat,
         &success);

      Vector_t q_trans_times_f;
      q_trans_times_f.Init(q_mat.n_cols());
      q_trans_times_f.SetAll(0.0);
      for(index_t i=0; i<q_mat.n_cols(); ++i) {
        for(index_t j=0; j<q_mat.n_rows(); ++j) {
          q_trans_times_f[i]+=q_mat.get(j, i)*f_vec[j];
        }
      }
      if (success==SUCCESS_FAIL) {
        fl::logger->Die()<<"Something went wrong in the QR decomposition";
      }
      ComputeQRNNLS(e_mat,
                    f_vec,
                    r_mat,
                    q_trans_times_f,
                    x_vec,
                    p_set,
                    stats);
    }

    bool LinearRegressionLIC::ComputeLDP(fl::dense::Matrix<double> &g_mat,
        fl::dense::Matrix<double, true> &h_mat,
        fl::dense::Matrix<double, true> *x_vec,
        std::set<index_t> *p_set,
        ModelStatistics *stats) {
      FL_SCOPED_LOG(RegressionLDP);
      // form the Augmented matrix E=[G h]^T      
      fl::dense::Matrix<double> augmented_matrix;
      augmented_matrix.Init(g_mat.n_cols()+1, g_mat.n_rows());
      for(index_t i=0; i<augmented_matrix.n_cols(); ++i) {
        for(index_t j=0; j<augmented_matrix.n_rows()-1; ++j) {
          augmented_matrix.set(j, i, g_mat.get(i, j));
        }
      }
      for(index_t i=0; i<h_mat.size(); ++i) {
        augmented_matrix.set(augmented_matrix.n_rows()-1, i, h_mat[i]);
      }
      Vector_t rhs;
      rhs.Init(augmented_matrix.n_rows());
      rhs.SetAll(0.0);
      rhs[rhs.size()-1]=1;
      
      success_t success;
      fl::dense::Matrix<double> q_mat;
      fl::dense::Matrix<double> r_mat;
      fl::dense::ops::QR<fl::la::Init>(augmented_matrix,
         &q_mat,
         &r_mat,
         &success);

      Vector_t q_trans_times_rhs;
      q_trans_times_rhs.Init(q_mat.n_cols());
      q_trans_times_rhs.SetAll(0.0);
      for(index_t i=0; i<q_mat.n_cols(); ++i) {
        for(index_t j=0; j<q_mat.n_rows(); ++j) {
          q_trans_times_rhs[i]+=q_mat.get(j, i)*rhs[j];
        }
      }
      if (success==SUCCESS_FAIL) {
        fl::logger->Die()<<"Something went wrong in the QR decomposition";
      }  
      Vector_t u_vec;
      ComputeQRNNLS(augmented_matrix,
                    rhs,
                    r_mat,
                    q_trans_times_rhs,
                    &u_vec, 
                    p_set,
                    NULL);
      Vector_t r_vec;
      r_vec.Init(augmented_matrix.n_rows());
      r_vec.SetAll(0.0);
      double r_vec_norm=0;
      for(index_t i=0; i<augmented_matrix.n_rows(); ++i) {
        r_vec[i]=-rhs[i];
        for(index_t j=0; j<augmented_matrix.n_cols(); ++j) {
          r_vec[i]+=augmented_matrix.get(i, j)*u_vec[j];
        }
        r_vec_norm+=r_vec[i]*r_vec[i];
      }
      if (sqrt(r_vec_norm)<tolerance_) {
        return false;
      }
      x_vec->Init(augmented_matrix.n_rows()-1);
      x_vec->SetAll(0.0);
      for(index_t i=0; i<x_vec->size(); ++i) {
        (*x_vec)[i]=-r_vec[i]/r_vec[r_vec.size()-1];
      }
      if (stats!=NULL) {
        ComputeModelStatistics(
              g_mat,
              h_mat,
              r_mat,
              *x_vec,
              *p_set,
              conf_prob_,
              stats);
      }

      return true;
    }

    bool LinearRegressionLIC::ComputeLSI(fl::dense::Matrix<double> &e_mat,
                   Vector_t &f_vec, 
                   fl::dense::Matrix<double> &g_mat,
                   Vector_t &h_vec,
                   Vector_t *x_vec,
                   std::set<index_t> *p_set, 
                   ModelStatistics *stats) {
      FL_SCOPED_LOG(RegressionLSI);  
      success_t success;
      fl::dense::Matrix<double> q_mat;
      fl::dense::Matrix<double> r_mat;
      fl::dense::ops::QR<fl::la::Init>(e_mat,
         &q_mat,
         &r_mat,
         &success);
      if (success==SUCCESS_FAIL) {
        fl::logger->Die()<<"Something went wrong in the QR decomposition";
      }
      Vector_t q_trans_times_f;
      q_trans_times_f.Init(q_mat.n_cols());
      q_trans_times_f.SetAll(0.0);
      for(index_t j=0; j<q_mat.n_rows(); ++j) {
        for(index_t i=0; i<q_mat.n_cols(); ++i) {
          q_trans_times_f[i]+=q_mat.get(j, i)*f_vec[j];
        } 
      }
      // Computing GR^-1
      fl::dense::Matrix<double> k_mat, k_mat_trans;
      // we need to transpose g_mat
      fl::dense::Matrix<double> g_mat_trans;
      fl::dense::ops::Transpose<fl::la::Init>(g_mat, &g_mat_trans);
      fl::dense::ops::SolveTriangular<fl::la::Init, fl::la::Trans>(
          r_mat, false, g_mat_trans, &k_mat_trans, &success);
      fl::dense::ops::Transpose<fl::la::Init>(k_mat_trans, &k_mat);
      if (success==SUCCESS_FAIL) {
        fl::logger->Die()<<"Numerical errors in linear algebra";
      } 
      // we need to transpose it here
      Vector_t new_rhs;
      new_rhs.Init(k_mat.n_rows());
      new_rhs.SetAll(0.0);
      for(index_t i=0; i<k_mat.n_rows(); ++i) {
          new_rhs[i]=h_vec[i];
        for(index_t j=0; j<k_mat.n_cols(); ++j) {
          new_rhs[i]-=k_mat.get(i, j)*q_trans_times_f[j];
        }
      }
      Vector_t y_vec;
      bool feasible=ComputeLDP(k_mat, new_rhs, &y_vec, p_set, NULL);
      if (feasible==false) {
        return false;
      }
      Vector_t y_vec_plus_q_trans_times_f;
      y_vec_plus_q_trans_times_f.Init(y_vec.size());
      for(index_t i=0; i<y_vec.size(); ++i) {
        y_vec_plus_q_trans_times_f[i]=y_vec[i]+
            q_trans_times_f[i];
      }
      
      std::set<index_t> aux_p_set;
      for(index_t i=0; i<std::min(r_mat.n_rows(), r_mat.n_cols()); ++i) {
        aux_p_set.insert(i);
      }
      Solve(r_mat, y_vec_plus_q_trans_times_f, aux_p_set, x_vec);
      p_set->clear();
      for(index_t i=0; i<x_vec->size(); ++i) {
        if ((*x_vec)[i]!=0) {
          p_set->insert(i);
        }
      }
      if (stats!=NULL) {
        ComputeModelStatistics(
              e_mat,
              f_vec,
              r_mat,
              *x_vec,
              *p_set,
              conf_prob_,
              stats);
      }
      return true;  
    }

    void LinearRegressionLIC::Solve(const fl::dense::Matrix<double> &r_mat,
        const Vector_t &rhs,
        const std::set<index_t> &p_set, 
        Vector_t *z_vec) {

      fl::dense::Matrix<double, false> rr_mat;
      rr_mat.Init(r_mat.n_rows(), p_set.size());
      rr_mat.SetAll(0.0);
      int counter1=0;
      for(index_t i=0; i<rr_mat.n_rows(); ++i) {
        int counter2=0;
        for(std::set<index_t>::const_iterator it2=p_set.begin();
            it2!=p_set.end(); ++it2) { 
          rr_mat.set(i, counter2,  r_mat.get(i, *it2));
          counter2++;
        }
        counter1++;
      }
      Vector_t z_vec_interim;
      success_t success;
      fl::dense::ops::LeastSquareFit<fl::la::Init>(rhs, 
          rr_mat, 
          &z_vec_interim, 
          &success);
      if (success!=SUCCESS_PASS) {
        fl::logger->Die()<<"Linear System Solver failed";
      }
      z_vec->Init(r_mat.n_cols());
      z_vec->SetAll(0.0);
      std::set<index_t>::const_iterator it=p_set.begin();
      for(index_t i=0; i<z_vec_interim.size(); ++i) {
        (*z_vec)[*it]=z_vec_interim[i];
        ++it;
      }

      /*
      z_vec->Init(r_mat.n_cols());
      z_vec->SetAll(0.0);
      for(std::set<index_t>::reverse_iterator it1=p_set.rbegin();
          it1!=p_set.rend(); ++it1) {
        z_vec->operator[](*it1)=rhs[*it1];
        for(std::set<index_t>::reverse_iterator it2=p_set.rbegin(); 
            it2!=it1; ++it2) {
          z_vec->operator[](*it1)-=r_mat.get(*it1, *it2)*(*z_vec)[*it2];
        }
        (*z_vec)[*it1]/=r_mat.get(*it1, *it1);
      } 
      */
    }

    void LinearRegressionLIC::ReverseSolve(const fl::dense::Matrix<double> &r_mat,
        const Vector_t &rhs,
        const std::set<index_t> &p_set, 
        Vector_t *z_vec) {

      fl::dense::Matrix<double, false> rr_mat;
      rr_mat.Init(p_set.size(), p_set.size());
      rr_mat.SetAll(0.0);
      int counter1=0;
      for(index_t i=0; i<rr_mat.n_rows(); ++i) {
        int counter2=0;
        for(std::set<index_t>::const_iterator it2=p_set.begin();
            it2!=p_set.end(); ++it2) { 
          rr_mat.set(i, counter2,  r_mat.get(*it2, i));
          counter2++;
        }
        counter1++;
      }
      Vector_t new_rhs;
      new_rhs.Init(p_set.size());
      counter1=0;
      for(std::set<index_t>::const_iterator it1=p_set.begin();
            it1!=p_set.end(); ++it1) {
        new_rhs[counter1]=rhs[*it1];
        counter1++;
      }
      Vector_t z_vec_interim;
      success_t success;
      fl::dense::ops::LeastSquareFit<fl::la::Init>(new_rhs, 
          rr_mat, 
          &z_vec_interim, 
          &success);
      if (success!=SUCCESS_PASS) {
        fl::logger->Warning()<<"Linear System Solver failed";
      }
      z_vec->Init(r_mat.n_cols());
      z_vec->SetAll(0.0);
      std::set<index_t>::const_iterator it=p_set.begin();
      for(index_t i=0; i<z_vec_interim.size(); ++i) {
        (*z_vec)[*it]=z_vec_interim[i];
        ++it;
      }

      /*
      z_vec->Init(r_mat.n_cols());
      Vector_t x_vec;
      z_vec->SetAll(0.0);
      for(std::set<index_t>::iterator it1=p_set.begin();
          it1!=p_set.end(); ++it1) {
        (*z_vec)[*it1]=rhs[*it1];
        for(std::set<index_t>::iterator it2=p_set.begin(); 
            it2!=it1; ++it2) {
          (*z_vec)[*it1]-=r_mat.get(*it2, *it1)*(*z_vec)[*it2];
        }
        (*z_vec)[*it1]/=r_mat.get(*it1, *it1);
      } 
      */
    }



  void LinearRegressionLIC::ComputeModelStatistics(
              const fl::dense::Matrix<double> &e_mat,
              const Vector_t &targets,
              const fl::dense::Matrix<double> &r_mat,
              const Vector_t &coefficients,
              std::set<index_t> &p_set,
              double conf_prob,
              ModelStatistics *stats) {
      // Declare the student t-distribution and find out the
      // appropriate quantile for the confidence interval
      // (currently hardcoded to 90 % centered confidence).
      boost::math::students_t_distribution<double> distribution(
        coefficients.size());
      // Compute the residual sum of squares
      double residual_sum_of_squares=0;
      for(index_t i=0; i<e_mat.n_rows(); ++i) {
        double predicted_value=0;
        for(std::set<index_t>::const_iterator it=p_set.begin(); 
            it!=p_set.end(); ++it) {
          predicted_value+=coefficients[*it]*e_mat.get(i, *it);
        }        
        residual_sum_of_squares+=fl::math::Pow<double, 2,1>(predicted_value-targets[i]);
      }

      double t_score = quantile(distribution, 0.5 + 0.5 * conf_prob_);
    
      double variance = residual_sum_of_squares /
                        (targets.size() -
                         p_set.size());
    
      // Store the computed standard deviation of the predictions.
      double sigma = sqrt(variance);
      stats->sigma=sigma; 
      Vector_t dummy_vector;
      dummy_vector.Init(index_t(std::min(r_mat.n_cols(), r_mat.n_rows())));
      dummy_vector.SetZero();
      Vector_t first_vector;
      Vector_t second_vector;
      stats->standard_errors.resize(r_mat.n_rows()); 
      stats->t_statistics.resize(r_mat.n_rows()); 
      stats->confidence_interval_los.resize(r_mat.n_rows()); 
      stats->confidence_interval_his.resize(r_mat.n_rows()); 
      stats->p_values.resize(r_mat.n_rows()); 

      for (std::set<index_t>::const_iterator it=p_set.begin(); 
           it!=p_set.end(); ++it) {
        dummy_vector[*it] = 1.0;
        if (*it > 0) {
          std::set<index_t>::iterator dummy_it=it;
          dummy_it--;
          dummy_vector[*(dummy_it)] = 0.0;
        }
        // we have to compute the diagonal elements of (e_mat * emat^{T})^{-1}
        // or (R * R^T)^{-1}
        // This can be achieved by solving
        // R R^{T}x=u
        // R^Tx=y
        // Ry=u
        Solve(r_mat, dummy_vector, p_set, &first_vector);
        ReverseSolve(r_mat, first_vector, p_set, &second_vector);
        stats->standard_errors[*it] = sqrt(variance * second_vector[*it]);
        stats->confidence_interval_los[*it] =
          coefficients[*it] - t_score * stats->standard_errors[*it];
        stats->confidence_interval_his[*it] =
          coefficients[*it] + t_score * stats->standard_errors[*it];
   
        // Compute t-statistics.
        stats->t_statistics[*it] = coefficients[*it] / stats->standard_errors[*it];
        // Compute p-values.
        // Here we take the absolute value of the t-statistics since
        // we want to push all p-values toward the right end.
        double min_t_statistic = std::min(stats->t_statistics[*it],
                                          -(stats->t_statistics[*it]));
        double max_t_statistic = std::max(stats->t_statistics[*it],
                                          -(stats->t_statistics[*it]));

        stats->p_values[*it] =
          1.0 - (cdf(distribution, max_t_statistic) -
                 cdf(distribution, min_t_statistic));
      }
    
      // Compute the r-squared coefficients (normal and adjusted).
      double  r_squared = 0;

      // Compute the average of the observed values.
      double avg_observed_value = 0;
      for (int i = 0; i < targets.size(); ++i) {
        avg_observed_value += targets[i];
      }
      avg_observed_value /= targets.size();

      // Compute something proportional to the variance of the observed
      // values, and the sum of squared residuals of the predictions
      // against the observations.
      double observed_variance = 0;
      for (int i = 0; i < targets.size(); ++i) {
        observed_variance += math::Sqr(targets[i] - avg_observed_value);
      }
      r_squared=(observed_variance - residual_sum_of_squares) / observed_variance;
      stats->r_squared=r_squared;
      // Compute the adjustedSquaredCorrelation Coefficient
      double adjusted_r_squared = 0;
      int num_points = targets.size();
      int num_coefficients = p_set.size();
      double factor = (((double) num_points - 1)) /
                ((double)(num_points - num_coefficients));
      adjusted_r_squared = 1.0 - (1.0 - r_squared) * factor;
      stats->adjusted_r_squared = adjusted_r_squared;
      // Compute the f-statistic between the final refined model and
      // the null model, i.e. the model with all zero coefficients.
      double f_statistic = 0;
      double numerator = r_squared /
                   (p_set.size() - 1.0);
      double denominator = (1.0 - r_squared) /
                     ((double) targets.size() - p_set.size());
      f_statistic=numerator / denominator;
      stats->f_statistic = f_statistic;
      // Compute the AIC score.
      double aic_score = 0;
      aic_score = residual_sum_of_squares;
      aic_score /= ((double) targets.size());
      aic_score = log(aic_score);
      aic_score *= ((double) targets.size());
      aic_score += (2 * p_set.size());
  }



  void LinearRegressionLIC::set_tolerance(double tolerance) {
    tolerance_=tolerance;
  }
  
  void LinearRegressionLIC::set_conf_prob(double conf_prob) {
    conf_prob_=conf_prob;
  }
}}


#endif
