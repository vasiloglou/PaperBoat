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

#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_LIC_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_LIC_H
#include <set>
#include "fastlib/dense/matrix.h"
#include "model_statistics.h"

namespace fl {namespace ml {
  /**
   * @brief This class solves the problem of 
   * Minimize ||Ex-f|| subject to x >= 0
   */
  class LinearRegressionLIC {
    public:
       typedef fl::dense::Matrix<double, true> Vector_t;
      /**
       * @brief Computes the non-negative linear least squares
       *  min ||Ex-f|| subject to x>=0
       */
      void ComputeNNLS(const fl::dense::Matrix<double> &e_mat,
                       const fl::dense::Matrix<double, true> &f_vec,
                       fl::dense::Matrix<double, true> *x_vec,
                       std::set<index_t> *p_set,
                       ModelStatistics *stats);
      /**
       * @brief Computes the non-negative linear least squares
       *  min ||Ex-f|| subject to x>=0, but instead it takes 
       *  as an input the QR factorization so it actually solves the problem
       *  min ||Rx-Q^Tf|| subject to x>=0
       */
      void ComputeQRNNLS(const fl::dense::Matrix<double> &e_mat,
                         const fl::dense::Matrix<double, true> &f_vec,
                         const fl::dense::Matrix<double> &r_mat,
                         const fl::dense::Matrix<double, true> &q_trans_times_f_vec,
                         fl::dense::Matrix<double, true> *x_vec,
                         std::set<index_t> *p_set,
                         ModelStatistics *stats);
      /**
       * @brief the LDP problem
       *  min ||x|| subject to Gx>=h
       */ 
      bool ComputeLDP(fl::dense::Matrix<double> &g_mat,
          fl::dense::Matrix<double, true> &h_mat,
          fl::dense::Matrix<double, true> *x_vec,
          std::set<index_t> *p_set,
          ModelStatistics *stats);

      /**
       * @brief Solve
       *  min ||Ex-f|| subject to Gx>=h
       */
      bool ComputeLSI(fl::dense::Matrix<double> &e_mat,
                     Vector_t &f_vec, 
                     fl::dense::Matrix<double> &g_mat,
                     Vector_t &h_vec,
                     Vector_t *x_vec,
                     std::set<index_t> *p_set, 
                     ModelStatistics *stats);


      /**
       *  @brief Solves an upper diagonal system
       */
      void Solve(const fl::dense::Matrix<double> &r_mat,
          const Vector_t &rhs,
          const std::set<index_t> &p_set, 
          Vector_t *z_vec);

      /**
       *  @brief It solves R^T z=rhs, where R is upper triangular
       *      
       *             
       */
      void ReverseSolve(const fl::dense::Matrix<double> &r_mat,
          const Vector_t &rhs,
          const std::set<index_t> &p_set, 
          Vector_t *z_vec);


    template<typename TableType>
    void Predict(const Vector_t &coefficients,
                 const TableType &query_table,
                 std::vector<double> *result) const ; 

    template<typename PointType>
    void Predict(const Vector_t &coefficients, 
                 const PointType &point, 
                 double *result) const ;

    void ComputeModelStatistics(
                const fl::dense::Matrix<double> &e_mat,
                const Vector_t &f_vec,
                const fl::dense::Matrix<double> &r_mat,
                const Vector_t &coefficients,
                std::set<index_t> &p_set,
                double conf_prob,
                ModelStatistics *stats);

    void set_tolerance(double tolerance);
    
    void set_conf_prob(double conf_prob);

    private:
      double tolerance_;
      double conf_prob_;
  };

}}


#endif
