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
#ifndef FL_LITE_FASTLIB_OPTIMIZATION_LBFGS_OPTIMIZATION_UTILS_H_
#define FL_LITE_FASTLIB_OPTIMIZATION_LBFGS_OPTIMIZATION_UTILS_H_

#include "fastlib/base/base.h"
#include "fastlib/math/fl_math.h"
#include "fastlib/dense/linear_algebra.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/dense/matrix.h"
namespace fl {
namespace optim {
class OptUtils {
  public:
    template<typename ContainerType>
    static void RemoveMean(ContainerType *data) {
      index_t dimension = data->n_rows();
      index_t num_of_points = data->n_cols();
      ContainerType mean;
      mean.Init(dimension);
      mean.SetAll(0);
      ContainerType col;
      for (index_t i = 0; i < num_of_points; i++) {
        data->MakeColumnVector(i, &col);
        fl::la::AddTo(col, &mean);
      }
      fl::la::SelfScale(-1.0 / num_of_points, &mean);
      for (index_t i = 0; i < num_of_points; i++) {
        data->MakeColumnVector(i, &col);
        fl::la::AddTo(mean, &col);
      }
    }

    template<typename ContainerType>
    static void NonNegativeProjection(ContainerType *data) {
      typename ContainerType::CalcPrecision_t *ptr = data->ptr();
      for (index_t i = 0; i < (index_t)data->n_elements(); i++) {
        if (ptr[i] < 0) {
          ptr[i] = 0;
        }
      }
    }

    template<typename ContainerType>
    static void BoundProjection(ContainerType *data,
                                typename ContainerType::CalcPrecision_t lo,
                                typename ContainerType::CalcPrecision_t hi) {
      for (index_t i = 0; i < (index_t)data->n_elements(); i++) {
        if ((*data)[i] > hi) {
          (*data)[i] = hi;
          continue;
        }
        if ((*data)[i] < lo) {
          (*data)[i] = lo;
        }
      }
    }

    template<typename PrecisionType>
    static success_t SVDTransform(fl::dense::Matrix<PrecisionType> &input_mat,
                                  fl::dense::Matrix<PrecisionType> *output_mat,
                                  index_t components_to_keep) {
      fl::dense::Matrix<PrecisionType> temp;
      temp.Copy(input_mat);
      RemoveMean(&temp);
      fl::dense::Matrix<PrecisionType> s;
      fl::dense::Matrix<PrecisionType> U, VT;
      success_t success;
      fl::dense::ops::SVD<fl::la::Init>(temp, &s, &U, &VT, &success);
      if (success == SUCCESS_PASS) {
        fl::logger->Message()<<"PCA successful !! Printing requested i"
          <<  components_to_keep << " eigenvalues..."<<std::endl;
             
        PrecisionType energy_kept = 0;
        PrecisionType total_energy = 0;
        for (index_t i = 0; i < components_to_keep; i++) {
          energy_kept += s[i];
        }
        printf("\n");
        for (index_t i = 0; i < s.length(); i++) {
          total_energy += s[i];
        }
        fl::logger->Message()<<"Kept "
          << energy_kept*100 / total_energy <<"%% of the energy"<<std::endl; 
      }

      fl::dense::Matrix<PrecisionType> s_chopped;
      s.MakeSubvector(0, components_to_keep, &s_chopped);
      fl::dense::Matrix<PrecisionType> temp_VT(components_to_keep, VT.n_cols());
      for (index_t i = 0; i < temp_VT.n_cols(); i++) {
        memcpy(temp_VT.GetColumnPtr(i),
               VT.GetColumnPtr(i), components_to_keep*sizeof(double));
      }

      fl::la::ScaleRows(s_chopped, &temp_VT);
      output_mat->Own(&temp_VT);
      return success;
    }

    template<typename PrecisionType>
    static void SparseProjection(fl::dense::Matrix<PrecisionType> *data,
                                 PrecisionType sparse_factor) {
      DEBUG_ASSERT(sparse_factor <= 1);
      DEBUG_ASSERT(sparse_factor >= 0);
      index_t dimension = data->n_rows();
      std::vector<index_t> zero_coeff;
      fl::dense::Matrix<PrecisionType> w_vector;
      w_vector.Init(dimension);
      fl::dense::Matrix<PrecisionType> v_vector;
      v_vector.Init(dimension);
      fl::dense::Matrix<PrecisionType> a_vector;
      fl::dense::Matrix<PrecisionType> ones;
      ones.Init(dimension);
      ones.SetAll(1.0);
      // This part of the sparsity constraint function formula can be
      // precomputed and it is the same for every iteration
      PrecisionType precomputed_sparse_factor = -sparse_factor * (
            fl::math::Pow<PrecisionType, 1, 2>(dimension) - 1) +
          fl::math::Pow<PrecisionType, 1, 2>(dimension);

      for (index_t i = 0; i < data->n_cols(); i++) {
        PrecisionType *point = data->GetColumnPtr(i);
        PrecisionType l2_norm = fl::la::LMetric<2>(dimension,
                                point, point);
        PrecisionType l1_norm = precomputed_sparse_factor * l2_norm;
        // (L1-\sum x_i)/dimension
        PrecisionType factor1 = l1_norm;
        for (index_t j = 0; j < dimension; j++) {
          factor1 -= point[j];
        }
        factor1 /= dimension;
        for (index_t j = 0; j < dimension; j++) {
          v_vector[j] += factor1;
        }
        zero_coeff.clear();
        fl::dense::Matrix<PrecisionType> midpoint;
        midpoint.Init(dimension);
        while (true) {
          midpoint.SetAll(l1_norm / (dimension - zero_coeff.size()));
          for (index_t j = 0; j < zero_coeff.size(); j++) {
            midpoint[zero_coeff[j]] = 0.0;
          }
          fl::la::Sub<fl::la::Overwrite>(midpoint, v_vector, &w_vector);
          PrecisionType w_norm = fl::la::LengthEuclidean(w_vector);
          PrecisionType w_times_v = 2 * fl::la::Dot(v_vector, w_vector);
          PrecisionType v_norm_minus_l2 = fl::la::LengthEuclidean(v_vector) - l2_norm;
          PrecisionType alpha = (-w_times_v + fl::math::Pow<1, 2>(w_times_v * w_times_v
                                 - 4 * w_norm * v_norm_minus_l2)) / (2 * w_norm);
          fl::la::AddExpert(alpha, w_vector, &v_vector);
          bool all_positive = true;
          zero_coeff.clear();
          fl::dense::Matrix<PrecisionType> v_sum = 0;
          for (index_t j = 0; j < dimension; j++) {
            if (v_vector[j] < 0) {
              all_positive = false;
              zero_coeff.push_back(j);
              v_vector[j] = 0;
            }
            else {
              v_sum += v_vector[j];
            }
          }
          if (all_positive == true) {
            break;
          }
          PrecisionType temp = (l1_norm - v_sum) / (dimension - zero_coeff.size());
          fl::la::AddExpert(temp, ones, &v_vector);
          for (index_t j = 0; j < zero_coeff.size(); j++) {
            v_vector[zero_coeff[j]] = 0;
          }
        }
      }
    }

};
}
} // namespaces
#endif // OPTIMIZATION_UTILS_H_

