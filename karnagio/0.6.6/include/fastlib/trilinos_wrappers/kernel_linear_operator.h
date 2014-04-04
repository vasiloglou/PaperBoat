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

#ifndef FL_LITE_FASTLIB_TRILINOS_WRAPPERS_KERNEL_LINEAR_OPERATOR_H
#define FL_LITE_FASTLIB_TRILINOS_WRAPPERS_KERNEL_LINEAR_OPERATOR_H

#include "fastlib/math/fl_math.h"
#include "fastlib/trilinos_wrappers/linear_operator.h"

namespace Anasazi {

template<bool dotproduct_selfcase_special>
class DotProductTrait {
  public:
    template<typename KernelType, typename PointType>
    DotProductTrait(const KernelType *kernel,
                    const PointType &first_point,
                    const PointType &second_point,
                    bool point_indices_are_same,
                    double *dotproduct);
};

template<>
class DotProductTrait<true> {
  public:
    template<typename KernelType, typename PointType>
    DotProductTrait(const KernelType *kernel,
                    const PointType &first_point,
                    const PointType &second_point,
                    bool point_indices_are_same,
                    double *dotproduct) {
      *dotproduct = kernel->Dot(first_point, second_point,
                                point_indices_are_same);
    }
};

template<>
class DotProductTrait<false> {
  public:
    template<typename KernelType, typename PointType>
    DotProductTrait(const KernelType *kernel,
                    const PointType &first_point,
                    const PointType &second_point,
                    bool point_indices_are_same,
                    double *dotproduct) {
      *dotproduct = kernel->Dot(first_point, second_point);
    }
};

template < typename TableType, typename KernelType, bool do_centering,
bool dotproduct_selfcase_special >
class KernelLinearOperator: public LinearOperator {

  private:

    TableType *table_;

    const KernelType *kernel_;

    fl::data::MonolithicPoint<double> average_row_;

    double average_;

  public:

    KernelLinearOperator(TableType &table_in,
                         const KernelType &kernel_in,
#ifdef EPETRA_MPI
                         const Epetra_MpiComm &comm_in,
#else
                         const Epetra_SerialComm &comm_in,
#endif
                         const Epetra_Map &map_in) {

      table_ = &table_in;
      kernel_ = &kernel_in;
      comm_ = &comm_in;
      map_ = &map_in;

      if (do_centering) {
        average_row_.Init(table_in.n_entries());
      }
      average_ = 0;

      if (do_centering) {

        // Precompute the average. This is a naive way of computing it.
        for (int i = 0; i < table_in.n_entries(); i++) {
          double average_for_i_th_point = 0;
          for (int j = 0; j < table_in.n_entries(); j++) {
            average_for_i_th_point += kernel_value(i, j);
          }
          average_for_i_th_point /= ((double) table_in.n_entries());
          average_row_[i] = average_for_i_th_point;
        }
        for (int i = 0; i < table_in.n_entries(); i++) {
          average_ += average_row_[i];
        }
        average_ /= ((double) table_in.n_entries());
      }
    }

    double centered_kernel_value(int row, int col) const {
      typename TableType::Dataset_t::Point_t row_point;
      typename TableType::Dataset_t::Point_t col_point;
      table_->get(row, &row_point);
      table_->get(col, &col_point);
      double dotproduct = 0;
      DotProductTrait<dotproduct_selfcase_special>(
        kernel_, row_point, col_point, row == col, &dotproduct);
      return dotproduct - average_row_[row] - average_row_[col] + average_;
    }

    double kernel_value(int row, int col) const {
      typename TableType::Dataset_t::Point_t row_point;
      typename TableType::Dataset_t::Point_t col_point;
      table_->get(row, &row_point);
      table_->get(col, &col_point);
      double dotproduct = 0;
      DotProductTrait<dotproduct_selfcase_special>(
        kernel_, row_point, col_point, row == col, &dotproduct);
      return dotproduct;
    }

    int Apply(
      const Epetra_MultiVector &vecs,
      Epetra_MultiVector &prods) const {

      prods.PutScalar(0);

      for (int j = 0; j < table_->n_entries(); j++) {
        for (int i = 0; i < table_->n_entries(); i++) {

          // This makes sure that the kernel is only computed for a
          // large weight.
          bool computed_kernel_value = false;
          double pair_kernel_value = 0.0;

          for (int k = 0; k < vecs.NumVectors(); k++) {

            // Cache the kernel value.
            if (fabs(vecs.Pointers()[k][j]) >
                std::numeric_limits<double>::min() &&
                computed_kernel_value == false) {

              pair_kernel_value = (do_centering) ?
                                  centered_kernel_value(i, j) :
                                  kernel_value(i, j);
              computed_kernel_value = true;
            }

            prods.Pointers()[k][i] += pair_kernel_value * vecs.Pointers()[k][j];
          }
        }
      }
      return 0;
    }

    int n_rows() const {
      return table_->n_entries();
    }

    int n_cols() const {
      return table_->n_entries();
    }

    double get(int row, int col) const {
      return (do_centering) ? centered_kernel_value(row, col) : kernel_value(row, col);
    }
};

template < typename TableType, typename KernelType, bool do_centering,
bool dotproduct_selfcase_special >
class OperatorTraits < double, Epetra_MultiVector,
    KernelLinearOperator < TableType, KernelType, do_centering,
      dotproduct_selfcase_special > > {
  public:

    static void Apply(const Epetra_Operator& Op,
                      const Epetra_MultiVector& x,
                      Epetra_MultiVector& y) {
      Op.Apply(x, y);
    }
};
};

#endif
