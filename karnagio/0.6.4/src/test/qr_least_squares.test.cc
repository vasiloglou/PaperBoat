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

// for BOOST testing
#define BOOST_TEST_MAIN

#include "boost/test/unit_test.hpp"
#include "fastlib/data/monolithic_point.h"
#include "fastlib/dense/matrix.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/math/fl_math.h"
#include "fastlib/table/branch_on_table_dev.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/table/default/dense/unlabeled/balltree/table.h"
#include "fastlib/table/table_dev.h"
#include "mlpack/regression/qr_least_squares_dev.h"
#include <time.h>

class TestQRLeastSquares {

  private:

    typedef fl::table::dense::unlabeled::balltree::Table TableType;

  private:

    void CompareResidualNorm_(
      const fl::dense::Matrix<double, false> &left_hand_side,
      const fl::dense::Matrix<double, false> &right_hand_side,
      const fl::dense::Matrix<double, true> &naive_solution,
      const fl::dense::Matrix<double, true> &incremental_solution) {

      // Right hand side alias.
      fl::dense::Matrix<double, true> right_hand_side_alias;
      right_hand_side.MakeColumnVector(0, &right_hand_side_alias);

      // Compute the residual for the naive solution.
      fl::dense::Matrix<double, true> naive_residual;
      fl::dense::Matrix<double, true> naive_product;
      fl::dense::ops::Mul<fl::la::Init>(left_hand_side, naive_solution,
                                        &naive_product);
      fl::dense::ops::Sub<fl::la::Init>(right_hand_side_alias, naive_product,
                                        &naive_residual);

      // Compute the residual for the incremental solution.
      fl::dense::Matrix<double, true> incremental_residual;
      fl::dense::Matrix<double, true> incremental_product;
      fl::dense::ops::Mul<fl::la::Init>(left_hand_side, incremental_solution,
                                        &incremental_product);
      fl::dense::ops::Sub<fl::la::Init>(right_hand_side_alias,
                                        incremental_product,
                                        &incremental_residual);

      // Compare the residual.
      double residual_norm_diff_squared = 0.0;
      for (int i = 0; i < incremental_residual.length(); i++) {
        residual_norm_diff_squared +=
          fl::math::Sqr(incremental_residual[i] - naive_residual[i]);
      }
      fl::logger->Message() << "Residual difference " <<
      residual_norm_diff_squared;
    }

    void BuildTable_(
      const fl::dense::Matrix<double, false> &random_left_hand_side,
      const fl::dense::Matrix<double, false> &random_right_hand_side,
      TableType *table,
      std::deque<int> *active_column_indices) {

      // Each row of random_left_hand_side is a point, and we are
      // transposing each of them, with the last element of each point
      // being the right hand side value.
      table->Init(std::vector<int>(1, random_left_hand_side.n_cols() + 1),
                  std::vector<int>(), random_left_hand_side.n_rows());
      for (int i = 0; i < random_left_hand_side.n_rows(); i++) {
        TableType::Dataset_t::Point_t point;
        table->get(i, &point);
        for (int j = 0; j < random_left_hand_side.n_cols(); j++) {
          point[j] = random_left_hand_side.get(i, j);
        }
        point[ point.length() - 1 ] = random_right_hand_side.get(i, 0);
      }

      // Each column index is an active set.
      active_column_indices->resize(0);
      for (int i = 0; i < random_left_hand_side.n_cols(); i++) {
        active_column_indices->push_back(i);
      }
    }

    void NaiveLeastSquares_(
      const fl::dense::Matrix<double, false> &left_hand_side,
      const fl::dense::Matrix<double, false> &right_hand_side,
      fl::dense::Matrix<double, true> *solution) {

      fl::dense::Matrix<double, false> q_factor, r_factor;
      success_t success_flag;
      fl::dense::ops::QR<fl::la::Init>(left_hand_side, &q_factor, &r_factor,
                                       &success_flag);

      solution->Init(left_hand_side.n_cols());
      solution->SetZero();

      // Triangular solve.
      fl::dense::Matrix<double> transformed_right_hand_side;
      fl::dense::ops::Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>(
        q_factor, right_hand_side, &transformed_right_hand_side);

      for (int j = solution->length() - 1; j >= 0; j--) {
        if (j >= r_factor.n_rows()) {
          continue;
        }
        double temp = transformed_right_hand_side[j];
        for (int i = solution->length() - 1; i > j; i--) {
          temp -= r_factor.get(j, i) * ((*solution)[i]);
        }
        if (fabs(r_factor.get(j, j)) >
            std::numeric_limits<double>::min()) {
          (*solution)[j] = temp / r_factor.get(j, j);
        }
        else {
          (*solution)[j] = 0;
        }
      }
    }

    void RandomLinearSystem_(
      fl::dense::Matrix<double, false> *random_left_hand_side,
      fl::dense::Matrix<double, false> *random_right_hand_side) {

      int random_num_rows = fl::math::Random(20, 60);
      int random_num_cols = fl::math::Random(10, 80);

      fl::logger->Message() << "Testing on a random linear system " <<
      random_num_rows << " by " << random_num_cols;
      random_left_hand_side->Init(random_num_rows, random_num_cols);

      for (int i = 0; i < random_num_cols; i++) {
        for (int j = 0; j < random_num_rows; j++) {
          random_left_hand_side->set(j, i, fl::math::Random<double>(0, 2));
        }
      }
      random_right_hand_side->Init(random_num_rows, 1);
      for (int i = 0; i < random_num_rows; i++) {
        random_right_hand_side->set(i, 0, fl::math::Random<double>(0.0, 5.0));
      }
    }

  public:
    void Trial() {
      for (int i = 0; i < 100; i++) {
        fl::dense::Matrix<double, false> random_left_hand_side;
        fl::dense::Matrix<double, false> random_right_hand_side;
        fl::dense::Matrix<double, true> naive_solution;
        RandomLinearSystem_(&random_left_hand_side, &random_right_hand_side);

        fl::logger->Message() << "Trial: " << i;
        fl::logger->Message() << "Solving " << random_left_hand_side.n_rows() <<
        " by " << random_left_hand_side.n_cols() << " system.";

        // Compute naively.
        NaiveLeastSquares_(random_left_hand_side, random_right_hand_side,
                           &naive_solution);

        // Compute in an incremental fashion.
        fl::dense::Matrix<double, true> incremental_solution;
        std::deque<int> initial_active_column_indices;
        TableType table;
        BuildTable_(random_left_hand_side, random_right_hand_side, &table,
                    &initial_active_column_indices);
        fl::ml::QRLeastSquares<false> qr_factor;
        qr_factor.Init(table, initial_active_column_indices,
                       table.n_attributes() - 1, false);
        fl::logger->Message() << "R factor has " << qr_factor.n_rows() <<
        " rows";
        incremental_solution.Init(initial_active_column_indices.size());
        qr_factor.Solve((fl::dense::Matrix<double, true> *)NULL,
                        &incremental_solution);

        // Compare the norm difference.
        double squared_norm_diff = 0.0;
        double naive_solution_squared_norm = 0.0;
        for (int i = 0; i < naive_solution.length(); i++) {
          squared_norm_diff += fl::math::Sqr(naive_solution[i] -
                                             incremental_solution[i]);
          naive_solution_squared_norm += fl::math::Sqr(naive_solution[i]);
        }

        fl::logger->Message() << "Squared norm difference between incremental "
        "and naive: " << squared_norm_diff;
        fl::logger->Message() << "Naive solution has a squared norm of " <<
        naive_solution_squared_norm;
        BOOST_ASSERT(squared_norm_diff <= 0.01 * naive_solution_squared_norm);

        CompareResidualNorm_(random_left_hand_side,
                             random_right_hand_side,
                             naive_solution,
                             incremental_solution);
      }
    }
};

BOOST_AUTO_TEST_SUITE(TestSuiteKde)
BOOST_AUTO_TEST_CASE(TestCaseKde) {

  // Set the logger to be verbose.
  fl::logger->SetLogger("verbose");

  srand(time(NULL));
  TestQRLeastSquares test;
  test.Trial();
  fl::logger->Message() << "All tests passed!";
}
BOOST_AUTO_TEST_SUITE_END()
