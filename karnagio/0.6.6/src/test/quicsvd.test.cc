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
/**
 * @file quicsvd_test.cc
 *
 * A "stress" test driver for the QuicSVD.
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include "boost/test/unit_test.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/int.hpp"
#include "mlpack/quicsvd/quicsvd_dev.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/table/file_data_access.h"
#include "fastlib/tree/tree.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/table/table_dev.h"

template<typename Precision, bool sort_points>
class TestQuicSVD {

  private:

    static const Precision acceptance_threshold_;

    static const Precision scaling_perturbation_;

    static const Precision additive_perturbation_;

    static const Precision required_relative_error_;

    static void GenerateRandomVector_
    (fl::dense::Matrix<Precision, true> *previous_column,
     fl::dense::Matrix<Precision, true> *current_column) {

      if (previous_column == NULL) {
        for (index_t i = 0; i < current_column->length(); i++) {
          (*current_column)[i] = fl::math::Random<Precision>(-5.0, 5.0);
        }
      }
      else {

        // The L2 norm of the previous column, used for scaling and
        // slight perturbation.
        Precision scaling_factor =
          fl::math::Random<Precision>(1.0 - scaling_perturbation_,
                                      1.0 + scaling_perturbation_);

        for (index_t i = 0; i < current_column->length(); i++) {

          Precision random_number = fl::math::Random<Precision>(-1.0, 1.0);
          (*current_column)[i] = scaling_factor * ((*previous_column)[i]) +
                                 additive_perturbation_ * random_number;
        }
      }
    }

  public:

    struct MyDatasetArgs : public fl::data::DatasetArgs {
      typedef boost::mpl::vector1<Precision> DenseTypes;
      typedef fl::data::DatasetArgs::Compact StorageType;
      typedef Precision CalcPrecision;
    };

    struct MyTableArgs : public fl::table::TableArgs {
      typedef fl::data::MultiDataset<MyDatasetArgs> DatasetType;
      typedef boost::mpl::bool_<sort_points> SortPoints;
    };

    struct MyTreeArgs : public fl::tree::TreeArgs {
      typedef boost::mpl::bool_<sort_points> SortPoints;
      typedef boost::mpl::bool_<false> StoreLevel;
      typedef fl::tree::SimilarityTree TreeSpecType;
      typedef fl::math::CosinePreMetric MetricType;
      typedef fl::tree::CosineBound<fl::data::MonolithicPoint<Precision> > BoundType;
    };

    struct MyTableMap {
      typedef MyTableArgs TableArgs;
      typedef MyTreeArgs TreeArgs;
    };

    typedef fl::table::Table<MyTableMap> Table_t;
    typedef typename Table_t::Tree_t Tree_t;

    struct QuicSVDArgs1 {
      typedef Table_t TableType;
      static const  bool mean_center = true;
    };
    struct QuicSVDArgs2 {
      typedef Table_t TableType;
      static const  bool mean_center = false;
    };

  public:

    static void Init() {
      srand(time(NULL));
    }

    static void FinalMessage() {
      printf("\nAll tests passed!\n");
    }

    static void PrintTable_(const Table_t &table) {

      for (int i = 0; i < table.n_attributes(); i++) {
        for (int j = 0; j < table.n_entries(); j++) {
          typename Table_t::Point_t point;
          table.get(j, &point);
          fprintf(stderr, "%+3.3f ", point[i]);
        }
        fprintf(stderr, "\n");
      }
    }

    static void PcaTest() {

      printf("PcaTest(): Generating a random three-dimensional dataset with"
             " the two-dimensional dataset embedded in.\n\n");

      // Three-dimensional random dataset.
      index_t num_rows = 3;
      index_t num_cols = fl::math::Random(50, 100);
      Table_t random_dataset;
      random_dataset.Init(std::vector<int>(1, num_rows),
                          std::vector<int>(),
                          num_cols);

      // Generate random points.
      for (index_t c = 0; c < num_cols; c++) {

        typename Table_t::Point_t random_dataset_point;
        random_dataset.get(c, &random_dataset_point);

        // Do a coin-flip to decide whether to put the point on the
        // x-axis or the y-axis.
        index_t random_coin_flip = fl::math::Random(0, 2);
        if (random_coin_flip == 0) {
          random_dataset_point.set(0, fl::math::Random<double>(-5.0, 5.0));
          random_dataset_point.set(1, 0.0);
        }
        else {
          random_dataset_point.set(0, 0.0);
          random_dataset_point.set(1, fl::math::Random<double>(-2.0, 2.0));
        }
        random_dataset_point.set(2, 0.0);
      }

      // Apply a random three-dimensional rotation to the matrix.
      fl::dense::Matrix<Precision, false> random_rotation;
      fl::dense::Matrix<Precision, true> random_vector;
      random_rotation.Init(3, 3);
      random_rotation.SetZero();
      random_vector.Init(3);
      random_vector.SetZero();
      double first_random_number = fl::math::Random<double>(0.0, 1.0);
      double second_random_number = fl::math::Random<double>(0.0, 1.0);
      double third_random_number = fl::math::Random<double>(0.0, 1.0);

      // Computed PI value from arc cosine.
      double pi = acos(-1.0);
      random_rotation.set(0, 0, cos(2 * pi * first_random_number));
      random_rotation.set(0, 1, sin(2 * pi * first_random_number));
      random_rotation.set(1, 0, -sin(2 * pi * first_random_number));
      random_rotation.set(1, 1, cos(2 * pi * first_random_number));
      random_rotation.set(2, 2, 1.0);
      random_vector[0] = cos(2 * pi * second_random_number) *
                         sqrt(third_random_number);
      random_vector[1] = sin(2 * pi * second_random_number) *
                         sqrt(third_random_number);
      random_vector[2] = sqrt(1 - third_random_number);

      // Compute the Householder matrix from the random vector.
      fl::dense::Matrix<Precision, false> householder;
      householder.Init(3, 3);
      householder.SetZero();
      householder.set(0, 0, 1.0);
      householder.set(1, 1, 1.0);
      householder.set(2, 2, 1.0);
      for (index_t j = 0; j < 3; j++) {
        for (index_t i = 0; i < 3; i++) {
          householder.set(i, j, householder.get(i, j) -
                          2.0 * random_vector[i] * random_vector[j]);
        }
      }

      // Apply the rotation to the dataset.
      for (index_t j = 0; j < num_cols; j++) {
        typename Table_t::Point_t random_dataset_point;
        random_dataset.get(j, &random_dataset_point);
        Precision first_coord, second_coord, third_coord;
        first_coord = random_rotation.get(0, 0) *
                      random_dataset_point.get(0) +
                      random_rotation.get(0, 1) *
                      random_dataset_point.get(1) +
                      random_rotation.get(0, 2) *
                      random_dataset_point.get(2);
        second_coord = random_rotation.get(1, 0) *
                       random_dataset_point.get(0) +
                       random_rotation.get(1, 1) *
                       random_dataset_point.get(1) +
                       random_rotation.get(1, 2) *
                       random_dataset_point.get(2);
        third_coord = random_rotation.get(2, 0) *
                      random_dataset_point.get(0) +
                      random_rotation.get(2, 1) *
                      random_dataset_point.get(1) +
                      random_rotation.get(2, 2) *
                      random_dataset_point.get(2);
        random_dataset_point.set(0, first_coord);
        random_dataset_point.set(1, second_coord);
        random_dataset_point.set(2, third_coord);
      }
      for (index_t j = 0; j < num_cols; j++) {
        Precision first_coord, second_coord, third_coord;
        typename Table_t::Point_t random_dataset_point;
        random_dataset.get(j, &random_dataset_point);
        first_coord = householder.get(0, 0) *
                      random_dataset_point.get(0) +
                      householder.get(0, 1) *
                      random_dataset_point.get(1) +
                      householder.get(0, 2) *
                      random_dataset_point.get(2);
        second_coord = householder.get(1, 0) *
                       random_dataset_point.get(0) +
                       householder.get(1, 1) *
                       random_dataset_point.get(1) +
                       householder.get(1, 2) *
                       random_dataset_point.get(2);
        third_coord = householder.get(2, 0) *
                      random_dataset_point.get(0) +
                      householder.get(2, 1) *
                      random_dataset_point.get(1) +
                      householder.get(2, 2) *
                      random_dataset_point.get(2);
        random_dataset_point.set(0, first_coord);
        random_dataset_point.set(1, second_coord);
        random_dataset_point.set(2, third_coord);
      }
      for (index_t j = 0; j < num_cols; j++) {
        typename Table_t::Point_t random_dataset_point;
        random_dataset.get(j, &random_dataset_point);
        for (index_t i = 0; i < num_rows; i++) {
          random_dataset_point.set(i, -random_dataset_point.get(i));
        }
      }

      // Run QuicPca.
      fl::ml::QuicSVDResult<Table_t> result;
      fl::ml::QuicSVD< QuicSVDArgs1 > quic_svd;
      quic_svd.Compute(required_relative_error_, std::string("covariance"),
                       std::max(num_rows, num_cols), random_dataset,
                       std::string("tmp_lsv"),
                       std::string("tmp_sv"),
                       std::string("tmp_rsv_transposed"),
                       &result);

      fprintf(stderr, "Printing the singular values...\n");
      PrintTable_(result.singular_values);
      fprintf(stderr,
              "Printing the left singular vectors "
              "(should be 2 columns or the last singular value should be "
              "really close to 0.)\n");
      PrintTable_(result.left_singular_vectors);

      typename Table_t::Point_t singular_values;
      result.singular_values.get(0, &singular_values);
      BOOST_ASSERT(result.singular_values.n_attributes() == 2 ||
                   fabs(singular_values[2]) < 1e-6);
      printf("PcaTest(): Passed!\n\n");
    }

    template<bool row_is_degenerate, bool column_is_degenerate>
    static void ZeroMatrixCaseTest() {

      printf("ZeroMatrixCaseTest(): Beginning\n\n");

      // Generate a zero matrix.
      index_t num_rows = (row_is_degenerate) ? 1 : (fl::math::Random(3, 50));
      index_t num_cols = (column_is_degenerate) ? 1 : (fl::math::Random(3, 5));
      fl::dense::Matrix<Precision, false> random_matrix;
      random_matrix.Init(num_rows, num_cols);
      random_matrix.SetZero();

      // Run QuicSVD.
      Table_t table;
      //typename Table_t::IndexArgs args;
      table.Init(random_matrix);
      fl::ml::QuicSVDResult<Table_t> result;
      fl::ml::QuicSVD< QuicSVDArgs2 > quic_svd;
      quic_svd.Compute(required_relative_error_, std::string("covariance"),
                       std::max(num_rows, num_cols), table,
                       std::string("tmp_lsv"),
                       std::string("tmp_sv"),
                       std::string("tmp_rsv_transposed"),
                       &result);

      // Verify that the resulting rank-1 decomposition has zero
      // singular value.
      BOOST_ASSERT(result.singular_values.n_attributes() == 1);
      typename Table_t::Point_t singular_values;
      result.singular_values.get(0, &singular_values);
      BOOST_ASSERT(singular_values[0] == 0.0);
      BOOST_ASSERT(result.left_singular_vectors.n_entries() == 1);

      typename Table_t::Point_t left_singular_vector;
      result.left_singular_vectors.get(0, &left_singular_vector);
      BOOST_ASSERT(left_singular_vector[0] == 1.0);
      for (int i = 1; i < result.left_singular_vectors.n_attributes(); i++) {
        BOOST_ASSERT(left_singular_vector[i] == 0.0);
      }
      for (int i = 0; i < result.right_singular_vectors_transposed.n_entries();
           i++) {
        typename Table_t::Point_t point;
        result.right_singular_vectors_transposed.get(i, &point);
        BOOST_ASSERT(point[0] == 0.0);
      }
      printf("ZeroMatrixCaseTest(): Passed!\n");
    }

    static void ConvertTableToDenseMatrix_(
      const Table_t &table,
      fl::dense::Matrix<double, false> *output) {

      output->Init(table.n_attributes(), table.n_entries());
      for (int i = 0; i < table.n_entries(); i++) {
        typename Table_t::Point_t point;
        table.get(i, &point);
        for (int j = 0; j < table.n_attributes(); j++) {
          output->set(j, i, point[j]);
        }
      }
    }

    static void StressTest() {

      printf("StressTest(): Beginning\n\n");

      // Let's test 100 times on a randomly generated matrix.
      for (index_t i = 0; i < 100; i++) {

        // Random number of rows and columns.
        index_t num_rows = fl::math::Random(3, 50);
        index_t num_cols = fl::math::Random(3, 50);
        fl::dense::Matrix<Precision, false> random_matrix,
        random_matrix_transpose;
        random_matrix.Init(num_rows, num_cols);

        // Generate the random matrix column by column.
        for (index_t j = 0; j < num_cols; j++) {

          // The current column to build.
          fl::dense::Matrix<Precision, true> current_column;
          random_matrix.MakeColumnVector(j, &current_column);

          // Flip a coin whether to replicate the previous column.
          Precision random_number = fl::math::Random<Precision>(0.0, 1.0);

          if (j == 0 || random_number > acceptance_threshold_) {
            GenerateRandomVector_(NULL, &current_column);
          }
          else {
            fl::dense::Matrix<Precision, true> previous_column;
            random_matrix.MakeColumnVector(j - 1, &previous_column);
            GenerateRandomVector_(&previous_column, &current_column);
          }
        } // end of generation...

        // Make a copy before running QuicSVD for later use against naive.
        fl::dense::Matrix<Precision, false> random_matrix_copy;
        random_matrix_copy.Copy(random_matrix);

        // Run QuicSVD.
        Table_t table;
        table.Init(random_matrix);
        fl::ml::QuicSVDResult< Table_t > result;
        fl::ml::QuicSVD< QuicSVDArgs2 > quic_svd;

        std::string svd_method("covariance");
        std::string tmp_lsv_file("tmp_lsv");
        std::string tmp_sv_file("tmp_sv");
        std::string tmp_rsv_transposed_file("tmp_rsv_transposed");
        int rank = std::max(num_rows, num_cols);
        quic_svd.Compute(required_relative_error_,
                         svd_method,
                         rank,
                         table,
                         tmp_lsv_file,
                         tmp_sv_file,
                         tmp_rsv_transposed_file,
                         &result);

        // Reconstruct and measure the Frobenius norm error.
        fl::dense::Matrix<Precision, false> reconstructed_intermed;
        fl::dense::Matrix<Precision, false> reconstructed;
        reconstructed_intermed.Init(result.left_singular_vectors.n_attributes(),
                                    result.left_singular_vectors.n_entries());
        for (int i = 0; i < result.left_singular_vectors.n_entries(); i++) {
          typename Table_t::Point_t point;
          result.left_singular_vectors.get(i, &point);
          for (int j = 0; j < result.left_singular_vectors.n_attributes(); j++) {
            reconstructed_intermed.set(j, i, point[j]);
          }
        }

        for (index_t i = 0; i < result.left_singular_vectors.n_entries(); i++) {
          fl::dense::Matrix<Precision, true> left_col;
          typename Table_t::Point_t singular_values;
          result.singular_values.get(0, &singular_values);
          reconstructed_intermed.MakeColumnVector(i, &left_col);
          fl::dense::ops::ScaleExpert(singular_values[i], &left_col);
        }

        fl::dense::Matrix<Precision, false>
        right_singular_vectors_transposed_copy;
        ConvertTableToDenseMatrix_(result.right_singular_vectors_transposed,
                                   &right_singular_vectors_transposed_copy);

        fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(
          reconstructed_intermed,
          right_singular_vectors_transposed_copy,
          &reconstructed);

        printf("Reconstructed %d by %d matrix.\n", reconstructed.n_rows(),
               reconstructed.n_cols());
        printf("Used the left singular vectors of %d by %d.\n",
               result.left_singular_vectors.n_attributes(),
               result.left_singular_vectors.n_entries());
        printf("Used the transposed right singular vectors of %d by %d.\n",
               result.right_singular_vectors_transposed.n_attributes(),
               result.right_singular_vectors_transposed.n_entries());
        printf("Original is %d by %d.\n", random_matrix.n_rows(),
               random_matrix.n_cols());

        // Compute the Frobenius norm of the original matrix and the
        // reconstructed matrix and the relative error.
        Precision original_frobenius_norm =
          fl::dense::ops::Dot(random_matrix_copy.n_elements(),
                              random_matrix_copy.ptr(),
                              random_matrix_copy.ptr());
        Precision error_norm = 0.0;

        for (index_t j = 0; j < reconstructed.n_cols(); j++) {
          for (index_t i = 0; i < reconstructed.n_rows(); i++) {
            error_norm += fl::math::Sqr(reconstructed.get(i, j) -
                                        random_matrix_copy.get(i, j));
          }
        }

        printf("Achieved %g absolute error versus %g required absolute error "
               "on a %d by %d matrix versus a %d by %d matrix\n",
               error_norm,
               required_relative_error_ * original_frobenius_norm,
               reconstructed.n_rows(), reconstructed.n_cols(), num_rows,
               num_cols);
        if (error_norm > required_relative_error_ * original_frobenius_norm) {
          FILE *failed_matrix_output = fopen("failed.txt", "w+");
          random_matrix_copy.PrintDebug("", failed_matrix_output);
          fclose(failed_matrix_output);
        }
        BOOST_ASSERT
        (error_norm <= required_relative_error_ * original_frobenius_norm);
      } // end of the trials.

      printf("StressTest(): Passed!\n\n");
    }
};

template<typename Precision_t, bool sort_points>
const Precision_t TestQuicSVD<Precision_t, sort_points>::acceptance_threshold_
= 0.4;

template<typename Precision_t, bool sort_points>
const Precision_t TestQuicSVD<Precision_t, sort_points>::scaling_perturbation_
= 0.2;

template<typename Precision_t, bool sort_points>
const Precision_t TestQuicSVD<Precision_t, sort_points>::additive_perturbation_
= 0.05;

template<typename Precision_t, bool sort_points>
const Precision_t TestQuicSVD<Precision_t, sort_points>::
required_relative_error_ = 0.01;

BOOST_AUTO_TEST_SUITE(TestSuiteQuicSVD)
BOOST_AUTO_TEST_CASE(TestCaseQuicSVD) {
  TestQuicSVD<float, true>::Init();
  TestQuicSVD<double, true>::StressTest();
  TestQuicSVD<double, false>::StressTest();
  //TestQuicSVD<float, true>::StressTest();
  //TestQuicSVD<float, false>::StressTest();
  TestQuicSVD<double, true>::ZeroMatrixCaseTest<true, true>();
  TestQuicSVD<double, true>::ZeroMatrixCaseTest<true, false>();
  TestQuicSVD<double, true>::ZeroMatrixCaseTest<false, true>();
  TestQuicSVD<double, true>::ZeroMatrixCaseTest<false, false>();
  TestQuicSVD<double, true>::PcaTest();
  TestQuicSVD<double, true>::FinalMessage();
}
BOOST_AUTO_TEST_SUITE_END()
