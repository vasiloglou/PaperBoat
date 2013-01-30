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
#undef BOOST_ALL_DYN_LINK
#include "boost/test/included/unit_test.hpp"
#include <stdio.h>
#include <limits>
#include <vector>
#include <map>
#include "fastlib/sparse/matrix.h"

class MatrixTest {
  private:
    void Init() {
      for (index_t i = 0; i < num_of_cols_; i++) {
        for (index_t j = 0; j < num_of_rows_; j++) {
          mat_[i][j] = (i + j) * ((i + j) % 2);
        }
      }
    }

    void Destruct() {
      delete smat_;
    }

    void TestInit1() {
      smat_ = new fl::sparse::Matrix<index_t, double>(
        num_of_rows_,
        num_of_cols_, 3);

      smat_->StartLoadingRows();
      std::vector<index_t> ind;
      std::vector<double>  val;
      for (index_t i = 0; i < num_of_cols_; i++) {
        ind.clear();
        val.clear();
        for (index_t j = 0; j < num_of_cols_; j++) {
          if (mat_[i][j] != 0) {
            ind.push_back(j);
            val.push_back(mat_[i][j]);
          }
        }
        smat_->LoadRow(i, ind, val);
      }

      for (index_t i = 0; i < num_of_rows_; i++) {
        for (index_t j = 0; j < num_of_cols_; j++) {
          BOOST_CHECK_CLOSE(smat_->get(i, j),
                            mat_[i][j],
                            std::numeric_limits<double>::epsilon());
        }
      }
      NOTIFY("TestInit1 sucess!!\n");
    }

    void TestInit2() {
      std::vector<index_t> rows;
      std::vector<index_t> cols;
      std::vector<double>  vals;
      std::vector<index_t> nnz(num_of_rows_);
      for (index_t i = 0; i < num_of_rows_; i++) {
        for (index_t j = 0; j < num_of_cols_; j++) {
          if (mat_[i][j] != 0) {
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(mat_[i][j]);
            nnz[i]++;
          }
        }
      }
      /*for(index_t i=0; i<(index_t)rows.size(); i++) {
        printf("%i %i %lg\n",
                rows[i],
               cols[i],
               vals[i]);
      }*/
      smat_ = new fl::sparse::Matrix<index_t, double>();
      smat_->Init(rows, cols, vals,
                  *(std::max_element(nnz.begin(), nnz.end())), num_of_rows_);
      // printf("%s\n", smat_->Print().c_str());
      for (index_t i = 0; i < num_of_rows_; i++) {
        for (index_t j = 0; j < num_of_cols_; j++) {
          BOOST_CHECK_CLOSE(smat_->get(i, j), mat_[i][j],
                            std::numeric_limits<double>::epsilon());
        }
      }
      NOTIFY("TestInit2 success!!\n");
    }

    void TestInit3() {
      FILE *fp = fopen("temp.txt", "w");
      if (fp == NULL) {
        FATAL("Cannot open temp.txt error %s", strerror(errno));
      }
      for (index_t i = 0; i < num_of_cols_; i++) {
        for (index_t j = 0; j < num_of_cols_; j++) {
          if (mat_[i][j] != 0) {
            fprintf(fp, "%i %i %g\n", i, j, mat_[i][j]);
          }
        }
      }
      fclose(fp);
      smat_ = new fl::sparse::Matrix<index_t, double>();
      smat_->Init("temp.txt");
      unlink("temp.txt");
      for (index_t i = 0; i < num_of_rows_; i++) {
        for (index_t j = 0; j < num_of_cols_; j++) {
          BOOST_CHECK_CLOSE(smat_->get(i, j),
                            mat_[i][j],
                            std::numeric_limits<double>::epsilon());
        }
      }
      NOTIFY("TestInit3 success!!");
    }

    void TestCopyConstructor() {
      smat_ = new fl::sparse::Matrix<index_t, double>();
      NOTIFY("TestCopyConstructor success!!\n");
    }

    void TestMakeSymmetric() {
      TestInit1();
      smat_->set(2, 3, 1.44);
      smat_->set(3, 2, 0.74);
      smat_->set(7, 8, 4.33);
      smat_->set(8, 7, 0.22);
      smat_->MakeSymmetric();
      for (index_t i = 0; i < num_of_rows_; i++) {
        for (index_t j = 0; j < num_of_cols_; j++) {
          BOOST_CHECK_CLOSE(smat_->get(i, j),
                            smat_->get(j, i),
                            std::numeric_limits<double>::epsilon());
        }
      }
      NOTIFY("Test MakeSymmetric success!!\n");
    }

    void TestNegate() {
      TestInit1();
      smat_->Negate();
      for (index_t i = 0; i < num_of_rows_; i++) {
        for (index_t j = 0; j < num_of_cols_; j++) {
          BOOST_CHECK_CLOSE(smat_->get(i, j), -mat_[i][j],
                            std::numeric_limits<double>::epsilon());
        }
      }
      NOTIFY("Test Negate success!!");
    }

    void TestColumnScale() {
      TestInit1();
      fl::dense::Matrix<double, false>  scale;
      scale.Init(num_of_cols_);
      for (index_t i = 0; i < num_of_cols_; i++) {
        scale[i] = i;
      }
      smat_->EndLoading();
      smat_->ColumnScale(scale);
      for (index_t i = 0; i < num_of_rows_; i++) {
        for (index_t j = 0; j < num_of_cols_; j++) {
          BOOST_CHECK_CLOSE(smat_->get(i, j), j*mat_[i][j],
                            std::numeric_limits<double>::epsilon());
        }
      }
      NOTIFY("Test ColumnScale success!!");
    }

    void TestRowScale() {
      TestInit1();
      fl::dense::Matrix<double, false> scale;
      scale.Init(num_of_cols_);
      for (index_t i = 0; i < num_of_cols_; i++) {
        scale[i] = i;
      }
      smat_->EndLoading();
      smat_->RowScale(scale);
      for (index_t i = 0; i < num_of_rows_; i++) {
        for (index_t j = 0; j < num_of_cols_; j++) {
          BOOST_CHECK_CLOSE(smat_->get(i, j), i*mat_[i][j],
                            std::numeric_limits<double>::epsilon());
        }
      }
      NOTIFY("Test RowScale success!!");
    }

    void TestRowSums() {
      TestInit1();
      fl::dense::Matrix<double, false> row_sums;
      smat_->EndLoading();
      smat_->RowSums(&row_sums);
      for (index_t i = 0; i < num_of_rows_; i++) {
        double row_sum = 0;
        for (index_t j = 0; j < num_of_cols_; j++) {
          row_sum += mat_[i][j];
        }
        BOOST_CHECK_CLOSE(row_sums[i], row_sum, 0.001);
      }
      NOTIFY("Test RowSums success!!");
    }

    void TestInvRowSums() {
      TestInit1();
      fl::dense::Matrix<double, false> row_sums;
      smat_->EndLoading();
      smat_->InvRowSums(&row_sums);
      for (index_t i = 0; i < num_of_rows_; i++) {
        double row_sum = 0;
        for (index_t j = 0; j < num_of_cols_; j++) {
          row_sum += mat_[i][j];
        }
        BOOST_CHECK_CLOSE(row_sums[i], 1.0 / row_sum ,
                          std::numeric_limits<double>::epsilon());
      }
      NOTIFY("Test InvRowSums success!!");
    }

    void TestInvColMaxs() {
      TestInit1();
      fl::dense::Matrix<double, false> col_maxs;
      smat_->EndLoading();
      smat_->InvColMaxs(&col_maxs);
      for (index_t i = 0; i < num_of_cols_; i++) {
        double col_max = 0;
        for (index_t j = 0; j < num_of_rows_; j++) {
          col_max = max(col_max, mat_[j][i]);
        }
        BOOST_CHECK_CLOSE(col_maxs[i], 1.0 / col_max,
                          std::numeric_limits<double>::epsilon());
      }
      NOTIFY("Test InvColMaxs success!!");

    }

    void TestEig() {
      TestInit1();
      smat_->EndLoading();
      fl::dense::Matrix<double, false> eigvalues_real;
      fl::dense::Matrix<double, false> eigvalues_imag;
      fl::dense::Matrix<double, false> eigvectors;
      smat_->Eig(1, "LM", &eigvectors, &eigvalues_real, &eigvalues_imag);
      // eigvectors.PrintDebug();
      NOTIFY("Test Eigenvector success!!\n");
    }


    void TestBasicOperations() {
      fl::sparse::Matrix<index_t, double> a(input_files_directory_ + "/sparse_matrices/A.txt");
      a.EndLoading();
      fl::sparse::Matrix<index_t, double> b(input_files_directory_ + "/sparse_matrices/B.txt");
      b.EndLoading();
      fl::sparse::Matrix<index_t, double> a_plus_b(input_files_directory_ + "/sparse_matrices/AplusB.txt");
      fl::sparse::Matrix<index_t, double> a_minus_b(input_files_directory_ + "/sparse_matrices/AminusB.txt");
      fl::sparse::Matrix<index_t, double> a_times_b(input_files_directory_ + "/sparse_matrices/AtimesB.txt");
      fl::sparse::Matrix<index_t, double> a_dot_times_b(input_files_directory_ + "/sparse_matrices/AdottimesB.txt");

      fl::sparse::Matrix<index_t, double> temp;
      fl::sparse::ops::Add<fl::la::Init>(a, b, &temp);
//    temp.EndLoading();
//      a_plus_b.EndLoading();
//    printf("%s\n", temp.Print().c_str());
//    printf("%s\n", a_plus_b.Print().c_str());
      for (index_t i = 0; i < 20; i++) {
        for (index_t j = 0; j < 20; j++) {
          BOOST_CHECK_CLOSE(a_plus_b.get(i, j), temp.get(i, j), 0.01);
        }
      }
      temp.Destruct();
      NOTIFY("Matrix addition sucess!!\n");

      fl::sparse::ops::Sub<fl::la::Init>(b, a, &temp);
//    temp.EndLoading();
//    a_minus_b.EndLoading();
      for (index_t i = 0; i < 21; i++) {
        for (index_t j = 0; j < 21; j++) {
          BOOST_CHECK_CLOSE(a_minus_b.get(i, j), temp.get(i, j), 0.01);
        }
      }
      temp.Destruct();
      NOTIFY("Matrix subtraction success!!\n");

      fl::sparse::ops::Multiply(a, b, &temp);
//    printf("%s\n", temp.Print().c_str());
//    printf("%s\n", a_times_b.Print().c_str());
//    temp.EndLoading();
//    a_times_b.EndLoading();
      for (index_t i = 0; i < 21; i++) {
        for (index_t j = 0; j < 21; j++) {
          BOOST_CHECK_CLOSE(a_times_b.get(i, j), temp.get(i, j), 0.01);
        }
      }
      temp.Destruct();
      NOTIFY("Matrix multiplication success!!\n");

      fl::sparse::Matrix<index_t, double> i_w(input_files_directory_ + "/sparse_matrices/I-W.txt");
      fl::sparse::Matrix<index_t, double> i_w_i_w_trans(input_files_directory_ + "/sparse_matrices/I-W_time_I-W_trans.txt");
      i_w.EndLoading();
      fl::sparse::ops::MultiplyT<index_t, double>(i_w, &temp);
//  printf("%s\n", temp.Print().c_str());
//  printf("%s\n", a_times_a_trans.Print().c_str());
//    temp.EndLoading();
//  printf("%s\n", i_w_i_w_trans.Print().c_str());
      for (index_t i = 0; i < i_w.n_rows(); i++) {
        for (index_t j = 0; j < i_w.n_cols(); j++) {
          BOOST_CHECK_CLOSE(i_w_i_w_trans.get(i, j), temp.get(i, j), 0.01);
        }
      }
      temp.Destruct();
      NOTIFY("Matrix multiplication success!!\n");


      fl::sparse::Matrix<>::DotMultiply<index_t, double>(a, b, &temp);
//    printf("%s\n", temp.Print().c_str());
//    printf("%s\n", a_dot_times_b.Print().c_str());
//    temp.EndLoading();
      a_dot_times_b.EndLoading();

      for (index_t i = 0; i < a_dot_times_b.n_rows(); i++) {
        for (index_t j = 0; j < a_dot_times_b.n_cols(); j++) {
          BOOST_CHECK_CLOSE(a_dot_times_b.get(i, j), temp.get(i, j), 0.01);
        }
      }
      temp.Destruct();
      NOTIFY("Matrix dot multiplication success!!\n");

      fl::sparse::ops::Multiply<index_t, double>(a, 3.45, &temp);
      for (index_t i = 0; i < 21; i++) {
        for (index_t j = 0; j < 21; j++) {
          BOOST_CHECK_CLOSE(3.45 * a.get(i, j), temp.get(i, j),
                            std::numeric_limits<double>::epsilon());
        }
      }
      temp.Destruct();
      NOTIFY("Matrix scalar multiplication success!!\n");

    }
    // void TestLinSolve() {
    //   TestInit1();
    //   fl::dense::Matrix<double, false> b, x;
    //   b.Init(num_of_cols_);
    //   b.SetZero();
    //   x.Init(num_of_cols_);
    //   x.SetAll(1);
    //   smat_->MakeSymmetric();
    //   smat_->EndLoading();
    //   smat_->LinSolve(b, &x);
    //   x.PrintDebug();
    //   NOTIFY("Test Linear Solve success!!\n");
    // }

  public:
    MatrixTest(std::string input_files_dir_in) {
      input_files_directory_ = input_files_dir_in;
    }

    void RunTests() {
      Init();
      TestInit1();
      Destruct();
      Init();
      TestInit2();
      Destruct();
      Init();
      TestInit3();
      Destruct();
      Init();
      TestCopyConstructor();
      Destruct();
      Init();
      TestMakeSymmetric();
      Destruct();
      Init();
      TestNegate();
      Destruct();
      Init();
      TestColumnScale();
      Destruct();
      Init();
      TestRowScale();
      Destruct();
      Init();
      TestRowSums();
      Destruct();
      Init();
      TestInvRowSums();
      Destruct();
      Init();
      TestInvColMaxs();
      Destruct();
      Init();
      TestEig();
      Destruct();
      Init();
      TestBasicOperations();
      // Init();
      // TestLinSolve();
      // Destruct();
    }

  private:
    fl::sparse::Matrix<index_t, double> *smat_;
    static const index_t num_of_cols_  = 80;
    static const index_t num_of_rows_  = 80;
    static const index_t num_of_nnz_   = 4;
    double mat_[num_of_rows_][num_of_cols_];
    std::vector<index_t>  indices_;
    std::vector<index_t>  rows_;
    std::string input_files_directory_;
};

class MatrixTestSuite : public boost::unit_test_framework::test_suite {
  public:
    MatrixTestSuite(std::string input_files_dir_in)
        : boost::unit_test_framework::test_suite("Matrix test suite") {
      // create an instance of the test cases class
      boost::shared_ptr<MatrixTest> instance(new MatrixTest(input_files_dir_in));
      // create the test cases
      boost::unit_test_framework::test_case* matrix_test_case
      = BOOST_CLASS_TEST_CASE(&MatrixTest::RunTests, instance);
      // add the test cases to the test suite
      add(matrix_test_case);
    }
};

boost::unit_test_framework::test_suite*
init_unit_test_suite(int argc, char** argv) {
  // create the top test suite
  boost::unit_test_framework::test_suite* top_test_suite
  = BOOST_TEST_SUITE("Matrix tests");
  if (argc != 2) {
    NOTIFY("Wrong number of arguments for matrix test. Expected test input files directory. Returning NULL.");
    return NULL;
  }
  // add test suites to the top test suite
  std::string input_files_directory = argv[1];
  top_test_suite->add(new MatrixTestSuite(input_files_directory));
  return top_test_suite;
}





