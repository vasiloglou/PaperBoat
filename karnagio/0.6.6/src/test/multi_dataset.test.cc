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

#include "boost/test/unit_test.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/lexical_cast.hpp"
#include "fastlib/base/mpl.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/data/multi_dataset_dev.h"

class MultiDatasetTest {
  private :
    struct Arguments1 : public fl::data::DatasetArgs {
      typedef boost::mpl::vector1<float> DenseTypes;
      typedef double CalcPrecisionType;
      typedef fl::data::DatasetArgs::Compact StorageType;
    };

    void TestCsv() {
      fl::data::MultiDataset<Arguments1> dataset1, dataset2, dataset3;
      std::string csv_file = input_files_directory_ + "/3_rows_double.csv";
      dataset1.Init(csv_file, "r");
      fl::data::MultiDataset<Arguments1>::Point_t point1;
      BOOST_ASSERT(dataset1.n_points() == 3);
      dataset1.get(0, &point1);
      BOOST_ASSERT(point1[0] == 1);
      BOOST_ASSERT(point1[1] == 2);
      BOOST_ASSERT(point1[2] == 3);
      dataset1.get(1, &point1);
      BOOST_ASSERT(point1[0] == 4);
      BOOST_ASSERT(point1[1] == 5);
      BOOST_ASSERT(point1[2] == 6);
      dataset1.get(2, &point1);
      BOOST_ASSERT(point1[0] == 7);
      BOOST_ASSERT(point1[1] == 8);
      BOOST_ASSERT(point1[2] == 9);
      // test serialization
      {
        std::ofstream ofs("filename");
        boost::archive::text_oarchive oa(ofs);
        oa << dataset1;
      }
      {
        fl::data::MultiDataset<Arguments1> new_dataset1;
        std::ifstream ifs("filename");
        boost::archive::text_iarchive ia(ifs);
        ia >> new_dataset1;
        BOOST_ASSERT(new_dataset1.n_points() == 3);
        new_dataset1.get(0, &point1);
        BOOST_ASSERT(point1[0] == 1);
        BOOST_ASSERT(point1[1] == 2);
        BOOST_ASSERT(point1[2] == 3);
        new_dataset1.get(1, &point1);
        BOOST_ASSERT(point1[0] == 4);
        BOOST_ASSERT(point1[1] == 5);
        BOOST_ASSERT(point1[2] == 6);
        new_dataset1.get(2, &point1);
        BOOST_ASSERT(point1[0] == 7);
        BOOST_ASSERT(point1[1] == 8);
        BOOST_ASSERT(point1[2] == 9);
      }
      // now write everything to a new file
      std::vector<index_t> dense_dims(1);
      dense_dims[0] = 3;
      dataset2.Init(dense_dims, std::vector<index_t>(), 3);
      dataset2.get(0, &point1);
      point1[0] = 1;
      point1[1] = 2;
      point1[2] = 3;
      dataset2.get(1, &point1);
      point1[0] = 4;
      point1[1] = 5;
      point1[2] = 6;
      dataset2.get(2, &point1);
      point1[0] = 7;
      point1[1] = 8;
      point1[2] = 9;
      dataset2.Save(input_files_directory_ + "/temp1.csv",
                    false,
                    std::vector<std::string>(),
                    ",");
      // reopen that file and see if you can read it
      dataset3.Init(csv_file, "r");
      BOOST_ASSERT(dataset3.n_points() == 3);
      dataset3.get(0, &point1);
      BOOST_ASSERT(point1[0] == 1);
      BOOST_ASSERT(point1[1] == 2);
      BOOST_ASSERT(point1[2] == 3);
      dataset3.get(1, &point1);
      BOOST_ASSERT(point1[0] == 4);
      BOOST_ASSERT(point1[1] == 5);
      BOOST_ASSERT(point1[2] == 6);
      dataset3.get(2, &point1);
      BOOST_ASSERT(point1[0] == 7);
      BOOST_ASSERT(point1[1] == 8);
      BOOST_ASSERT(point1[2] == 9);
    }
    struct Arguments2 : public fl::data::DatasetArgs {
      typedef boost::mpl::vector1<double> SparseTypes;
      typedef double CalcPrecisionType;
      typedef fl::data::DatasetArgs::Compact StorageType;
    };

    void TestSparse() {
      /*
            typedef boost::mpl::vector1<double> SparseTypeList;
            typedef boost::mpl::map3<
            boost::mpl::pair<fl::data::DatasetArgs::SparseTypes, SparseTypeList>,
            boost::mpl::pair<fl::data::DatasetArgs::CalcPrecision, double>,
            boost::mpl::pair<fl::data::DatasetArgs::StorageType, fl::data::DatasetArgs::StorageType::Compact>
            > Arguments1;
      */
      fl::data::MultiDataset<Arguments2> dataset1;
      std::string csv_file = input_files_directory_ + "/3_rows_double_sparse.csv";
      dataset1.Init(csv_file, "r");
      fl::data::MultiDataset<Arguments2>::Point_t point1;
      BOOST_ASSERT(dataset1.n_points() == 3);
      dataset1.get(0, &point1);
      BOOST_ASSERT(point1[0] == 1);
      BOOST_ASSERT(point1[4] == 5);
      BOOST_ASSERT(point1[66] == 67);
      dataset1.get(1, &point1);
      BOOST_ASSERT(point1[1] == 1);
      BOOST_ASSERT(point1[4] == 4);
      BOOST_ASSERT(point1[76] == 76);
      dataset1.get(2, &point1);
      BOOST_ASSERT(point1[2] == 3);
      BOOST_ASSERT(point1[88] == 89);
      BOOST_ASSERT(point1[98] == 99);
      {
        std::ofstream ofs("filename");
        boost::archive::text_oarchive oa(ofs);
        oa << dataset1;
      }
      {
        fl::data::MultiDataset<Arguments1> new_dataset1;
        std::ifstream ifs("filename");
        boost::archive::text_iarchive ia(ifs);
        ia >> new_dataset1;
        BOOST_ASSERT(new_dataset1.n_points() == 3);
        dataset1.get(0, &point1);
        BOOST_ASSERT(point1[0] == 1);
        BOOST_ASSERT(point1[4] == 5);
        BOOST_ASSERT(point1[66] == 67);
        dataset1.get(1, &point1);
        BOOST_ASSERT(point1[1] == 1);
        BOOST_ASSERT(point1[4] == 4);
        BOOST_ASSERT(point1[76] == 76);
        dataset1.get(2, &point1);
        BOOST_ASSERT(point1[2] == 3);
        BOOST_ASSERT(point1[88] == 89);
        BOOST_ASSERT(point1[98] == 99);
      }

      // test invalid file
      //fl::data::MultiDataset<Arguments1> dataset2;
      //csv_file= input_files_directory_ + "/3_rows_double_sparse_invalid.csv";
      //dataset2.Init(csv_file, "r");
    }

    struct Arguments2_2 : public fl::data::DatasetArgs {
      typedef boost::mpl::vector2<float, double> SparseTypes;
      typedef double CalcPrecisionType;
      typedef fl::data::DatasetArgs::Compact StorageType;
    };

    void TestSparse1() {
      fl::data::MultiDataset<Arguments2_2> dataset1;
      std::string csv_file = input_files_directory_ + "/3_rows_float_double_sparse.csv";
      dataset1.Init(csv_file, "r");
      fl::data::MultiDataset<Arguments2_2>::Point_t point1;
      BOOST_ASSERT(dataset1.n_points() == 3);
      dataset1.get(0, &point1);
      BOOST_ASSERT(point1[0] == 1);
      BOOST_ASSERT(point1[4] == 5);
      BOOST_ASSERT(point1[66] == 67);
      dataset1.get(1, &point1);
      BOOST_ASSERT(point1[1] == 1);
      BOOST_ASSERT(point1[4] == 4);
      BOOST_ASSERT(point1[76] == 76);
      dataset1.get(2, &point1);
      BOOST_ASSERT(point1[2] == 3);
      BOOST_ASSERT(point1[88] == 89);
      BOOST_ASSERT(point1[98] == 99);

      // test invalid file
      //fl::data::MultiDataset<Arguments1> dataset2;
      //csv_file= input_files_directory_ + "/3_rows_double_sparse_invalid.csv";
      //dataset2.Init(csv_file, "r");
    }

    struct Arguments3 : public fl::data::DatasetArgs {
      typedef boost::mpl::vector2<float, double> DenseTypes;
      typedef boost::mpl::vector1<double> SparseTypes;
      typedef float CalcPrecisionType;
      typedef fl::data::DatasetArgs::Compact StorageType;
    };

    void Test1Dense1Sparse() {
      fl::data::MultiDataset<Arguments3> dataset1, dataset2, dataset3;
      std::string csv_file = input_files_directory_ + "/3_rows_mixed_1.csv";
      dataset1.Init(csv_file, "r");
      fl::data::MultiDataset<Arguments3>::Point_t point1;

      BOOST_ASSERT(dataset1.n_points() == 3);
      BOOST_ASSERT(dataset1.n_attributes() == 34);
      dataset1.get(0, &point1);
    }

    struct Arguments4 : public fl::data::DatasetArgs {
      typedef boost::mpl::vector1<double> DenseTypes;
      typedef boost::mpl::vector1<bool> SparseTypes;
      typedef double CalcPrecisionType;
      typedef fl::data::DatasetArgs::Compact StorageType;
    };

    void TestDenseCategorical() {

      fl::data::MultiDataset<Arguments4> dataset1, dataset2;
      std::string csv_file = input_files_directory_ + "/cont_categorical.csv";
      dataset1.Init(csv_file, "r");
      fl::data::MultiDataset<Arguments4>::Point_t point1;

      BOOST_ASSERT(dataset1.n_points() == 2000);
      dataset1.get(0, &point1);
      dataset2.InitRandom(std::vector<index_t>(1,10),
                   std::vector<index_t>(1,1000),
                   1000,
                   0.0,
                   1.0,
                   0.01);
    }

    struct Arguments5 : public fl::data::DatasetArgs {
      typedef boost::mpl::vector1<float> DenseTypes;
      typedef float CalcPrecisionType;
      typedef fl::data::DatasetArgs::Compact StorageType;
      typedef fl::MakeIntIndexedStruct<boost::mpl::vector1<index_t> >::Generated MetaDataType;
    };


    void TestDenseMeta() {
      fl::data::MultiDataset<Arguments5> dataset1, dataset2;
      std::string csv_file = input_files_directory_ + "/3_rows_double_meta.csv";
      dataset1.Init(csv_file, "r");
      fl::data::MultiDataset<Arguments5>::Point_t point1;

      BOOST_ASSERT(dataset1.n_points() == 3);
      for (index_t i = 0; i < dataset1.n_points(); i++) {
        dataset1.get(i, &point1);
        for (size_t j = 0; j < point1.size(); j++) {
          BOOST_ASSERT(point1[j] == 3*i + j + 1);
          BOOST_ASSERT(point1.meta_data().get<0>() == i);
        }
      }
      dataset1.Save(input_files_directory_ + "/temp1.csv",
                    true,
                    std::vector<std::string>(),
                    ",");
      dataset2.Init(input_files_directory_ + "/temp1.csv", "r");
      BOOST_ASSERT(dataset2.n_points() == 3);
      for (index_t i = 0; i < dataset1.n_points(); i++) {
        dataset2.get(i, &point1);
        for (size_t j = 0; j < point1.size(); j++) {
          BOOST_ASSERT(point1[j] == 3*i + j + 1);
          BOOST_ASSERT(point1.meta_data().get<0>() == i);
        }
      }
    }

  public:
    MultiDatasetTest(std::string input_files_dir_in) {
      input_files_directory_ = input_files_dir_in;
    }

    void RunTests() {
      TestCsv();
      BOOST_MESSAGE("Csv dataset test passed !!!");
      TestSparse();
      BOOST_MESSAGE("Sparse dataset tests passed !!!");
      TestSparse1();
      BOOST_MESSAGE("Sparse dataset tests passed !!!");
      Test1Dense1Sparse();
      BOOST_MESSAGE("1 Dense, 1 Sparse dataset test passed !!!");
      TestDenseCategorical();
      BOOST_MESSAGE("Dense Categorical passed!!");
      TestDenseMeta();
      BOOST_MESSAGE("Dense with metadata passed !!!");
      BOOST_MESSAGE("Congratulations All tests passed");
    }

  private:

    std::string input_files_directory_;

};



class MultiDatasetTestSuite : public boost::unit_test_framework::test_suite {
  public:
    MultiDatasetTestSuite(std::string input_files_dir_in)
        : boost::unit_test_framework::test_suite("MultiDataset test suite") {
      // create an instance of the test cases class
      boost::shared_ptr<MultiDatasetTest> instance(new MultiDatasetTest(input_files_dir_in));
      // create the test cases
      boost::unit_test_framework::test_case* multidataset_test_case
      = BOOST_CLASS_TEST_CASE(&MultiDatasetTest::RunTests, instance);
      // add the test cases to the test suite
      add(multidataset_test_case);
    }
};


boost::unit_test_framework::test_suite*
init_unit_test_suite(int argc, char** argv) {
  // create the top test suite
  boost::unit_test::framework::master_test_suite().p_name.value="Multi-Dataset tests";
  if (argc != 2) {
    fl::logger->Die()<< "Wrong number of arguments for multidatset test. "
      "Expected test input files directory. Returning NULL.";
    return NULL;
  }
  // add test suites to the top test suite
  std::string input_files_directory = argv[1];

  boost::unit_test::framework::master_test_suite().add(new MultiDatasetTestSuite(input_files_directory));
  return 0;
}


int main(int argc, char**argv) {
  if (argc != 2) {
    fl::logger->Die()<< "Wrong number of arguments for multidatset test. "
      "Expected test input files directory. Returning NULL.";
    return 0;
  }
  init_unit_test_suite(argc, argv);
  // add test suites to the top test suite
  std::string input_files_directory = argv[1];
  MultiDatasetTest(input_files_directory).RunTests();
return 0;
}

