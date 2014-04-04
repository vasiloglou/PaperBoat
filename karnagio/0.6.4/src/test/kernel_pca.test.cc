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
#include "boost/program_options.hpp"
#include "boost/test/included/unit_test.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/if.hpp"
#include "fastlib/base/base.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/table/file_data_access.h"
#include "fastlib/table/branch_on_table_dev.h"
#include "fastlib/table/table_dev.h"
#include "mlpack/kernel_pca/greedy_kernel_pca_dev.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "fastlib/metric_kernel/cosine_premetric.h"
#include "fastlib/table/default/dense/labeled/balltree/table.h"

template<bool do_centering>
class KernelPcaTestSuite : public boost::unit_test_framework::test_suite {

  private:

    class KernelPcaTest {
      private:

        std::vector<std::string> point_types_;

        void ConstructArgsForTrial_(const fl::table::FileDataAccess &data,
                                    const std::string &point_type,
                                    std::vector<std::string> *args_out) {

          // Push in the point type and the tree type.
          std::string point_arg = std::string("--point=") + point_type;
          std::string tree_arg = std::string("--tree=balltree");
          std::string references_arg =
            std::string("--references_in=random.csv");
          std::string bandwidth_rag("");
          std::stringstream bandwidth_sstr;
          double bandwidth = fl::math::Random<double>(0.5, 2.0);
          bandwidth_sstr << "--bandwidth=" << bandwidth;

          // Push in the references_in argument.
          args_out->push_back(point_arg);
          args_out->push_back(tree_arg);
          args_out->push_back(references_arg);

          // Generate a random bandwidth.
          args_out->push_back(bandwidth_sstr.str());

          // Push in the random number of components to extract.
          double num_components = fl::math::Random(2, 10);
          std::stringstream k_eigenvectors_sstr;
          k_eigenvectors_sstr << "--k_eigenvectors=" << num_components;
          args_out->push_back(k_eigenvectors_sstr.str());
        }

      public:

        template<typename TableType>
        class DenseDatasetAllocator {
          public:
            static void Main(fl::table::FileDataAccess &data,
                             int num_dense_dimensions,
                             int num_sparse_dimensions,
                             int num_points,
                             TableType *random_dataset) {

              // This line assumes that the datasets that are dense
              // contain only 1 dense type.
              data.Attach(std::string("random.csv"),
                          std::vector<int>(1, num_dense_dimensions +
                                           num_sparse_dimensions),
                          std::vector<int>(),
                          num_points,
                          random_dataset);
            }
        };

        template<typename TableType>
        class SparseDatasetAllocator {
          public:
            static void Main(fl::table::FileDataAccess &data,
                             int num_dense_dimensions,
                             int num_sparse_dimensions,
                             int num_points,
                             TableType *random_dataset) {

              // This line assumes that the datasets that are dense
              // contain only 1 dense type.
              data.Attach(std::string("random.csv"),
                          std::vector<int>(),
                          std::vector<int>(1, num_dense_dimensions +
                                           num_sparse_dimensions),
                          num_points,
                          random_dataset);
            }
        };

        template<typename TableType>
        class MixedDatasetAllocator {
          public:

            static void Main(fl::table::FileDataAccess &data,
                             int num_dense_dimensions,
                             int num_sparse_dimensions,
                             int num_points,
                             TableType *random_dataset) {

              // This line assumes that the datasets with categorical
              // features are 1 dense type, and 1 sparse type.
              data.Attach(std::string("random.csv"),
                          std::vector<int>(1, num_dense_dimensions),
                          std::vector<int>(1, num_sparse_dimensions),
                          num_points,
                          random_dataset);
            }
        };

        template<typename TableType1>
        class Core {

          public:

            static int Main(fl::table::FileDataAccess *data,
                            boost::program_options::variables_map &vm) {

              // Generate a random dataset.
              int num_points = fl::math::Random(100, 500);
              int num_dense_dimensions = fl::math::Random(10, 15);
              int num_sparse_dimensions = fl::math::Random(10, 15);

              TableType1 random_dataset;

              // Depending on whether the dataset contains a dense
              // type or not, branch on the random table generation.
              if (vm["point"].as<std::string>() == std::string("dense")) {
                DenseDatasetAllocator<TableType1>::Main(
                  *data, num_dense_dimensions,
                  num_sparse_dimensions,
                  num_points, &random_dataset);
              }
              else if (vm["point"].as<std::string>() == std::string("sparse")) {
                SparseDatasetAllocator<TableType1>::Main(
                  *data, num_dense_dimensions,
                  num_sparse_dimensions,
                  num_points, &random_dataset);
              }
              else if (vm["point"].as<std::string>() == std::string("dense_sparse")) {
                MixedDatasetAllocator<TableType1>::Main(
                  *data, num_dense_dimensions,
                  num_sparse_dimensions,
                  num_points, &random_dataset);
              }
              else if (vm["point"].as<std::string>() == std::string("categorical")) {
                SparseDatasetAllocator<TableType1>::Main(
                  *data, num_dense_dimensions,
                  num_sparse_dimensions,
                  num_points, &random_dataset);
              }
              else if (vm["point"].as<std::string>() == std::string("dense_categorical")) {
                MixedDatasetAllocator<TableType1>::Main(
                  *data, num_dense_dimensions,
                  num_sparse_dimensions,
                  num_points, &random_dataset);
              }

              // Randomize the points.
              for (int i = 0; i < num_points; i++) {
                typename TableType1::Point_t point;
                random_dataset.get(i, &point);
                point.SetRandom(0.0, 12.0, 0.5);
              }
              data->Purge(random_dataset);
              data->Detach(random_dataset);

              // Call the KPCA driver.
              fl::ml::GreedyKernelPca < boost::mpl::void_, do_centering
              >::template Core<TableType1>::Main(
                data, vm);

              return 0;
            }
        };

      public:

        KernelPcaTest(const std::vector<std::string> &point_types_in) {
          point_types_ = point_types_in;
        }

        void RunTests() {

          // For each point type, run the test.
          for (int i = 0; i < point_types_.size(); i++) {

            fl::logger->Message() << "\nTesting the point type: " <<
            point_types_[i];

            for (int trial = 0; trial < 10; trial++) {
              fl::table::FileDataAccess data;
              boost::program_options::variables_map vm;

              // Construct the argument list for the current trial.
              std::vector< std::string > args;
              ConstructArgsForTrial_(data, point_types_[i], &args);

              // Construct the boost variable map.
              fl::ml::GreedyKernelPca < boost::mpl::void_, do_centering
              >::ConstructBoostVariableMap(args, &vm);

              fl::table::Branch::BranchOnTable <
              KernelPcaTest, fl::table::FileDataAccess > (
                &data, vm);

            } // end of running the trials.
          } // end of running over each point type.
        }
    };

  public:

    KernelPcaTestSuite(const std::vector<std::string> &point_types_in)
        : boost::unit_test_framework::test_suite("Kpca test suite") {

      boost::shared_ptr< KernelPcaTest > instance(
        new KernelPcaTest(point_types_in));

      // Create the test cases.
      boost::unit_test_framework::test_case *kpca_test_case
      = BOOST_CLASS_TEST_CASE(
          &KernelPcaTest::RunTests, instance);

      // Add the test cases to the test suite.
      add(kpca_test_case);
    }
};

boost::unit_test_framework::test_suite*
init_unit_test_suite(int argc, char** argv) {

  // Seed the random number.
  srand(time(NULL));

  // create the top test suite
  boost::unit_test_framework::test_suite* top_test_suite
  = BOOST_TEST_SUITE("Kpca tests");

  // Turn on the logger.
  fl::Logger::SetLogger(std::string("verbose"));

  // Point types to test.
  std::vector< std::string > point_types;
  point_types.push_back(std::string("dense"));
  point_types.push_back(std::string("sparse"));
  point_types.push_back(std::string("dense_sparse"));
  point_types.push_back(std::string("categorical"));
  point_types.push_back(std::string("dense_categorical"));

  top_test_suite->add(new KernelPcaTestSuite<true>(point_types));
  return top_test_suite;
}
