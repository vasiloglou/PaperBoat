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
#include "mlpack/bnneighbors/branch_on_table_dev.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/util/timer.h"
#include "fastlib/metric_kernel/kl_divergence.h"
#include "fastlib/metric_kernel/hellinger_metric.h"
#include "mlpack/bnneighbors/bnneighbors_defs.h"
#include "mlpack/allkn/allkn_dev.h"

class BregmanNNTestSuite : public boost::unit_test_framework::test_suite {

  private:

    class BregmanNNTest {
      private:

        std::vector<std::string> point_types_;

        std::vector<std::string> algorithm_types_;

        std::vector<std::string> divergence_types_;

        void ConstructArgsForTrial_(const fl::table::FileDataAccess &data,
                                    const std::string &point_type,
                                    const std::string &algorithm_type,
                                    const std::string &divergence_type,
                                    std::vector<std::string> *args_out) {

          // Construct the string arguments for the driver.
          std::string divergence_arg = std::string("--divergence=") +
                                       divergence_type;
          std::string point_arg = std::string("--point=") + point_type;
          std::string tree_arg = std::string("--tree=balltree");
          std::string references_arg =
            std::string("--references_in=random.csv");
          std::string indices_out_arg =
            std::string("--indices_out=indices.csv");
          std::string distances_out_arg =
            std::string("--distances_out=distances.csv");
          std::string leaf_size_arg = std::string("--leaf_size=20");
          std::string k_neighbors_arg =
            std::string("--k_neighbors=1");
          std::string algorithm_arg =
            std::string("--algorithm=") + algorithm_type;

          // Push in the arguments.
          args_out->push_back(divergence_arg);
          args_out->push_back(point_arg);
          args_out->push_back(tree_arg);
          args_out->push_back(references_arg);
          args_out->push_back(indices_out_arg);
          args_out->push_back(distances_out_arg);
          args_out->push_back(leaf_size_arg);
          args_out->push_back(k_neighbors_arg);
          args_out->push_back(algorithm_arg);
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

            template<typename TableType>
            static void NaiveBregmanNN(
              fl::table::FileDataAccess &data,
              const TableType &table) {

              // Read in the indices and the distances computed from
              // the tree-based algorithm.
              typename fl::table::FileDataAccess::UIntegerTable_t indices_table;
              data.Attach(std::string("indices.csv"), &indices_table);
              typename fl::table::FileDataAccess::DefaultTable_t distances_table;
              data.Attach(std::string("distances.csv"), &distances_table);

              fl::math::KLDivergence kl_divergence;
              for (int i = 0; i < table.n_entries(); i++) {
                typename TableType::Point_t query_point;
                table.get(i, &query_point);
                int min_index = -1;
                double min_divergence = std::numeric_limits<double>::max();
                for (int j = 0; j < table.n_entries(); j++) {
                  typename TableType::Point_t reference_point;
                  table.get(j, &reference_point);
                  if (i == j) {
                    continue;
                  }
                  double divergence = kl_divergence.Divergence(
                                        reference_point, query_point);
                  if (divergence < min_divergence) {
                    min_index = j;
                    min_divergence = divergence;
                  }
                }
                typename fl::table::FileDataAccess::UIntegerTable_t::Point_t index_point;
                indices_table.get(i, &index_point);
                typename fl::table::FileDataAccess::DefaultTable_t::Point_t distance_point;
                distances_table.get(i, &distance_point);

                if (min_index != index_point[0] &&
                    fabs(min_divergence - distance_point[0]) > 1e-6) {

                  typename TableType::Point_t naive_candidate_point;
                  typename TableType::Point_t tree_candidate_point;
                  table.get(min_index, &naive_candidate_point);
                  table.get(index_point[0], &tree_candidate_point);
                  double naive_candidate_divergence =
                    kl_divergence.Divergence(
                      naive_candidate_point, query_point);
                  double tree_candidate_divergence =
                    kl_divergence.Divergence(tree_candidate_point, query_point);

                  if (fabs(naive_candidate_divergence -
                           tree_candidate_divergence) >= 1e-4) {
                    printf("For query %d: Naive: %g %d vs tree: %g %d\n", i,
                           min_divergence, min_index,
                           distance_point[0], index_point[0]);
                    printf("The naive candidate should have divergence "
                           "of %g.\n", naive_candidate_divergence);
                    printf("The tree candidate should have divergence of %g.\n",
                           tree_candidate_divergence);
                    throw std::runtime_error("The divergence doesn't match.");
                  }
                }
              }
            }

            static int Main(fl::table::FileDataAccess *data,
                            boost::program_options::variables_map &vm) {

              // Generate a random dataset.
              int num_points = fl::math::Random(1000, 2001);
              int num_dense_dimensions = fl::math::Random(2, 4);
              int num_sparse_dimensions = fl::math::Random(2, 5);

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

                // Make sure that each coordinate is positive.
                point.SetRandom(1.0, 12.0, 0.0);
              }
              data->Purge(random_dataset);
              data->Detach(random_dataset);

              fl::logger->Message() << "Testing on the dataset of dimension: "
              << random_dataset.n_attributes() <<
              " containing " << random_dataset.n_entries() <<
              " points.";
              // Call the Bregman NN driver.
              fl::ml::BNNeighbors::template Core<TableType1>::Main(
                data, vm);

              // Run the naive algorithm.
              fl::util::Timer timer;
              timer.Start();

              // Normalize the data before doing naive.
              fl::ml::BNNeighbors::template Core<TableType1>::Normalize(
                &random_dataset);

              NaiveBregmanNN(*data, random_dataset);
              timer.End();
              fl::logger->Message() << "Took " <<
              timer.GetTotalElapsedTime() << " seconds.";

              return 0;
            }
        };

      public:

        BregmanNNTest(
          const std::vector<std::string> &point_types_in,
          const std::vector<std::string> &algorithm_types_in,
          const std::vector<std::string> &divergence_types_in) {

          point_types_ = point_types_in;
          algorithm_types_ = algorithm_types_in;
          divergence_types_ = divergence_types_in;
        }

        void ConstructBoostVariableMap(
          const std::vector< std::string > &args,
          boost::program_options::variables_map *vm) {

          boost::program_options::options_description desc("Available options");
          desc.add_options()(
            "help", "Print this information."
          )(
            "references_in",
            boost::program_options::value<std::string>(),
            "REQUIRED file containing reference data"
          )(
            "references_out",
            boost::program_options::value<std::string>()->default_value(""),
            "OPTIONAL file where the references_in data will be serialized. You can"
            " use this option to save your data after they have been indexed with a tree."
            " Then you can reuse them for this or another algorithm without having to"
            " rebuild the tree"
          )(
            "queries_in",
            boost::program_options::value<std::string>()->default_value(""),
            "OPTIONAL file containing query positions.  If omitted, allkn "
            "finds leave-one-out neighbors for each reference point."
          )(
            "queries_out",
            boost::program_options::value<std::string>()->default_value(""),
            "OPTIONAL file where the queries_in data will be serialized. You can"
            " use this option to save your data after they have been indexed with a tree."
            " Then you can reuse them for this or another algorithm without having to"
            " rebuild the tree"
          )(
            "serialize",
            boost::program_options::value<std::string>()->default_value("binary"),
            "OPTIONAL when you serialize the tables to files, you have the option to use:\n "
            "  binary for smaller files\n"
            "  text   for portability  (bigger files)\n"
            "  xml    for interpretability and portability (even bigger files)"
          )(
            "indices_out",
            boost::program_options::value<std::string>()->default_value(""),
            "OPTIONAL file to store found neighbor indices"
          )(
            "distances_out",
            boost::program_options::value<std::string>()->default_value(""),
            "OPTIONAL file to store found neighbor distances"
          )(
            "distances_in",
            boost::program_options::value<std::string>()->default_value(""),
            "OPTIONAL file containing distances between neighboring points. "
            "if flag --method=classification then you should provide --distances_in"
          )(
            "k_neighbors",
            boost::program_options::value<index_t>()->default_value(-1),
            "The number of neighbors to find for the all-k-neighbors method."
          )(
            "r_neighbors",
            boost::program_options::value<double>()->default_value(-1.0),
            "The query radius for the all-range-neighbors method.\n"
            "One of --k_neighbors or --r_neighbors must be given."
          )(
            "point",
            boost::program_options::value<std::string>()->default_value("dense"),
            "Point type used by allkn.  One of:\n"
            "  dense, sparse, dense_sparse, categorical, dense_categorical"
          )(
            "divergence",
            boost::program_options::value<std::string>()->default_value("kl"),
            "Divergence function used by allkn.  One of:\n"
            "  kl : Kullback–Leibler divergence (\\sum x_i \\log x_i/y_i), data must be "
            "  between zero and one and ||x||_1=1, ||y||_1=1. KL divergence is not valid for sparse data\n"
            "  hl : Hellinger divergence (\\sqrt(1-<x,y)/\\sqrt(1-||y||^2)-\\sqrt(1-||x||^2)), "
            "  same restrictions apply for data. This option is valid for sparse \n"
            "  is : itakura-saito (\\sum(x_i/y_i-\\log(x_i/y_i)-1)), valid only for dense data"

          )(
            "normalize",
            boost::program_options::value<bool>()->default_value(true),
            " For divergences the data must be normalized and behave like probabilities."
            " In short every point must have nonnegative entries and the L1 norm must be 1."
            " If your data do not satisfy these conditions, the results are unpredicctable."
            " We strongly recommend you set this flag to true if you are not sure about the"
            " nature of your data."
          )(
            "algorithm",
            boost::program_options::value<std::string>()->default_value("dual"),
            "Algorithm used to compute densities.  One of:\n"
            "  dual, single"
          )(
            "tree",
            boost::program_options::value<std::string>()->default_value("balltree"),
            "Tree structure used by allkn.  One of:\n"
            "  balltree"
          )(
            "leaf_size",
            boost::program_options::value<index_t>()->default_value(20),
            "Maximum number of points at a leaf of the tree.  More saves on tree "
            "overhead but too much hurts asymptotic run-time."
          )(
            "iterations",
            boost::program_options::value<index_t>()->default_value(-1),
            "Allkn can run in either batch or progressive mode.  If --iterations=i "
            "is omitted, allkn finds exact neighbors; otherwise, it terminates after "
            "i progressive refinements."
          )(
            "log_tree_stats",
            boost::program_options::value<bool>()->default_value(false),
            "If this flag is set true then it outputs some statistics about the tree after it is built. "
            "We suggest you set that flag on. If the tree is not correctly built, due to wrong options"
            " or due to pathological data then there is not point in running nearest neighbors"
          )(
            "cores",
            boost::program_options::value<int>()->default_value(1),
            "Number of cores to use for running the algorithm. If you use large number of cores "
            "increase the leaf_size, This feature is disabled for the moment"
          )(
            "log",
            boost::program_options::value<std::string>()->default_value(""),
            "A file to receive the log, or omit for stdout."
          )(
            "loglevel",
            boost::program_options::value<std::string>()->default_value("debug"),
            "Level of log detail.  One of:\n"
            "  debug: log everything\n"
            "  verbose: log messages and warnings\n"
            "  warning: log only warnings\n"
            "  silent: no logging"
          );

          boost::program_options::command_line_parser clp(args);
          clp.style(boost::program_options::command_line_style::default_style
                    ^ boost::program_options::command_line_style::allow_guessing);
          boost::program_options::store(clp.options(desc).run(), *vm);
        }

        void RunTests() {

          // For each point type, run the test.
          for (int i = 0; i < point_types_.size(); i++) {

            fl::logger->Message() << "\nTesting the point type: " <<
            point_types_[i];

            for (int j = 0; j < algorithm_types_.size(); j++) {

              fl::logger->Message() << "\nTesting the algorithm type: " <<
              algorithm_types_[j];

              for (int k = 0; k < divergence_types_.size(); k++) {

                for (int trial = 0; trial < 10; trial++) {
                  fl::table::FileDataAccess data;
                  boost::program_options::variables_map vm;

                  // Construct the argument list for the current trial.
                  std::vector< std::string > args;
                  ConstructArgsForTrial_(data,
                                         point_types_[i],
                                         algorithm_types_[j],
                                         divergence_types_[k], &args);

                  // Construct the boost variable map.
                  ConstructBoostVariableMap(args, &vm);

                  fl::table::BranchBNNeighbors::BranchOnTable <
                  BregmanNNTest, fl::table::FileDataAccess > (
                    &data, vm);

                } // end of running the trials.
              } // end of running over each divergence type.
            } // end of running over each algorithm type.
          } // end of running over each point type.
        }
    };

  public:

    BregmanNNTestSuite(
      const std::vector<std::string> &point_types_in,
      const std::vector<std::string> &algorithm_types_in,
      const std::vector<std::string> &divergence_types_in)
        : boost::unit_test_framework::test_suite("Bregman NN test suite") {

      boost::shared_ptr< BregmanNNTest > instance(
        new BregmanNNTest(
          point_types_in, algorithm_types_in, divergence_types_in));

      // Create the test cases.
      boost::unit_test_framework::test_case *bregman_nn_test_case
      = BOOST_CLASS_TEST_CASE(
          &BregmanNNTest::RunTests, instance);

      // Add the test cases to the test suite.
      add(bregman_nn_test_case);
    }
};

boost::unit_test_framework::test_suite*
init_unit_test_suite(int argc, char** argv) {

  // Seed the random number.
  srand(time(NULL));

  // create the top test suite
  boost::unit_test_framework::test_suite* top_test_suite
  = BOOST_TEST_SUITE("Bregman NN tests");

  // Turn on the logger.
  fl::Logger::SetLogger(std::string("verbose"));

  // Point types to test.
  std::vector< std::string > point_types;
  point_types.push_back(std::string("dense"));

  // Algorithm types to test.
  std::vector< std::string> algorithm_types;
  algorithm_types.push_back(std::string("dual"));
  algorithm_types.push_back(std::string("single"));

  // Divergence types to test.
  std::vector<std::string> divergence_types;
  divergence_types.push_back(std::string("kl"));

  top_test_suite->add(
    new BregmanNNTestSuite(point_types, algorithm_types, divergence_types));
  return top_test_suite;
}
