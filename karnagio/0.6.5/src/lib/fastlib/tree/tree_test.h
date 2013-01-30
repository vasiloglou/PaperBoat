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
#include <iostream>
#include "boost/program_options.hpp"
#include "boost/test/unit_test.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/if.hpp"
#include "fastlib/base/base.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/table/file_data_access.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/tree/cart_impurity.h"
#include "fastlib/tree/classification_decision_tree.h"
#include "fastlib/tree/kdtree.h"
#include "fastlib/tree/metric_tree.h"
#include "fastlib/tree/similarity_tree.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "fastlib/metric_kernel/cosine_premetric.h"
#include "fastlib/table/default/dense/labeled/balltree/table.h"
#include "fastlib/table/default/dense/labeled/kdtree/table.h"
#include "fastlib/tree/tree_test_private.h"
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"

#ifndef FL_LITE_FASTLIB_TREE_TREE_TEST_H
#define FL_LITE_FASTLIB_TREE_TREE_TEST_H

namespace fl {
namespace tree {
namespace tree_test {
class TreeTestSuite : public boost::unit_test_framework::test_suite {

  public:

    /** @brief The definition of the table that uses the CART tree.
     */
    struct CartTableMap {
      struct TableArgs {
        struct DatasetArgs : public fl::data::DatasetArgs {
          typedef boost::mpl::vector1<double> DenseTypes;
          typedef boost::mpl::vector3<double, int, unsigned char> SparseTypes;
          typedef fl::MakeIntIndexedStruct <
          boost::mpl::vector3<signed char, double, int>
          >::Generated MetaDataType;
          typedef double CalcPrecision;
          typedef fl::data::DatasetArgs::Extendable StorageType;
        };
        typedef fl::data::MultiDataset<DatasetArgs> DatasetType;
        typedef boost::mpl::bool_<true> SortPoints;
      };
      struct TreeArgs : public fl::tree::TreeArgs {
        typedef fl::tree::ClassificationDecisionTree TreeSpecType;
        typedef fl::tree::CartBound<double, double, 2, signed char> BoundType;
        typedef boost::mpl::bool_<true> SortPoints;
      };
    };

    /** @brief The definition of the default ball tree table map.
     */
    typedef fl::table::dense::labeled::balltree::TableMap BalltreeTableMap;
    /** @brief The definition of the default ball tree table map.
     */
    typedef fl::table::dense::labeled::bregmantree::TableMap BregmantreeTableMap; 
    /** @brief The definition of the default midpoint kdtree table map.
     */
    typedef fl::table::dense::labeled::kdtree::TableMap MidpointKdtreeTableMap;


  public:

    template<typename TableMap, typename MetricType>
    class TreeTest {
      private:

        std::string input_file_dir_;

        std::string tree_type_;

        std::vector< std::string > point_types_;

      private:

        class PushRandomSize {
          private:
            std::vector<int> *sizes_;

          public:
            PushRandomSize(std::vector<int> &sizes_in) {
              sizes_ = &sizes_in;
            }

            template<typename T>
            void operator()(T) {
              int num_dimensions = fl::math::Random<int>(5, 20);
              fl::logger->Message() << "Generated " << num_dimensions <<
              " dimensions for " << typeid(T).name();
              sizes_->push_back(num_dimensions);
            }
        };

        void ConstructBoostVariableMap_(
          const std::string &point_type,
          const std::string &tree_type,
          boost::program_options::variables_map *vm) {

          boost::program_options::options_description desc("Available options");
          desc.add_options()
          (
            "point",
            boost::program_options::value<std::string>()->default_value("dense"),
            "Point type used by the test.  One of:\n"
            "  dense, sparse, dense_sparse, categorical, dense_categorical"
          )
          (
            "tree",
            boost::program_options::value<std::string>()->default_value("kdtree"),
            "Tree structure used by the test.  One of:\n"
            "  kdtree, balltree"
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

          std::vector< std::string > args;
          std::string point_arg = std::string("--point=") + point_type;
          std::string tree_arg = std::string("--tree=") + tree_type;

          args.push_back(point_arg);
          args.push_back(tree_arg);
          boost::program_options::store(
            boost::program_options::command_line_parser(args).options(desc).run(), *vm);
          boost::program_options::notify(*vm);
        }

      public:

        template<typename TableType1>
        class Core {

          public:

            /** @brief The template map of the real table that enables
             *         work around for testing CART trees.
             */
            struct RealTableMap {
              struct TableArgs {
                struct DatasetArgs : public fl::data::DatasetArgs {
                  typedef typename TableType1::Point_t::DenseTypes_t DenseTypes;
                  typedef typename TableType1::Point_t::SparseTypes_t SparseTypes;
                  typedef typename TableType1::Point_t::MetaData_t MetaDataType;
                  typedef typename TableType1::CalcPrecision_t CalcPrecision;
                  typedef typename TableMap::TableArgs::DatasetArgs::StorageType StorageType;
                };
                typedef typename TableType1::Dataset_t DatasetType;
                typedef typename TableMap::TableArgs::SortPoints SortPoints;
              };
              struct TreeArgs : public fl::tree::TreeArgs {
                typedef typename TableMap::TreeArgs::TreeSpecType TreeSpecType;
                typedef typename fl::tree::tree_test_private::BoundExtractor < 
                  typename TableMap::TreeArgs::BoundType,
                typename TableType1::Dataset_t::Point_t >::BoundType BoundType;
                typedef typename TableMap::TreeArgs::SortPoints SortPoints;
              };
            };
            typedef fl::table::Table<RealTableMap> RealTableType;

            template<typename DataAccessType>
            static int Main(DataAccessType *data,
                            boost::program_options::variables_map &vm) {

              typename RealTableType::template IndexArgs< MetricType > index_args;
              index_args.leaf_size = 20;

              // Declare the object used for the real test.
              TreeTest<RealTableMap, MetricType> real_test;

              // Generate a random table.
              RealTableType random_dataset;
              real_test.RandomDataset(*data, &random_dataset);

              // Temporarily write the datasetto the file, and re-read it
              // into another copy.
              data->Purge(random_dataset);
              data->Detach(random_dataset);
              RealTableType random_dataset_indexed;
              RealTableType random_dataset_original;
              data->Attach(std::string("random.csv"), &random_dataset_original);
              data->Attach(std::string("random.csv"), &random_dataset_indexed);

              // Index the tree.
              random_dataset_indexed.IndexData(index_args);

              // Check the table integrity after building the tree.
              real_test.TableIntegrityTest(random_dataset_original,
                                           random_dataset_indexed);
              // test serialization
              fl::logger->Message() << "Testing serialization" << std::endl;
              {
                std::ofstream ofs("filename");
                boost::archive::text_oarchive oa(ofs);
                oa << random_dataset_indexed;
              }
              {
                RealTableType new_random_dataset_indexed;
                std::ifstream ifs("filename");
                boost::archive::text_iarchive ia(ifs);
                ia >> new_random_dataset_indexed;
                real_test.TableIntegrityTest(random_dataset_original,
                                             new_random_dataset_indexed);
                // Check the tree integrity.
                if (new_random_dataset_indexed.get_tree() ==
                    (typename RealTableType::Tree_t *) NULL) {
                  fl::logger->Die() << "The tree is NULL";
                }
                fl::logger->Message() << "Checking the integrity of the tree";
                real_test.TreeIntegrityTest(index_args.metric,
                                            new_random_dataset_indexed,
                                            new_random_dataset_indexed.get_tree());
              }

              // Check the tree integrity.
              if (random_dataset_indexed.get_tree() ==
                  (typename RealTableType::Tree_t *) NULL) {
                fl::logger->Die() << "The tree is NULL";
              }
              fl::logger->Message() << "Checking the integrity of the tree";
              real_test.TreeIntegrityTest(index_args.metric,
                                          random_dataset_indexed,
                                          random_dataset_indexed.get_tree());

              return 0;
            }
        };

      public:

        template<typename DataAccessType, typename TableType>
        void RandomDataset(DataAccessType &data, TableType *table_out) {

          // Generate the random number of points and the dimensionality.
          int num_points = fl::math::Random(300, 1000);

          // For each dense type, generate a vector of random sizes.
          fl::logger->Message() << "Generating dense dimensions:";
          std::vector<int> dense_sizes;
          boost::mpl::for_each <
          typename TableType::Point_t::DenseTypes_t > (
            PushRandomSize(dense_sizes));

          // For each sparse type, generate a vector of random sizes.
          std::vector<int> sparse_sizes;
          fl::logger->Message() << "Generating sparse dimensions";
          boost::mpl::for_each <
          typename TableType::Point_t::SparseTypes_t > (
            PushRandomSize(sparse_sizes));

          data.Attach(std::string("random.csv"),
                      dense_sizes, sparse_sizes, num_points, table_out);
          for (int i = 0; i < num_points; i++) {
            typename TableType::Point_t point;
            table_out->get(i, &point);
            point.SetRandom(0.0, 12.0, 0.5);

            // Set the meta data label.
            point.meta_data().template get<0>() =
              (signed char) fl::math::Random(0, 10);
          }
        }

        template<typename TableType>
        void TreeIntegrityTest(const MetricType &metric,
                               const TableType &table,
                               typename TableType::Tree_t *node) {

          // Test the bound integrity of the current node.
          typename fl::tree::tree_test_private::TestBoundIntegrity <
          typename TableType::Tree_t::Bound_t > (metric, table, node);

          // Test the partition integrity of the current node.
          typename fl::tree::tree_test_private::TestPartitionIntegrity <
          typename TableType::Tree_t::Bound_t > (metric, table, node);

          if (table.node_is_leaf(node) == false) {
            TreeIntegrityTest(metric, table,
                              table.get_node_left_child(node));
            TreeIntegrityTest(metric, table,
                              table.get_node_right_child(node));
          }
        }

        template<typename TableType>
        bool TableIntegrityTest(const TableType &original_table,
                                const TableType &indexed_table) {

          fl::logger->Message() << "Checking the integrity of the table "
          "before and after building the tree.";

          // Compare both datasets on a point basis.
          for (int i = 0; i < indexed_table.n_entries(); i++) {
            typename TableType::Point_t point;
            typename TableType::Point_t original_point;
            indexed_table.get(i, &point);
            original_table.get(i, &original_point);
            for (int j = 0; j < point.length(); j++) {
              if (point[j] != original_point[j]) {
                fl::logger->Warning() << i <<
                "-th point's" << j << "-th dimensions is different: should be " <<
                original_point[j] << " but it is " << point[j] << "\n";
                fl::logger->Die() << "The tree building has a fatal bug!";
                return false;
              }
            }
          }
          return true;
        }

        TreeTest() {
        }

        TreeTest(const std::string &input_file_dir_in,
                 const std::string &tree_type_in,
                 const std::vector<std::string> &point_types_in) {
          input_file_dir_ = input_file_dir_in;
          tree_type_ = tree_type_in;
          point_types_ = point_types_in;
        }

        void RunTests() {

          fl::logger->Message() << "\nTesting the tree type: " << tree_type_;

          // This is to by-pass the current branch on table's limitation.
          if (tree_type_ != "kdtree" || tree_type_ != "balltree") {
            tree_type_ = "balltree";
          }

          // For each point type,
          for (int i = 0; i < point_types_.size(); i++) {

            fl::logger->Message() << "\nTesting the point type: " <<
            point_types_[i];

            // We repeat trials.
            for (int trial = 0; trial < 10; trial++) {
              fl::table::FileDataAccess data;
              boost::program_options::variables_map vm;

              // Construct the boost variable map.
              ConstructBoostVariableMap_(point_types_[i], tree_type_, &vm);
              TreeTest<TableMap, MetricType>::
                Core<fl::table::dense::labeled::kdtree::Table>::Main(&data, vm);
             // fl::table::Branch::BranchOnTable<
             //     TreeTest<TableMap, MetricType>, 
             //     fl::table::FileDataAccess>(&data, vm);
            }
          }
        }
    };

  public:

    template<typename TableMap, typename MetricType>
    TreeTestSuite(std::string &input_files_dir_in,
                  const std::string &tree_type_in,
                  const std::vector<std::string> &point_types_in,
                  const fl::table::Table<TableMap> &dummy_table,
                  const MetricType &dummy_metric)
        : boost::unit_test_framework::test_suite("Tree test suite") {

      // create an instance of the test cases class for CART
      typedef TreeTest<TableMap, MetricType> TestType;
      boost::shared_ptr< TestType > instance(
        new TestType(input_files_dir_in, tree_type_in, point_types_in));

      // Create the test cases.
      boost::unit_test_framework::test_case* tree_test_case
      = BOOST_CLASS_TEST_CASE(
          &TestType::RunTests, instance);
      // add the test cases to the test suite
      add(tree_test_case);
    }
};
};
};
};

#endif
