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
 * @file cart_test.cc
 *
 * A "stress" test driver for the CART.
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include "boost/test/unit_test.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/int.hpp"
#include "fastlib/tree/classification_decision_tree.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/table/file_data_access.h"
#include "fastlib/table/table.h"
#include "fastlib/table/table_dev.h"

class TestCart {
  private:

    struct TableMap {
      struct TableArgs {
        struct DatasetArgs : public fl::data::DatasetArgs {
          typedef boost::mpl::vector1<double> DenseTypes;
          typedef boost::mpl::vector3<double, int, bool> SparseTypes;
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

    typedef fl::table::Table<TableMap> TableType;

  public:

    template<typename DataAccessType>
    void RandomDataset(DataAccessType &data, TableType *table_out) {

      // Generate the random number of points and the dimensionality.
      int num_points = fl::math::Random(300, 1000);
      int num_dense_dimensions = fl::math::Random(10, 20);
      int num_sparse_double_dimensions = fl::math::Random(5, 20);
      int num_sparse_int_dimensions = fl::math::Random(3, 20);
      int num_sparse_bool_dimensions = fl::math::Random(3, 50);

      std::vector<int> sparse_sizes;
      sparse_sizes.resize(3);
      sparse_sizes[0] = num_sparse_double_dimensions;
      sparse_sizes[1] = num_sparse_int_dimensions;
      sparse_sizes[2] = num_sparse_bool_dimensions;

      data.Attach(std::string("random.csv"),
                  std::vector<int>(1, num_dense_dimensions),
                  sparse_sizes, num_points, table_out);
      for (int i = 0; i < num_points; i++) {
        TableType::Point_t point;
        table_out->get(i, &point);
        point.SetRandom(0.0, 12.0, 0.5);

        // Set the meta data label.
        point.meta_data().get<0>() = (signed char) fl::math::Random(0, 10);
      }
    }

    template<typename TreeType>
    void TreeIntegrityTest(const TableType &table, TreeType *node) {

      // Check the class distribution count.
      typename TableType::TreeIterator it = table.get_node_iterator(node);
      typedef typename TreeType::Bound_t::ClassLabel_t
      ClassLabelType;
      const std::map<ClassLabelType, int> &class_counts =
        table.get_node_bound(node).class_counts();
      std::map<ClassLabelType, int> check;

      // Check the total number of points against the class distribution.
      int total_count_check = 0;
      for (typename std::map<ClassLabelType, int>::const_iterator
           class_counts_it = class_counts.begin();
           class_counts_it != class_counts.end();
           class_counts_it++) {
        total_count_check += class_counts_it->second;
      }
      if (total_count_check != table.get_node_count(node)) {
        fl::logger->Die() << "The total number points do not match up!";
      }

      // Now, check the individual class distribution.
      for (; it.HasNext();) {
        typename TableType::Point_t point;
        int point_index;
        it.Next(&point, &point_index);
        if (check.find(point.meta_data().template get<0>()) != check.end()) {
          check[point.meta_data().template get<0>()] += 1;
        }
        else {
          check[point.meta_data().template get<0>()] = 1;
        }
      }
      if (check.size() != class_counts.size()) {
        fl::logger->Die() << "Encountered different number of classes!";
      }
      for (typename std::map<ClassLabelType, int>::const_iterator
           class_counts_it = class_counts.begin();
           class_counts_it != class_counts.end(); class_counts_it++) {
        if (check.find(class_counts_it->first) == check.end() ||
            check[class_counts_it->first] != class_counts_it->second) {
          fl::logger->Die() << "The class count is different!";
        }
      }

      // Check both the left and the right children.
      if (table.node_is_leaf(node) == false) {
        TreeIntegrityTest(table, table.get_node_left_child(node));
        TreeIntegrityTest(table, table.get_node_right_child(node));
      }
    }

    bool TableIntegrityTest(const TableType &original_table,
                            const TableType &indexed_table) {

      // Compare both datasets on a point basis.
      for (int i = 0; i < indexed_table.n_entries(); i++) {
        TableType::Point_t point;
        TableType::Point_t original_point;
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

    void Trial() {

      fl::table::FileDataAccess data;
      fl::logger->Message() << "Begining trial\n";
      TableType::IndexArgs< fl::tree::EntropyImpurity > index_args;
      index_args.leaf_size = 20;
      for (int j = 0; j < 10; j++) {
        TableType random_dataset;
        RandomDataset(data, &random_dataset);

        fl::logger->Message() << "Random dataset containing " <<
        random_dataset.n_entries() << " entries with " <<
        random_dataset.n_attributes() << " attributes" << std::endl;

        // Temporarily write the dataset to the file, and re-read it
        // into another copy.
        data.Purge(random_dataset);
        data.Detach(random_dataset);
        TableType random_dataset_indexed;
        TableType random_dataset_original;
        data.Attach(std::string("random.csv"), &random_dataset_original);
        data.Attach(std::string("random.csv"), &random_dataset_indexed);

        // Index the tree.
        random_dataset_indexed.IndexData(index_args);

        // Test the integrity of the table.
        fl::logger->Message() << "Checking the integrity of the table.";
        TableIntegrityTest(random_dataset_original, random_dataset_indexed);

        // Test the integrity of the tree.
        fl::logger->Message() << "Checking the integirty of the tree.";
        TreeIntegrityTest(random_dataset_indexed,
                          random_dataset_indexed.get_tree());
      }
    }
};

BOOST_AUTO_TEST_SUITE(TestSuiteCart)
BOOST_AUTO_TEST_CASE(TestCaseCart) {
  fl::Logger::SetLogger(std::string("verbose"));
  TestCart test;
  test.Trial();
}
BOOST_AUTO_TEST_SUITE_END()
