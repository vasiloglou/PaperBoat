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
#include "boost/mpl/if.hpp"
#include "fastlib/base/base.h"
#include "fastlib/table/default_table.h"


class TableTest {
    // struct DatasetArgs : public fl::data::DatasetArgs {
    //   typedef boost::mpl::vector1<double> DenseTypes;
    //   typedef double CalcPrecision;
    // };

    // typedef fl::data::MultiDataset<DatasetArgs> MyDataset;
    // struct TreeArgs1 : public  fl::tree::TreeArgs {
    //   typedef boost::mpl::bool_<true> SortPoints;
    //   typedef boost::mpl::bool_<true> StoreLevel;
    //   typedef fl::tree::MidpointKdTree TreeSpecType;
    //   typedef fl::tree::GenHrectBound<double, double, 2> BoundType;
    // };

    // struct TableArgs1 {
    //   typedef MyDataset DatasetType;
    //   typedef boost::mpl::bool_<true> SortPoints;
    // };

    // struct TemplateMap1 {
    //   typedef TreeArgs1 TreeArgs;
    //   typedef TableArgs1 TableArgs;
    // };
  typedef fl::table::DefaultTableMap TemplateMap1;

    void TestIndexedVsUnindexedTable() {

      BOOST_MESSAGE("Testing table");
      typedef fl::tree::Tree<TemplateMap1> Tree_t;
      typedef fl::table::Table<TemplateMap1> Table1_t;
      Table1_t table1;
      Table1_t::IndexArgs<fl::math::LMetric<2> > index_args;
      index_args.leaf_size = 20;
      table1.Init(input_files_directory_ + "/test_data_3_1000.csv", "r");
      table1.IndexData(index_args);

      typedef fl::table::Table<TemplateMap1> Table2_t;
      Table2_t table2;
      table2.Init(input_files_directory_ + "/test_data_3_1000.csv", "r");
      BOOST_ASSERT(table1.n_attributes() == table2.n_attributes());
      BOOST_ASSERT(table1.n_entries() == table2.n_entries());
      for (index_t i = 0; i < table1.n_entries(); i++) {
        Table1_t::Point_t point1;
        Table2_t::Point_t point2;
        table1.get(i, &point1);
        table2.get(i, &point2);
        for (index_t j = 0; j < table1.n_attributes(); j++) {
          BOOST_ASSERT(point1[j] == point2[j]);
        }
      }
      table1.DeleteIndex();
      for (index_t i = 0; i < table1.n_entries(); i++) {
        Table1_t::Point_t point1;
        Table2_t::Point_t point2;
        table1.get(i, &point1);
        table2.get(i, &point2);
        for (index_t j = 0; j < table1.n_attributes(); j++) {
          BOOST_ASSERT(point1[j] == point2[j]);
        }
      }
      BOOST_MESSAGE("Finished");
    }

  public:

    TableTest(std::string input_files_dir_in) {
      input_files_directory_ = input_files_dir_in;
    }


    void RunTests() {
      TestIndexedVsUnindexedTable();
    }

  private:

    std::string input_files_directory_;

};

class TableTestSuite : public boost::unit_test_framework::test_suite {
  public:
    TableTestSuite(std::string input_files_dir_in)
        : boost::unit_test_framework::test_suite("Table test suite") {
      // create an instance of the test cases class
      boost::shared_ptr<TableTest> instance(new TableTest(input_files_dir_in));
      // create the test cases
      boost::unit_test_framework::test_case* table_test_case
      = BOOST_CLASS_TEST_CASE(&TableTest::RunTests, instance);
      // add the test cases to the test suite
      add(table_test_case);
    }
};


boost::unit_test_framework::test_suite*
init_unit_test_suite(int argc, char** argv) {
  // create the top test suite
  boost::unit_test_framework::test_suite* top_test_suite
  = BOOST_TEST_SUITE("Table tests");
  if (argc != 2) {
    NOTIFY("Wrong number of arguments for table test. Expected test input files directory. Returning NULL.");
    return NULL;
  }
  // add test suites to the top test suite
  std::string input_files_directory = argv[1];
  top_test_suite->add(new TableTestSuite(input_files_directory));
  return top_test_suite;
}



