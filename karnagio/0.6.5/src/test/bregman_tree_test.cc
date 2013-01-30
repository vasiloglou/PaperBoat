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
// build it in static mode and use the cmake boost static on
#include "boost/program_options.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/if.hpp"
#include "fastlib/base/base.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/table/file_data_access.h"
#include "fastlib/table/branch_on_table.h"
#include "fastlib/table/table.h"
#include "fastlib/tree/metric_tree.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "fastlib/table/default/dense/labeled/bregmantree/table.h"
#include "fastlib/tree/tree_test.h"
#include "fastlib/metric_kernel/kl_divergence.h"

boost::unit_test_framework::test_suite*
init_unit_test_suite(int argc, char** argv) {

  // create the top test suite
  boost::unit_test_framework::test_suite* top_test_suite
  = BOOST_TEST_SUITE("Ball tree tests");

  if (argc != 2) {
    NOTIFY("Wrong number of arguments for tree test. Expected test input files directory. Returning NULL.");
    return NULL;
  }

  // Turn on the logger.
  fl::Logger::SetLogger(std::string("verbose"));

  // add test suites to the top test suite
  fl::table::dense::labeled::bregmantree::Table dummy_table;
  fl::math::KLDivergence dummy_metric;
  std::string input_files_directory = argv[1];

  // Point types to test.
  std::vector< std::string > point_types;
  point_types.push_back(std::string("dense"));
  point_types.push_back(std::string("sparse"));
  point_types.push_back(std::string("dense_sparse"));
  point_types.push_back(std::string("categorical"));
  point_types.push_back(std::string("dense_categorical"));

  top_test_suite->add(new fl::tree::tree_test::TreeTestSuite(
                        input_files_directory,
                        std::string("bregmantree"),
                        point_types,
                        dummy_table, dummy_metric));
  return top_test_suite;
}
