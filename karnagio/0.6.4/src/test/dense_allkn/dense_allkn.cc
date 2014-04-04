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
 * @file allkn_test.cc
 *
 * A "stress" test driver for all-nearest neighbor furthest and
 * nearest neighbors, both monochromatic and bichromatic.
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include <time.h>
#include <sys/time.h>
#include <algorithm>
#include <string>
#include <stdexcept>
#include "boost/test/unit_test.hpp"
#include "allkn_test.h"

 
BOOST_AUTO_TEST_SUITE(TestSuiteAllKFN)
BOOST_AUTO_TEST_CASE(TestCaseAllKFN) {

  typedef boost::mpl::vector1_c< int, 0 > dataset_type_options;
  typedef boost::mpl::vector1_c< bool, false > use_range_cut_off_options;
  typedef boost::mpl::vector1_c< bool, true > use_dualtree_options;
  typedef boost::mpl::vector1_c< bool, false> is_progressive_options;
  typedef boost::mpl::vector1_c< bool, false > sort_points_options;
  typedef boost::mpl::vector1_c< int, 1 > tree_type_options;
  typedef boost::mpl::vector1_c< int, 0> query_type_options;

  fl::Logger::SetLogger(std::string("verbose"));

  // Call the tests.
  TemplateRecursion7<
    dataset_type_options,
    use_range_cut_off_options,
    use_dualtree_options,
    is_progressive_options,
    sort_points_options,
    tree_type_options,
    query_type_options
  >::BuildMap();

  printf("\nAll tests passed!\n");
}
BOOST_AUTO_TEST_SUITE_END()
