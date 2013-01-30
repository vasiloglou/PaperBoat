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
 * @file ortho_range_search_test.cc
 *
 * A "stress" test driver for the orthogonal range search code.
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include "boost/test/unit_test.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/int.hpp"
#include "fastlib/tree/hyper_rectangle_tree.h"
#include "mlpack/ortho_range_search/ortho_range_search_dev.h"
#include "mlpack/ortho_range_search/ortho_range_search_default_table.h"
#include "fastlib/table/file_data_access.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/tree/tree.h"
#include "fastlib/table/table.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/table/default_sparse_int_table.h"
#include <time.h>
#include <stdexcept>

template<typename Precision, bool sort_window_queries, bool sort_points>
class TestOrthoRangeSearch {

  private:
    struct TableMap {
      struct TableArgs {
        typedef typename fl::table::FileDataAccess::DefaultTable_t::Dataset_t DatasetType;
        typedef boost::mpl::bool_<false> SortPoints ;
      };

      typedef double CalcPrecision_t;
      struct TreeArgs : public fl::tree::TreeArgs {
        typedef fl::tree::HyperRectangleTree TreeSpecType;
        typedef boost::mpl::bool_<false> StoreLevel;
        typedef boost::mpl::bool_<false> SortPoints ;
        typedef fl::tree::HyperRectangleBound<CalcPrecision_t> BoundType;
      };
    };
    typedef fl::table::Table<TableMap>  QueryTable_t;
    typedef fl::table::FileDataAccess::DefaultTable_t ReferenceTable_t;
    struct CoreOrthoArgs {
      typedef QueryTable_t  WindowTableType;
      typedef ReferenceTable_t ReferenceTableType;
    };

    typedef typename CoreOrthoArgs::WindowTableType WindowTable_t;
    typedef fl::table::DefaultSparseIntTable OutputTableType;

    template<typename WindowType, typename PointType>
    static bool Contains_(const WindowType &window, const PointType &point) {
      bool flag = true;
      for (index_t i = 0; flag && i < point.length(); i++) {
        if (point[i] < window[i] || point[i] > window[i + point.length()]) {
          flag = false;
        }
      }
      return flag;
    }

    static bool UltraNaive_(WindowTable_t &window_queries_table,
                            ReferenceTable_t &reference_table,
                            const OutputTableType &compare_results) {

      typename WindowTable_t::TreeIterator window_it =
        window_queries_table.get_node_iterator(
          window_queries_table.get_tree());

      index_t discrepancies = 0;
      do {

        typename WindowTable_t::Dataset_t::Point_t window;
        index_t window_id;
        window_it.Next(&window, &window_id);
        typename ReferenceTable_t::TreeIterator reference_it =
          reference_table.get_node_iterator(reference_table.get_tree());
        do {

          typename ReferenceTable_t::Dataset_t::Point_t rpoint;
          index_t rpoint_id;
          reference_it.Next(&rpoint, &rpoint_id);
          typename OutputTableType::Point_t point;
          compare_results.get(window_id, &point);

          if (Contains_(window, rpoint) != point.get(rpoint_id)) {
            discrepancies++;
          }
        }
        while (reference_it.HasNext());
      }
      while (window_it.HasNext());
      printf("%d discrepancies found!\n", discrepancies);

      return discrepancies == 0;
    }

    static void GenerateRandomDataset_(const index_t &num_dimensions,
                                       const index_t &num_points,
                                       fl::dense::Matrix<Precision, false>
                                       *random_dataset,
                                       fl::dense::Matrix<Precision, false>
                                       *dataset_range) {
      random_dataset->Init(num_dimensions, num_points);
      dataset_range->Init(2, num_dimensions);
      for (index_t j = 0; j < num_points; j++) {
        for (index_t i = 0; i < num_dimensions; i++) {
          random_dataset->set(i, j, fl::math::Random<Precision>(-1.0, 1.0));
        }
      }
      for (index_t j = 0; j < num_dimensions; j++) {
        dataset_range->set(0, j, std::numeric_limits<double>::max());
        dataset_range->set(1, j, -std::numeric_limits<double>::max());
      }
      for (index_t j = 0; j < num_points; j++) {
        for (index_t i = 0; i < num_dimensions; i++) {
          dataset_range->set(0, i, std::min(dataset_range->get(0, i),
                                            random_dataset->get(i, j)));
          dataset_range->set(1, i, std::max(dataset_range->get(1, i),
                                            random_dataset->get(i, j)));
        }
      }
    }

    static void GenerateRandomWindow_(const index_t &num_dimensions,
                                      const index_t &num_windows,
                                      const fl::dense::Matrix<Precision, false>
                                      &dataset_range,
                                      fl::dense::Matrix<Precision, false>
                                      *random_windows) {
      random_windows->Init(2 * num_dimensions, num_windows);
      for (index_t j = 0; j < num_windows; j++) {

        // Throw a coin to decide whether to duplicate the previous
        // window and perturb it.
        bool duplicate = ((fl::math::Random<double>() >= 0.4) && j > 0);
        if (duplicate) {
          for (index_t i = 0; i < num_dimensions; i++) {
            double previous_window_width =
              random_windows->get(i + num_dimensions, j - 1) -
              random_windows->get(i, j - 1);
            bool move_lower_bound = (fl::math::Random<double>() >= 0.5);
            bool move_upper_bound = (fl::math::Random<double>() >= 0.5);
            double lower_bound_sign = (move_lower_bound) ? 1.0 : -1.0;
            double upper_bound_sign = (move_upper_bound) ? 1.0 : -1.0;
            random_windows->set(i, j, random_windows->get(i, j - 1) +
                                lower_bound_sign * 0.01 *
                                previous_window_width);
            random_windows->set(i + num_dimensions, j,
                                random_windows->get(i + num_dimensions, j - 1)
                                + upper_bound_sign * 0.01 *
                                previous_window_width);
          }
        }
        else {

          // Generate a new window.
          for (index_t i = 0; i < num_dimensions; i++) {

            // Generate the lower bound coordinate.
            double factor = fl::math::Random<double>(0.01, 0.05);
            double length = dataset_range.get(1, i) - dataset_range.get(0, i);
            random_windows->set(i, j, fl::math::Random<double>
                                (dataset_range.get(0, i),
                                 dataset_range.get(1, i)));
            random_windows->set(i + num_dimensions, j,
                                random_windows->get(i, j) + factor * length);
          }
        }
      }
    }

  public:

    static void Init() {
      srand(time(NULL));
    }

    static void FinalMessage() {
      printf("\nAll tests passed!\n");
    }

    static void StressTest() {

      printf("StressTest(): Beginning\n\n");

      if (sort_window_queries) {
        printf("Window tree is sorted, ");
      }
      else {
        printf("Window tree is not sorted, ");
      }
      if (sort_points) {
        printf("Points are sorted.\n");
      }
      else {
        printf("Points are not sorted.\n");
      }

      for (index_t i = 0; i < 5; i++) {

        index_t window_leaf_size = fl::math::Random(16, 33);
        index_t reference_leaf_size = fl::math::Random(16, 33);
        fl::dense::Matrix<Precision, false> window_queries;
        fl::dense::Matrix<Precision, false> reference_points;
        fl::dense::Matrix<Precision, false> dataset_range;
        OutputTableType candidate_points;
        index_t num_dimensions = fl::math::Random(2, 7);
        index_t num_points = fl::math::Random(5000, 10001);
        index_t num_windows = fl::math::Random(10, 21);

        // Generate the random dataset and the windows and query!
        GenerateRandomDataset_(num_dimensions, num_points,
                               &reference_points, &dataset_range);
        GenerateRandomWindow_(num_dimensions, num_windows,
                              dataset_range, &window_queries);
        WindowTable_t window_queries_table;
        ReferenceTable_t reference_points_table;
        window_queries_table.Init(window_queries);
        reference_points_table.Init(reference_points);

        candidate_points.Init(
          std::string("output"),
          std::vector<index_t>(),
          std::vector<index_t>(1, reference_points_table.n_entries()),
          window_queries_table.n_entries());

        printf("Generated %d points in %d dimensions.\n",
               num_points, num_dimensions);
        printf("Window dataset: %d features for lower and upper ranges.\n",
               window_queries.n_rows());
        printf("Running the tree-based algorithm.\n");
        clock_t start = clock();
        fl::ml::OrthoRangeSearch < CoreOrthoArgs >::Compute(
          window_queries_table,
          window_leaf_size,
          reference_points_table,
          reference_leaf_size,
          &candidate_points);
        clock_t end = clock();
        printf("Time elapsed: %g seconds.\n",
               ((double)(end - start)) / ((double) CLOCKS_PER_SEC));

        printf("Running the ultra-naive.\n");
        start = clock();
        bool flag = UltraNaive_(window_queries_table, reference_points_table,
                                candidate_points);
        end = clock();
        printf("Time elapsed: %g seconds.\n",
               ((double)(end - start)) / ((double) CLOCKS_PER_SEC));

        if (!flag) {
          throw std::runtime_error("Aborting\n");
        }

      } // end of the trials.

      printf("StressTest(): Passed!\n\n");
    }
};

BOOST_AUTO_TEST_SUITE(TestSuiteOrthoRangeSearch)
BOOST_AUTO_TEST_CASE(TestCaseOrthoRangeSearch) {
  TestOrthoRangeSearch<double, false, false>::Init();
  TestOrthoRangeSearch<double, false, false>::StressTest();
  TestOrthoRangeSearch<double, false, true>::StressTest();
  TestOrthoRangeSearch<double, true, false>::StressTest();
  TestOrthoRangeSearch<double, true, true>::StressTest();
  TestOrthoRangeSearch<double, false, false>::FinalMessage();
}
BOOST_AUTO_TEST_SUITE_END()
