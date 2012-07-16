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

#ifndef FL_LITE_MLPACK_ALLKN_ALLKN_TEST_H
#define FL_LITE_MLPACK_ALLKN_ALLKN_TEST_H

#include <omp.h>
#include "boost/mpl/int.hpp"
#include "boost/mpl/bool.hpp"
#include "boost/mpl/or.hpp"
#include "boost/mpl/vector_c.hpp"
#include "mlpack/allkn/allkn.h"
#include "mlpack/allkn/allkn_computations.h"
#include "fastlib/base/mpl.h"
#include "fastlib/tree/kdtree.h"
#include "fastlib/tree/metric_tree.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/tree/tree.h"
#include "fastlib/table/file_data_access.h"
#include "fastlib/table/branch_on_table.h"
#include "fastlib/table/table.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/table/default_table.h"
#include "fastlib/tree/bounds.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "fastlib/metric_kernel/cosine_metric.h"
#include "mlpack/allkn/default_allknn.h"
#include "mlpack/quicsvd/quicsvd_mpl_defs.h"

struct  DatasetArgs : public fl::data::DatasetArgs {
  typedef boost::mpl::vector1<double> DenseTypes;
  typedef fl::data::DatasetArgs::Compact StorageType;
};
typedef fl::data::MultiDataset<DatasetArgs> Dataset;

template<int P, bool sort_points>
struct TableMap; // stub

template<bool sort_points>
struct TableMap<0, sort_points> {
  struct TreeArgs : public fl::tree::TreeArgs {
    typedef boost::mpl::bool_<sort_points> SortPoints;
    typedef boost::mpl::bool_<false> StoreLevel;
    typedef fl::tree::MidpointKdTree TreeSpecType;
    typedef fl::math::LMetric<2> MetricType;
    typedef fl::tree::GenHrectBound<double, double, 2> BoundType;
  };

  struct TableArgs : public fl::table::TableArgs {
    typedef Dataset DatasetType;
    typedef boost::mpl::bool_<sort_points> SortPoints;
  };
};

template<bool sort_points>
struct TableMap<1, sort_points> {
  struct TreeArgs : public fl::tree::TreeArgs {
    typedef boost::mpl::bool_<sort_points> SortPoints;
    typedef boost::mpl::bool_<false> StoreLevel;
    typedef fl::tree::MidpointKdTree TreeSpecType;
    typedef fl::math::LMetric<2> MetricType;
    typedef fl::tree::GenHrectBound<double, double, 2> BoundType;
  };

  struct TableArgs : public fl::table::TableArgs {
    typedef Dataset DatasetType;
    typedef boost::mpl::bool_<sort_points> SortPoints;
  };
};

template<bool sort_points>
struct TableMap<2, sort_points> {
  struct TreeArgs : public fl::tree::TreeArgs {
    typedef boost::mpl::bool_<sort_points> SortPoints;
    typedef boost::mpl::bool_<false> StoreLevel;
    typedef fl::tree::MetricTree TreeSpecType;
    typedef fl::math::LMetric<2> MetricType;
    typedef fl::tree::BallBound<fl::data::MonolithicPoint<double> > BoundType;
  };

  struct TableArgs : public fl::table::TableArgs {
    typedef Dataset DatasetType;
    typedef boost::mpl::bool_<sort_points> SortPoints;
  };
};

template<bool sort_points>
struct TableMap<3, sort_points> {
  struct TreeArgs : public fl::tree::TreeArgs {
    typedef boost::mpl::bool_<sort_points> SortPoints;
    typedef boost::mpl::bool_<false> StoreLevel;
    typedef fl::tree::MetricTree TreeSpecType;
    typedef fl::math::WeightedLMetric <
      2,fl::data::MonolithicPoint<double> > MetricType;
    typedef fl::tree::BallBound<fl::data::MonolithicPoint<double> > BoundType;
  };

  struct TableArgs : public fl::table::TableArgs {
    typedef Dataset DatasetType;
    typedef boost::mpl::bool_<sort_points> SortPoints;
  };
};

template<bool sort_points>
struct TableMap<4, sort_points> {
  struct TreeArgs : public fl::tree::TreeArgs {
    typedef boost::mpl::bool_<sort_points> SortPoints;
    typedef boost::mpl::bool_<false> StoreLevel;
    typedef fl::tree::MetricTree TreeSpecType;
    typedef fl::math::CosineMetric MetricType;
    typedef fl::tree::BallBound<fl::data::MonolithicPoint<double> > BoundType;
  };

  struct TableArgs : public fl::table::TableArgs {
    typedef Dataset DatasetType;
    typedef boost::mpl::bool_<sort_points> SortPoints;
  };
};



template < int dataset_type, bool use_range_cut_off, bool use_dualtree,
bool is_progressive, bool sort_points, int tree_type, int query_type >
class TestAllKN {

  private:

    template<typename Precision>
    static void GenerateRandomDataset_(const index_t &num_dimensions,
                                       const index_t &num_points,
                                       fl::dense::Matrix<Precision, false>
                                       *random_dataset) {
      random_dataset->Init(num_dimensions, num_points);
      for (index_t j = 0; j < num_points; j++) {
        for (index_t i = 0; i < num_dimensions; i++) {
          random_dataset->set(i, j, fl::math::Random<Precision>(0, 1.0));
        }
      }
    }

    static int QsortComparator_(const void *elem1, const void *elem2) {
      int q_id1 = (int)(*((double *) elem1));
      int r_id1 = (int)(*((double *) elem1 + 1));
      int q_id2 = (int)(*((double *) elem2));
      int r_id2 = (int)(*((double *) elem2 + 1));

      if ((q_id1 < q_id2 || (q_id1 == q_id2 && r_id1 < r_id2))) {
        return -1;
      }
      else
        if (q_id1 == q_id2 && r_id1 == r_id2) {
          return 0;
        }
        else {
          return 1;
        }
    }

    template<typename Table_t, typename ContainerDistIndPairType>
    static void DriverTest_(Table_t &reference_table,
                            const index_t &knns,
                            const double &random_range,
                            ContainerDistIndPairType &result) {
      // First, dump the reference file to the file.
      fl::table::FileDataAccess data;
      std::string reference_file("test_reference.csv");
      data.Detach(reference_table);
      data.Purge(reference_table);

      // Argument list.
      std::vector< std::string > args;

      // Push "indices_out".
      std::string indices_out_file("result_indices_out_file.txt");
      std::string distances_out_file("result_distances_out_file.txt");
      args.push_back(std::string("--indices_out=" + indices_out_file));

      // Push "distances_out".
      args.push_back(std::string("--distances_out=" + distances_out_file));

      // Monochromatic assumption, so no "--queries_in"
      args.push_back(std::string("--references_in=") + reference_file);

      // Push the tree-type in "--tree"
      if (tree_type == 0 || tree_type == 1) {
        args.push_back(std::string("--tree=kdtree"));
      }
      else {
        args.push_back(std::string("--tree=balltree"));
      }

      // Push the point-type in "--point"
      switch (dataset_type) {
        case 0:
          args.push_back(std::string("--point=dense"));
          break;
        case 1:
          args.push_back(std::string("--point=sparse"));
          break;
        case 2:
          args.push_back(std::string("--point=dense_sparse"));
          break;
        case 3:
          args.push_back(std::string("--point=categorical"));
          break;
        case 4:
          args.push_back(std::string("--point=dense_categorical"));
          break;
      }

      // Decide whether to do range or non-range neighbors.
      if (use_range_cut_off) {
        std::stringstream sstr;
        sstr << "--r_neighbors=" << random_range;
        args.push_back(sstr.str());
      }
      else {
        std::stringstream sstr;
        sstr << "--k_neighbors=" << knns;
        args.push_back(sstr.str());
      }

      // Nearest or furthest.
      if (query_type == 0) {
        args.push_back(std::string("--method=nearest"));
      }
      else {
        args.push_back(std::string("--method=furthest"));
      }

      // dual-tree or single-tree?
      if (use_dualtree) {
        args.push_back(std::string("--algorithm=dual"));
      }
      else {
        args.push_back(std::string("--algorithm=single"));
      }

      // progressive or non-progressive?
      if (is_progressive) {
        args.push_back(std::string("--iterations=1000000"));
      }

      args.push_back(std::string("--cores=1"));
      // Call the main driver.
      fl::ml::AllKN<boost::mpl::void_>::Main <
      fl::table::FileDataAccess,
      fl::table::Branch
      > (&data, args);

      // Load the result: assumes monochromaticity.
      std::vector < std::vector< std::pair< double, index_t> > > driver_result;
      driver_result.resize(reference_table.n_entries());
      if (use_range_cut_off) {
        typename fl::table::DefaultTable tmp_table;
        tmp_table.Init(distances_out_file, "r");
        fl::dense::Matrix<double, false> input_data;
        input_data.Alias(tmp_table.get_point_collection().dense->
                         template get<double>());
        double *pointer = input_data.ptr();
        qsort(pointer, input_data.n_cols(), 3 * sizeof(double),
              QsortComparator_);
        for (int i = 0; i < tmp_table.n_entries(); i++) {
          typename fl::table::DefaultTable::Dataset_t::Point_t point;
          tmp_table.get(i, &point);
          driver_result[ point[0] ].push_back(
            std::pair<double, int>(point[2], point[1]));
        }

      }
      else {
        typename fl::table::DefaultTable distance_table;
        typename fl::table::FileDataAccess::UIntegerTable_t index_table;
        index_table.Init(indices_out_file, "r");
        distance_table.Init(distances_out_file, "r");

        for (int i = 0; i < index_table.n_entries(); i++) {
          typename fl::table::DefaultTable::Dataset_t::Point_t distance_point;
          typename fl::table::FileDataAccess::UIntegerTable_t::Point_t
          index_point;
          index_table.get(i, &index_point);
          distance_table.get(i, &distance_point);

          for (int j = 0; j < index_point.length(); j++) {
            driver_result[i].push_back(
              std::pair<double, int>(distance_point[j], index_point[j]));
          }
        }
      }

      // Compare against the result obtained outside the driver.
      for (unsigned int i = 0; i < result.size(); i++) {
        if (result[i].size() != driver_result[i].size()) {
          printf("The driver got %d results for the %d-th query, but it "
                 "should be %d!\n", driver_result[i].size(), i,
                 result[i].size());
          //throw std::runtime_error("Aborting\n");
        }
        for (unsigned j = 0; j < result[i].size(); j++) {
          if (result[i][j].second != driver_result[i][j].second) {
            printf("For %d-th query, the %d-th result for the driver is %d but"
                   " outside the driver is %d!\n", i, j,
                   driver_result[i][j].second, result[i][j].second);
            throw std::runtime_error("Aborting\n");
          }
        }
      }
    }

    template < typename Table_t, typename MetricType,
    typename ContainerDistIndPairType >
    static void UltraNaive_(Table_t &reference_table, Table_t &query_table,
                            const MetricType &metric,
                            const index_t &knns,
                            const double &random_range,
                            ContainerDistIndPairType &check_result) {

      // Get the iterator to the query table and the reference table.
      typename Table_t::TreeIterator q_iterator =
        reference_table.get_node_iterator(reference_table.get_tree());
      typename Table_t::TreeIterator r_iterator =
        reference_table.get_node_iterator(reference_table.get_tree());

      std::vector<std::pair<double, index_t> > neighbors;
      neighbors.resize(reference_table.n_entries());
      while (q_iterator.HasNext()) {
        typename Table_t::Dataset_t::Point_t q_point;
        index_t q_index;
        q_iterator.Next(&q_point, &q_index);

        r_iterator.Reset();
        while (r_iterator.HasNext()) {

          typename Table_t::Dataset_t::Point_t r_point;
          index_t r_index;
          r_iterator.Next(&r_point, &r_index);
          double squared_distance =
            metric.DistanceSq(q_point, r_point);

          // This is a hack for the monochromatic case. This needs to
          // be corrected for the bi-chromatic case. For monochromatic
          // nearest-neighbor, we make the self-distance to be
          // infinity.
          if (q_index == r_index) {
            if (query_type == 0) {
              squared_distance = std::numeric_limits<double>::max();
            }
          }
          neighbors[r_index] = std::make_pair(squared_distance, r_index);
        }

        // Sort the distances and the indices.
        unsigned int limit = knns;
        if (query_type == 0) {
          if (use_range_cut_off) {
            NearestNeighborComparator2 nc(random_range);
            std::sort(neighbors.begin(), neighbors.end(), nc);
          }
          else {
            std::sort(neighbors.begin(), neighbors.end(),
                      std::less< std::pair<double, index_t> >());
          }
          if (use_range_cut_off) {
            limit = 0;
            for (unsigned int i = 0; i < neighbors.size(); i++) {
              if (neighbors[i].first < random_range) {
                limit++;
              }
            }
          }
        }
        else {
          if (use_range_cut_off) {
            FurthestNeighborComparator2 fc(random_range);
            std::sort(neighbors.begin(), neighbors.end(), fc);
          }
          else {
            std::sort(neighbors.begin(), neighbors.end(),
                      std::greater< std::pair<double, index_t> >());
          }
          if (use_range_cut_off) {
            limit = 0;
            for (unsigned int i = 0; i < neighbors.size(); i++) {
              if (neighbors[i].first > random_range) {
                limit++;
              }
            }
          }
        }

        if (limit != check_result[q_index].size()) {
          fl::logger->Message() <<
          "While checking " << q_index << "-th query, "
          "the number of neighbors returned should be " << limit <<
          " but got " << check_result[q_index].size() << "\n";
          throw std::runtime_error("Aborting\n");
        }

        // Check the indices and distances. Note that this is
        // specialized for monochromatic case now, but will be fixed.
        for (unsigned int j = 0; j < limit; j++) {
          if (check_result[q_index][j].second != neighbors[j].second &&
              check_result[q_index][j].first != neighbors[j].first) {

            printf("While checking %d-th query, we found:\n", q_index);
            if (query_type == 0) {
              printf("%d-th nearest neighbor should have the index of %d, "
                     "the squared distance of %g but I got %d and %g.\n",
                     j + 1, neighbors[j].second, neighbors[j].first,
                     check_result[q_index][j].second,
                     check_result[q_index][j].first);
            }
            else {
              printf("%d-th furthest neighbor should have the index of %d, "
                     "the squared distance of %g but I got %d and %g.\n",
                     j + 1, neighbors[j].second, neighbors[j].first,
                     check_result[q_index][j].second,
                     check_result[q_index][j].first);
            }

            throw std::runtime_error("Aborting\n");
          }
        }
      }
    }

    struct Choise1 {
      struct type {
        template<typename MetricType, typename PointType>
        static void set_weights(MetricType *metric, PointType &p) {
          metric->set_weights(p);
        }
      };
    };

    struct Choise2 {
      struct type {
        template<typename MetricType, typename PointType>
        static void set_weights(MetricType *metric, PointType &p) {
        }
      };
    };

    class NearestNeighborComparator {
      public:
        bool operator()(std::pair< std::pair< int, double >, int > a,
                        std::pair< std::pair< int, double >, int > b) {
          return a.first.first < b.first.first ||
                 (a.first.first == b.first.first &&
                  a.second < b.second) ||
                 (a.first.first == b.first.first &&
                  a.second == b.second &&
                  a.first.second < b.first.second);
        }
    };

    class NearestNeighborComparator2 {
      private:
        const double &cut_off_;

      public:
        NearestNeighborComparator2(const double &cut_off_in):
            cut_off_(cut_off_in) {
        }

        bool operator()(std::pair< double, int > a,
                        std::pair< double, int > b) {
          return (a.first < cut_off_ && b.first < cut_off_ &&
                  (a.second < b.second ||
                   (a.second == b.second && a.first < b.first))) ||
                 (a.first < cut_off_ && b.first >= cut_off_);
        }
    };

    class FurthestNeighborComparator {
      public:
        bool operator()(std::pair< std::pair< int, double >, int > a,
                        std::pair< std::pair< int, double >, int > b) {
          return a.first.first > b.first.first ||
                 (a.first.first == b.first.first &&
                  a.second < b.second) ||
                 (a.first.first == b.first.first &&
                  a.second == b.second &&
                  a.first.second < b.first.second);
        }
    };

    class FurthestNeighborComparator2 {
      private:
        const double &cut_off_;

      public:
        FurthestNeighborComparator2(const double &cut_off_in):
            cut_off_(cut_off_in) {
        }

        bool operator()(std::pair< double, int > a,
                        std::pair< double, int > b) {
          return
            (a.first > cut_off_ && b.first > cut_off_ &&
             (a.second < b.second ||
              (a.second == b.second && a.first > b.first))) ||
            (a.first > cut_off_ && b.first <= cut_off_);
        }
    };

    template<typename AllKNN, typename MetricType>
    static void StressTestHelper_(
      fl::dense::Matrix<double, false> &random_dataset,
      int knns, double random_range) {

      // The leaf size.
      int leaf_size = fl::math::Random(5, 120);
      fl::logger->Message() << "Dataset: " << random_dataset.n_cols() <<
      " points, "
      << random_dataset.n_rows() << " dimensions";

      fl::logger->Message() << "Using leaf size of " << leaf_size;

      // Create the table.
      AllKNN *allknn = new AllKNN();
      typename AllKNN::ReferenceTable_t *reference_table = new
      typename AllKNN::ReferenceTable_t();

      fl::table::FileDataAccess data;
      std::string reference_file("test_reference.csv");
      data.Attach(reference_file,
                  std::vector<int>(1, random_dataset.n_rows()),
                  std::vector<int>(),
                  random_dataset.n_cols(),
                  reference_table);
      for (int i = 0; i < random_dataset.n_cols(); i++) {
        typename AllKNN::ReferenceTable_t::Point_t point;
        reference_table->get(i, &point);
        for (int j = 0; j < random_dataset.n_rows(); j++) {
          point[j] = random_dataset.get(j, i);
        }
      }

      typename AllKNN::QueryTable_t::template IndexArgs<MetricType>
      r_index_args;
      fl::data::MonolithicPoint<double> dummy;
      dummy.Init(reference_table->n_attributes());
      dummy.SetAll(1.0);
      boost::mpl::eval_if <
      boost::mpl::or_ <
      boost::is_same <
      MetricType,
      fl::math::LMetric<2>
      > ,
      boost::is_same <
      MetricType,
      fl::math::CosineMetric
      >
      > ,
      Choise2,
      Choise1
      >::type::set_weights(&r_index_args.metric, dummy);
      r_index_args.leaf_size = leaf_size;
      fl::logger->Message() << "Data contains " <<
      reference_table->n_entries() << " points.";
      reference_table->IndexData(r_index_args);
      fl::logger->Message() << "Built index on query data.";

      allknn->Init(reference_table, NULL);

      std::vector<index_t> ind_neighbors_method(
        reference_table->n_entries() * knns);
      std::vector< std::pair<index_t, index_t> > range_ind_neighbors_method;
      std::vector<typename AllKNN::CalcPrecision_t> dist_neighbors_method(
        reference_table->n_entries() * knns);

      if (use_range_cut_off == false) {
        fl::logger->Message() << "Computing nearest neighbors for KNN=" << knns;
      }
      else {
        fl::logger->Message() << "Computing nearest neighbors with range=" <<
        random_range;
        ind_neighbors_method.resize(0);
        dist_neighbors_method.resize(0);
      }

      // Decide whether to use the dual-tree or single-tree.
      std::string computation_method((use_dualtree) ? "dual" : "single");

      // Compute neighbors using AllKN.
      if (is_progressive) {
        typename fl::ml::DefaultAllKNN::ReferenceTable_t
        ::template IndexArgs<fl::math::LMetric<2> > r_index_args;
        r_index_args.leaf_size = leaf_size;

        if (use_range_cut_off) {
          typename AllKNN::template iterator < fl::math::LMetric<2>, double,
          std::vector<double>, std::vector< std::pair<index_t, index_t> > > it;
          it.Init(allknn, computation_method, r_index_args.metric,
                  knns, &dist_neighbors_method, &range_ind_neighbors_method);

          while ((*it) != 0) {
            ++it;
          }
        }
        else {
          typename AllKNN::template iterator < fl::math::LMetric<2>, index_t,
          std::vector<double>, std::vector<index_t> > it;
          it.Init(allknn, computation_method, r_index_args.metric,
                  knns, &dist_neighbors_method, &ind_neighbors_method);

          while ((*it) != 0) {
            ++it;
          }
        }

      }
      else {
        if (use_range_cut_off == false) {
          allknn->ComputeNeighbors(computation_method,
                                   r_index_args.metric,
                                   knns,
                                   &dist_neighbors_method,
                                   &ind_neighbors_method);
        }
        else {
          allknn->ComputeNeighbors(computation_method,
                                   r_index_args.metric,
                                   random_range,
                                   &dist_neighbors_method,
                                   &range_ind_neighbors_method);
        }
      }

      // Convert the result so that it can be compared with the ultra-naive.
      std::vector < std::vector< std::pair< double, index_t> > > result;

      // This line assumes that it is monochromatic.
      result.resize(reference_table->n_entries());

      bool test_driver = true;
      if (use_range_cut_off == false) {
        // Copy to the final result.
        int query_id = -1;
        for (unsigned int i = 0; i < dist_neighbors_method.size(); i++) {
          if (i % knns == 0) {
            query_id++;
          }
          result[query_id].push_back(std::pair<double, index_t>(
                                       dist_neighbors_method[i],
                                       ind_neighbors_method[i]));
        }
      }
      else {

        // Query index, reference dist, reference index.
        std::vector< std::pair< std::pair< index_t, double>, index_t > >
        tmp_result;
        tmp_result.resize(range_ind_neighbors_method.size());

        for (unsigned int i = 0; i < range_ind_neighbors_method.size(); i++) {

          tmp_result[i].first.first = range_ind_neighbors_method[i].first;

          // If furthest neighbor, flip the query index to be negative.
          if (query_type == 1) {
            tmp_result[i].first.first = -tmp_result[i].first.first;
          }
          tmp_result[i].first.second = dist_neighbors_method[i];
          tmp_result[i].second = range_ind_neighbors_method[i].second;
        }

        if (query_type == 0) {
          NearestNeighborComparator nc;
          std::sort(tmp_result.begin(), tmp_result.end(), nc);
        }
        else {
          FurthestNeighborComparator fc;
          std::sort(tmp_result.begin(), tmp_result.end(), fc);
        }
        for (unsigned int i = 0; i < tmp_result.size(); i++) {
          tmp_result[i].first.first = abs(tmp_result[i].first.first);
        }

        // Copy to the final result.
        for (unsigned int i = 0; i < tmp_result.size(); i++) {

          // Get the query ID.
          int query_id = tmp_result[i].first.first;
          double distance = tmp_result[i].first.second;
          int ref_id = tmp_result[i].second;
          result[query_id].push_back(std::pair<double, index_t>(distance,
                                     ref_id));
        }

        // Don't test against the driver if the result is empty.
        test_driver = (tmp_result.size() > 0);
      }

      fl::logger->Message() << "Comparing with the ultra-naive.";
      UltraNaive_(*reference_table, *reference_table, r_index_args.metric,
                  knns, random_range, result);

      if (test_driver) {
        fl::logger->Message() << "Comparing with the driver result.";
        DriverTest_(*reference_table, knns, random_range, result);
      }

      fl::logger->Message() << "Comparison complete!\n";

      delete reference_table;
      delete allknn;
    }

  public:

    template<int P>
    struct AllKNArgs : public fl::ml::AllKNArgs {
      typedef fl::table::Table< TableMap<P, sort_points> > QueryTableType;
      typedef fl::table::Table< TableMap<P, sort_points> > ReferenceTableType;
      typedef boost::mpl::int_<query_type> KNmode;
    };

    typedef fl::ml::AllKN< AllKNArgs<tree_type> > MyAllKNN;
    typedef typename TableMap<tree_type, sort_points>::TreeArgs::MetricType MyMetric;

    static void StressTestMain() {

      fl::logger->Message() << "StressTestMain()";

      switch (dataset_type) {
        case 0:
          fl::logger->Message() << "Dense, ";
          break;
        case 1:
          fl::logger->Message() << "Sparse, ";
          break;
      }
      if (use_range_cut_off) {
        fl::logger->Message() << "Range cut-off, ";
      }
      else {
        fl::logger->Message() <<  "No range cut-off, ";
      }
      if (use_dualtree) {
        fl::logger->Message() << "Using dual-tree, ";
      }
      else {
        fl::logger->Message() << "Using single-tree, ";
      }
      if (is_progressive) {
        fl::logger->Message() << "Progressive, ";
      }
      else {
        fl::logger->Message() << "Non-progressive, ";
      }
      if (sort_points) {
        fl::logger->Message() << "Sorting points, ";
      }
      else {
        fl::logger->Message() << "Not sorting points, ";
      }
      switch (tree_type) {
        case 0:
          fl::logger->Message() << "Midpoint kd-tree, ";
          break;
        case 1:
          fl::logger->Message() << "Median kd-tree, ";
          break;
        case 2:
          fl::logger->Message() << "Metric tree, with centroid ";
          break;
        case 3:
          fl::logger->Message() << "Metric tree, with mid-point ";
          break;
        case 4:
          fl::logger->Message() << "Metric tree, with cosine metric";
          break;
        default:
          fl::logger->Message() << "Invalid tree type, ";
          break;
      }
      switch (query_type) {
        case 0:
          fl::logger->Message() << "Nearest neighbor.\n";
          break;
        case 1:
          fl::logger->Message() << "Furthest neighbor.\n";
          break;
        default:
          fl::logger->Message() << "Invalid query type.\n";
          break;
      }

      for (index_t i = 0; i < 5; i++) {

        // Random dataset and create a table from it so that you can
        // run the algorithm.
        fl::dense::Matrix<double, false> random_dataset;
        index_t num_dimensions = fl::math::Random(2, 15);
        index_t num_points = fl::math::Random(100, 201);
        index_t knns = fl::math::Random(1, 10);
        double random_range = fl::math::Random<double>(0.1 *
                              sqrt(num_dimensions),
                              sqrt(num_dimensions));

        GenerateRandomDataset_(
          num_dimensions, num_points, &random_dataset);

        StressTestHelper_<MyAllKNN,MyMetric>(
          random_dataset, knns, random_range);
      }
    }
};



template <
  int      dataset_type,
  bool     use_range_cut_off,
  bool     use_dualtree,
  bool     is_progressive,
  bool     sort_points,
  int      tree_type,
  typename query_type_options
>
struct TemplateRecursion1 {
  struct RunOperator {
    template<typename T>
    void operator()(T) {
      TestAllKN<
        dataset_type,
        use_range_cut_off,
        use_dualtree,
        is_progressive,
        sort_points,
        tree_type,
        T::value
      >::StressTestMain();
    }
  };

  static void BuildMap() {
    boost::mpl::for_each < query_type_options >(RunOperator());
  }
};

template <
  int      dataset_type,
  bool     use_range_cut_off,
  bool     use_dualtree,
  bool     is_progressive,
  bool     sort_points,
  typename tree_type_options,
  typename query_type_options
>
struct TemplateRecursion2 {
  struct RunOperator {
    template<typename T>
    void operator()(T) {
      TemplateRecursion1<
        dataset_type,
        use_range_cut_off,
        use_dualtree,
        is_progressive,
        sort_points,
        T::value,
        query_type_options
      >::BuildMap();
    }
  };

  static void BuildMap() {
    boost::mpl::for_each < tree_type_options >(RunOperator());
  }
};

template <
  int      dataset_type,
  bool     use_range_cut_off,
  bool     use_dualtree,
  bool     is_progressive,
  typename sort_points_options,
  typename tree_type_options,
  typename query_type_options
>
struct TemplateRecursion3 {
  struct RunOperator {
    template<typename T>
    void operator()(T) {
      TemplateRecursion2<
        dataset_type,
        use_range_cut_off,
        use_dualtree,
        is_progressive,
        T::value,
        tree_type_options,
        query_type_options
      >::BuildMap();
    }
  };

  static void BuildMap() {
    boost::mpl::for_each < sort_points_options >(RunOperator());
  }
};

template <
  int      dataset_type,
  bool     use_range_cut_off,
  bool     use_dualtree,
  typename is_progressive_options,
  typename sort_points_options,
  typename tree_type_options,
  typename query_type_options
>
struct TemplateRecursion4 {
  struct RunOperator {
    template<typename T>
    void operator()(T) {
      TemplateRecursion3<
        dataset_type,
        use_range_cut_off,
        use_dualtree,
        T::value,
        sort_points_options,
        tree_type_options,
        query_type_options
      >::BuildMap();
    }
  };

  static void BuildMap() {
    boost::mpl::for_each < is_progressive_options >(RunOperator());
  }
};

template <
  int      dataset_type,
  bool     use_range_cut_off,
  typename use_dualtree_options,
  typename is_progressive_options,
  typename sort_points_options,
  typename tree_type_options,
  typename query_type_options
>
struct TemplateRecursion5 {
  struct RunOperator {
    template<typename T>
    void operator()(T) {
      TemplateRecursion4<
        dataset_type,
        use_range_cut_off,
        T::value,
        is_progressive_options,
        sort_points_options,
        tree_type_options,
        query_type_options
      >::BuildMap();
    }
  };

  static void BuildMap() {
    boost::mpl::for_each < use_dualtree_options >(RunOperator());
  }
};

template <
  int      dataset_type,
  typename use_range_cut_off_options,
  typename use_dualtree_options,
  typename is_progressive_options,
  typename sort_points_options,
  typename tree_type_options,
  typename query_type_options
>
struct TemplateRecursion6 {
  struct RunOperator {
    template<typename T>
    void operator()(T) {
      TemplateRecursion5<
        dataset_type,
        T::value,
        use_dualtree_options,
        is_progressive_options,
        sort_points_options,
        tree_type_options,
        query_type_options
      >::BuildMap();
    }
  };

  static void BuildMap() {
    boost::mpl::for_each < use_range_cut_off_options >(RunOperator());
  }
};

template <
  typename dataset_type_options,
  typename use_range_cut_off_options,
  typename use_dualtree_options,
  typename is_progressive_options,
  typename sort_points_options,
  typename tree_type_options,
  typename query_type_options
>
struct TemplateRecursion7 {
  struct RunOperator {
    template<typename T>
    void operator()(T) {
      TemplateRecursion6<
        T::value,
        use_range_cut_off_options,
        use_dualtree_options,
        is_progressive_options,
        sort_points_options,
        tree_type_options,
        query_type_options
      >::BuildMap();
    }
  };

  static void BuildMap() {
    boost::mpl::for_each < dataset_type_options >(RunOperator());
  }
};

#endif
