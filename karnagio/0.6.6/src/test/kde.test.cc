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
 * @file kde_test.cc
 *
 * A "stress" test driver for KDE.
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include "boost/mpl/bool.hpp"
#include "boost/mpl/vector_c.hpp"
#include "boost/test/unit_test.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/int.hpp"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/tree/tree.h"
#include "fastlib/table/branch_on_table_dev.h"
#include "fastlib/table/file_data_access.h"
#include "fastlib/table/table.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/tree/kdtree.h"
#include "mlpack/kde/kde_dev.h"
#include <time.h>
#include "fastlib/table/table_dev.h"
namespace fl {
namespace ml {
  template<typename MetricType>
  class SetMetricWeights {
    public:
      template<typename PointType>
      SetMetricWeights(PointType &metric_weights_in,
                       MetricType &metric);
  };
  
  template<>
  class SetMetricWeights< fl::math::LMetric<2> > {
  
    public:
      template<typename PointType>
      SetMetricWeights(PointType &metric_weights_in,
                       fl::math::LMetric<2> &metric) {
      }
  };
  
  template<>
  class SetMetricWeights < fl::math::WeightedLMetric < 2,
        fl::data::MonolithicPoint<double> > > {
  
    public:
      template<typename PointType>
      SetMetricWeights(PointType &metric_weights_in,
                       fl::math::WeightedLMetric < 2,
                       fl::data::MonolithicPoint<double> > &metric) {
        metric.set_weights(metric_weights_in);
      }
  };
  

namespace test_kde {
int num_dimensions_;
int num_points_;
};
};
};

template<int KernelType, int PointType, int MetricType>
class TestKde {

  public:
    template<typename TableType1>
    class Core {

      private:

        template<typename QueryResultType, typename NaiveQueryResultType>
        static bool CheckAccuracy_(
          const QueryResultType &query_results,
          const NaiveQueryResultType &naive_query_results,
          const double &relative_error) {

          // Compute the collective L1 norm of the products.
          double achieved_error = 0;
          for (unsigned int j = 0; j < query_results.size(); j++) {
            double per_relative_error =
              fabs(naive_query_results[j] - query_results[j]) /
              fabs(naive_query_results[j]);
            achieved_error = std::max(achieved_error, per_relative_error);
            if (relative_error < per_relative_error) {
              fl::logger->Message() << query_results[j] << " against " <<
              naive_query_results[j] << ": " << per_relative_error;
            }
          }
          fl::logger->Message() <<
          "Achieved a relative error of " << achieved_error;
          return achieved_error <= relative_error;
        }

        template<typename DataAccessType>
        static void AllocateTable_(const std::string &filename,
                                   DataAccessType *data,
                                   boost::program_options::variables_map &vm,
                                   int num_dimensions,
                                   int num_points, TableType1 *table) {

          if (vm["point"].as<std::string>() == "dense") {
            data->Attach(filename,
                         std::vector<int>(1, num_dimensions),
                         std::vector<int>(),
                         num_points,
                         table);
          }
          else if (vm["point"].as<std::string>() == "sparse") {
            data->Attach(filename,
                         std::vector<int>(),
                         std::vector<int>(1, num_dimensions),
                         num_points,
                         table);
          }
          else {
            fl::logger->Die() << "Unsupported point type in the test!";
          }
        }

        template<typename DataAccessType, typename TableType>
        static void GenerateRandomDataset_(
          const std::string &filename,
          DataAccessType *data,
          boost::program_options::variables_map &vm,
          int num_dimensions,
          int num_points,
          TableType *random_dataset) {

          AllocateTable_(filename, data, vm, num_dimensions, num_points,
                         random_dataset);

          for (index_t j = 0; j < num_points; j++) {
            typename TableType::Dataset_t::Point_t point;
            random_dataset->get(j, &point);
            for (index_t i = 0; i < num_dimensions; i++) {

              // If the dataset is dense or the sparse but the current
              // dimension is to be set to a value, then set it.
              int random_flip = fl::math::Random(2);
              if (random_flip || vm["point"].as<std::string>() == "dense") {
                point.set(i, fl::math::Random<double>(0.1, 1.0));
              }
            }
          }
        }

        template < typename Table_t, typename Kernel_t, typename Result_t,
        typename Metric_t >
        static void UltraNaive_(Table_t &query_table, Table_t &reference_table,
                                const Kernel_t &kernel,
                                const Metric_t &metric,
                                Result_t &ultra_naive_query_results) {

          typedef typename Table_t::Point_t Point_t;
          typedef typename Point_t::CalcPrecision_t CalcPrecision_t;

          for (index_t i = 0; i < query_table.n_entries(); i++) {
            Point_t query_point;
            query_table.get(i, &query_point);

            for (index_t j = 0; j < reference_table.n_entries(); j++) {
              Point_t reference_point;
              reference_table.get(j, &reference_point);

              // By default, monochromaticity is assumed in the test -
              // this will be addressed later for general bichromatic
              // test.
              if (i == j) {
                continue;
              }

              CalcPrecision_t squared_distance =
                metric.DistanceSq(query_point, reference_point);
              CalcPrecision_t kernel_value =
                kernel.EvalUnnormOnSq(squared_distance);

              ultra_naive_query_results[i] += kernel_value;
            }

            // Divide by N - 1 for LOO. May have to be adjusted later.
            ultra_naive_query_results[i] *=
              (1.0 / (kernel.CalcNormConstant(query_table.n_attributes()) *
                      ((double)
                       reference_table.n_entries() - 1)));
          }
        }

      public:

        template < typename KernelType1, typename MetricType1,
        typename DataAccessType >
        static void Branch_(DataAccessType *data,
                            boost::program_options::variables_map &vm,
                            int level) {

          switch (level) {
            case 0:

              // Choose the kernel.
              if (vm["kernel"].as<std::string>() == "epan") {
                Branch_< fl::math::EpanKernel<double>, MetricType1>(data, vm,
                    level + 1);
              }
              else if (vm["kernel"].as<std::string>() == "gaussian") {
                Branch_<fl::math::GaussianKernel<double>, MetricType1>(data, vm,
                    level + 1);
              }
              break;
            case 1:
              // Choose the metric.
              if (vm["metric"].as<std::string>() == "l2") {
                Branch_<KernelType1, fl::math::LMetric<2> >(data, vm,
                    level + 1);
              }
              else if (vm["metric"].as<std::string>() == "weighted_l2") {
                Branch_ < KernelType1, fl::math::WeightedLMetric < 2,
                fl::data::MonolithicPoint<double> > > (data, vm, level + 1);
              }
              break;
            case 2:

              typename TableType1::template IndexArgs< MetricType1 > args;

              // Read the densities from the file.
              typename fl::table::DefaultTable densities;
              data->Attach(vm["densities_out"].as<std::string>(), &densities);
              if (fl::ml::test_kde::num_points_ != densities.n_entries()) {
                throw std::runtime_error("The number of density values "
                                         "incorrect");
              }

              // Preparing for the ultra-naive computation.
              TableType1 *reference_table = new TableType1();
              std::vector< std:: string > references_in_set =
                vm["references_in"].as< std::vector< std::string > >();
              data->Attach(references_in_set[0], reference_table);
              TableType1 *query_table = reference_table;
              if (fl::ml::test_kde::num_points_ != query_table->n_entries()) {
                throw std::runtime_error("Number of query points incorrect");
              }

              typename DataAccessType::DefaultTable_t::Point_t metric_weights_in;
              typename DataAccessType::DefaultTable_t metric_weights_table;
              data->Attach(vm["metric_weights_in"].as<std::string>(),
                           &metric_weights_table);
              metric_weights_table.get(0, &metric_weights_in);
              fl::data::MonolithicPoint<double> metric_weights_in_copy;
              metric_weights_in_copy.Copy(metric_weights_in);
              fl::ml::SetMetricWeights< MetricType1> (
                metric_weights_in_copy, args.metric);

              // Ultra naive results.
              std::vector<double> ultra_naive_query_results(
                query_table->n_entries(), 0);

              KernelType1 kernel;
              kernel.Init(vm["bandwidth"].as<double>());
              UltraNaive_(*query_table, *reference_table, kernel, args.metric,
                          ultra_naive_query_results);

              // Check the ultra naive result.
              std::vector<double> densities_results;
              densities_results.resize(densities.n_entries());
              for (unsigned int i = 0; i < densities.n_entries(); i++) {
                typename fl::table::DefaultTable::Point_t density_point;
                densities.get(i, &density_point);
                densities_results[i] = density_point[0];
              }

              bool achieved_accuracy = CheckAccuracy_(
                                         densities_results,
                                         ultra_naive_query_results,
                                         vm["relative_error"].as<double>());
              if (! achieved_accuracy) {
                throw std::runtime_error("Aborting\n");
              }

              // Delete the reference table.
              delete reference_table;
              break;
          }
        }

        template<typename DataAccessType>
        static int Main(DataAccessType *data,
                        boost::program_options::variables_map &vm) {

          // Randomly generate the number of dimensions and the points.
          fl::logger->Message() << "Number of dimensions: " <<
          fl::ml::test_kde::num_dimensions_;
          fl::logger->Message() << "Number of points: " <<
          fl::ml::test_kde::num_points_;

          // Generate the dataset and save it.
          TableType1 random_table;

          // Parse the references_in
          std::vector< std::string > references_in_set =
            vm["references_in"].as< std::vector< std::string > >();

          GenerateRandomDataset_(references_in_set[0],
                                 data, vm,
                                 fl::ml::test_kde::num_dimensions_,
                                 fl::ml::test_kde::num_points_,
                                 &random_table);
          data->Purge(random_table);
          data->Detach(random_table);

          // Generate the random metric weight and save it.
          typename DataAccessType::DefaultTable_t random_metric_weights;
          data->Attach(vm["metric_weights_in"].as<std::string>(),
              std::vector<int>(1, fl::ml::test_kde::num_dimensions_),
              std::vector<int>(),
              1,
              &random_metric_weights);
         // AllocateTable_(vm["metric_weights_in"].as<std::string>(), data, vm,
         //                fl::ml::test_kde::num_dimensions_, 1,
         //                &random_metric_weights);

          for (int j = 0; j < 1; j++) {
            typename DataAccessType::DefaultTable_t::Point_t metric_weight;
            random_metric_weights.get(j, &metric_weight);
            for (int i = 0; i < fl::ml::test_kde::num_dimensions_; i++) {
              metric_weight.set(i, fl::math::Random<double>(0.25, 3.5));
            }
          }
          data->Purge(random_metric_weights);
          data->Detach(random_metric_weights);

          // Call the main KDE core to do the actual driver call.
          fl::ml::Kde<boost::mpl::void_>::Core<TableType1>::Main(
            data, vm);

          // Call the naive code to verify.
          Branch_< fl::math::EpanKernel<double>, fl::math::LMetric<2> >(
            data, vm, 0);
          return 1;
        }
    };

    static int StressTestMain() {
      for (int i = 0; i < 5; i++) {

        // Randomly choose the number of dimensions and the points.
        fl::ml::test_kde::num_dimensions_ = fl::math::Random(3, 20);
        fl::ml::test_kde::num_points_ = fl::math::Random(100, 1001);
        StressTest();
      }
      return 0;
    }

    static int StressTest() {

      fl::table::FileDataAccess data;
      std::vector< std::string > args;

      // Push in the reference dataset name.
      args.push_back(std::string("--references_in=random.csv"));

      // Push in the densities output file name.
      args.push_back(std::string("--densities_out=densities.txt"));

      // Push in the kernel type.
      fl::logger->Message() << "\n==================";
      fl::logger->Message() << "Test trial begin";
      switch (KernelType) {
        case 0:
          fl::logger->Message() << "Epan kernel, ";
          args.push_back(std::string("--kernel=epan"));
          break;
        case 1:
          fl::logger->Message() << "Gaussian kernel, ";
          args.push_back(std::string("--kernel=gaussian"));
          break;
      }

      // Push in the randomly generated bandwidth.
      double bandwidth =
        fl::math::Random<double>(0.1 *
                                 sqrt(2.0 * fl::ml::test_kde::num_dimensions_),
                                 0.2 *
                                 sqrt(2.0 * fl::ml::test_kde::num_dimensions_));
      std::stringstream bandwidth_sstr;
      bandwidth_sstr << "--bandwidth=" << bandwidth;
      args.push_back(bandwidth_sstr.str());

      // Push in the point type.
      switch (PointType) {
        case 0:
          fl::logger->Message() << "Dense, ";
          args.push_back(std::string("--point=dense"));
          break;
        case 1:
          fl::logger->Message() << "Sparse, ";
          args.push_back(std::string("--point=sparse"));
          break;
      }

      // Push in the metric type.
      switch (MetricType) {
        case 0:
          fl::logger->Message() << "L2 metric";
          args.push_back(std::string("--metric=l2"));
          break;
        case 1:
          fl::logger->Message() << "Weighted L2 metric";
          args.push_back(std::string("--metric=weighted_l2"));
          break;
      }

      // Push in the metric weights file name.
      args.push_back(std::string("--metric_weights_in=random_metric.csv"));

      // Push in the tree type.
      args.push_back(std::string("--tree=balltree"));

      // Parse the boost program options.
      boost::program_options::variables_map vm;
      fl::ml::Kde<boost::mpl::void_>::ConstructBoostVariableMap(args, &vm);

      fl::logger->Message() << "Bandwidth value " <<
      vm["bandwidth"].as<double>();

      // Call to determine the dataset type and finish the KDE computation.
      return fl::table::Branch::template BranchOnTable <
             TestKde<KernelType, PointType, MetricType>,
             fl::table::FileDataAccess > (&data, vm);
    }
};

template < int KernelType, int PointType, int MetricType,
typename kernel_type_options, typename point_type_options,
typename metric_type_options, int level >
class TemplateRecursion {

  public:

    struct RunOperator1 {
      template<typename T>
      void operator()(T) {
        TemplateRecursion < KernelType, PointType, T::value,
        kernel_type_options, point_type_options, metric_type_options,
        level - 1 >::BuildMap();
      }
    };

    struct RunOperator2 {
      template<typename T>
      void operator()(T) {
        TemplateRecursion < KernelType, T::value, MetricType,
        kernel_type_options, point_type_options, metric_type_options,
        level - 1 >::BuildMap();
      }
    };

    struct RunOperator3 {
      template<typename T>
      void operator()(T) {
        TemplateRecursion < T::value, PointType, MetricType,
        kernel_type_options, point_type_options, metric_type_options,
        level - 1 >::BuildMap();
      }
    };

    static void BuildMap() {

      switch (level) {
        case 3:
          boost::mpl::for_each < kernel_type_options >(RunOperator3());
          break;
        case 2:
          boost::mpl::for_each < point_type_options >(RunOperator2());
          break;
        case 1:
          boost::mpl::for_each < metric_type_options >(RunOperator1());
          break;
      };
    }
};

template < int KernelType, int PointType, int MetricType,
typename kernel_type_options, typename point_type_options,
typename metric_type_options >
class TemplateRecursion < KernelType, PointType, MetricType,
    kernel_type_options, point_type_options,
      metric_type_options, 0 > {
  public:
    static void BuildMap() {
      TestKde<KernelType, PointType, MetricType>::StressTestMain();
    }
};

// In the order of int KernelType, int PointType, int MetricType
typedef boost::mpl::vector_c<int, 0, 1> kernel_type_options;
typedef boost::mpl::vector_c<int, 0, 1> point_type_options;
typedef boost::mpl::vector_c<int, 0, 1> metric_type_options;

BOOST_AUTO_TEST_SUITE(TestSuiteKde)
BOOST_AUTO_TEST_CASE(TestCaseKde) {
  srand(time(NULL));

  // Call the tests.
  TemplateRecursion < 0, 0, 0, kernel_type_options, point_type_options,
  metric_type_options, 3 >::BuildMap();

  fl::logger->Message() << "All tests passed!";
}
BOOST_AUTO_TEST_SUITE_END()
