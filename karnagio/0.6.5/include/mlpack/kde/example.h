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
 * @file example.h
 * @brief This file contains examples about how to use the kernel density estimation
 *        class in the 1305 code base. For defining tables and loading them 
 *        @sa include/fastlib/table/example.h
 * We will walk through the file /examples/kde.cc 
 * to show how it is possible to use fl-lite outside the build system. You can find a
 * Makefile and a visual studio solution in the examples directory that you can use.
 *
 * First we have to include all the necessary files
 * @code 
 *   #include <vector>
 *   #include "mlpack/kde/kde_dev.h"
 *   #include "mlpack/kde/kde_lscv_function.h"
 *   #include "mlpack/kde/dualtree_dfs_dev.h"
 *   #include "mlpack/kde/kde_stat.h"
 *   #include "fastlib/optimization/lbfgs/lbfgs_dev.h"
 *   #include "fastlib/data/multi_dataset_dev.h"
 *   #include "fastlib/table/default_table.h"
 *   #include "fastlib/metric_kernel/lmetric.h"
 * @endcode
 *
 * The next step is to instantiate the classes we need for Kde
 * @code
 *   // Define an instance of kde
 *   // These types are neccessary for instantiating Kde
 *   struct GlobalArgs {
 * 
 *     // The type of the table is dense.
 *     typedef fl::table::DefaultTable TableType;
 * 
 *     // Using the Gaussian kernel.
 *     typedef fl::math::GaussianKernel<double> KernelType;
 * 
 *     // Calculation precision type.
 *     typedef double CalcPrecisionType;
 *   };
 * 
 *   struct KdeComputationType {
 *     typedef fl::ml::KdeDelta<double> Delta_t;
 *     typedef fl::ml::KdeGlobal<GlobalArgs> Global_t;
 *     typedef fl::ml::KdePostponed<double> Postponed_t;
 *     typedef fl::ml::KdeResult<std::vector<double> > Result_t;
 *     typedef fl::ml::KdeStatistic<double> Statistic_t;
 *     typedef fl::ml::KdeSummary<double> Summary_t;
 *   };
 * 
 * 
 *   struct KdeArgs {
 *     typedef fl::table::DefaultTable TableType;
 *     typedef fl::math::GaussianKernel<double> KernelType;
 *     typedef KdeComputationType ComputationType;
 *     typedef fl::math::LMetric<2> MetricType;
 *   };
 *   typedef fl::ml::Kde<KdeArgs> Kde_t;
 * @endcode
 *
 * Now we are ready for the main function
 * @code
 *   int main(int argc, char* argv[]) {
 * 
 *      // Turn on the logger. to get messages
 *      // Right now the logger outputs everything on the screen
 *      // In general you can redirect it to any struct tha inherits
 *      // from std::stream, for more information take a look at the files
 *       // include/fastlib/base/logger.h and src/fastlib/base/logger.cc
 *      fl::Logger::SetLogger(std::string("verbose"));
 * 
 *      // Next we create a dense table
 *      // (that has a kd-tree as an index) and read data from a file
 * @endcode
 * The next step is to load data to a table. In this example we use simple
 * dense data. If you want to avoid copies there is a way you can pass a pointer
 * @code
 *     fl::table::DefaultTable table;
 *     index_t num_of_points = 1000;
 *     index_t dim = 5;
 *     table.Init("mydata.txt",
 *                std::vector<int>(1, dim),
 *                std::vector<int>(),
 *                num_of_points);
 *     for (index_t i = 0; i < table.n_entries(); ++i) {
 *       fl::table::DefaultTable::Point_t point;
 *       table.get(i, &point);
 *       for (index_t j = 0; j < point.size(); ++j) {
 *         point.set(j, fl::math::Random<double>());
 *       }
 *     }
 * @endcode
 *
 * After we are done with loading we need to build a tree and run KDE
 *
 * @code
 *     // The next commands generate a tree on the data based on eucledian metric
 *     fl::table::DefaultTable::IndexArgs<fl::math::LMetric<2> > l_index_args;
 *     // sets the leaf size of the tree
 *     l_index_args.leaf_size = 40;
 *     table.IndexData(l_index_args); 
 *     // Now we are ready to initialize a KDE instance.
 * 
 *     Kde_t kde_instance;
 *     Kde_t::Result_t result;
 *     double bandwidth = 1.4;
 *     double probability = 1; // we want the approximation to be exact with
 *     // meaning that we are 100% confident KDE will be
 *     // withing the approximation
 *     double relative_error = 0.1;
 *     kde_instance.Init(&table, // reference table
 *                       NULL,  // query table is NULL so  it is a monochromatic case
 *                       bandwidth,
 *                       relative_error,
 *                       probability);
 * 
 *     // Initialize the dual-tree engine for the KDE instance.
 *     fl::ml::DualtreeDfs<Kde_t> dualtree_engine;
 *     dualtree_engine.Init(kde_instance); 
 * @endcode
 *  
 *  There are two modes we can run KDE, one is in progressive mode
 *  where we traverse the tree progressively. After every iteration
 *  the densities are updated
 *
 * Let's see the progressive mode first
 * @code 
 *    // Computing for a fixed bandwidth,
 * 
 *    fl::ml::DualtreeDfs<Kde_t>::iterator<fl::math::LMetric<2> > it
 *      = dualtree_engine.get_iterator(l_index_args.metric, &result);
 *      index_t iterations = 5;
 *      for (int i = 0; i < iterations; i++) {
 *        ++it;
 *      }
 * 
 *     // Tell the iterator that we are done using it so that the
 *     // result can be finalized.
 *     it.Finalize();
 * @endcode
 * This is the non-progressive mode where the algorithm terminates after it has
 * completed its goal. Sometimes this can take too long depending on the dataset and
 * on the approximation error. If your operation is time critical, we suggest you use
 * the progressive mode which has fixed time for every iteration. In reality the execution time
 * of iteration n can be less or equal to the execution of iteration n-1. 
 * @code
 *     // Non-progressive.
 *     fl::util::Timer timer;
 *     timer.Start();
 *     dualtree_engine.Compute(l_index_args.metric, &result);
 *     timer.End();
 *     fl::logger->Message() << "Took " << timer.GetTotalElapsedTime() << "seconds.";
 *     fl::logger->Message() << "Densites"<<std::endl;
 *     for(index_t i=0; i<result.densities_.size(); ++i) {
 *       std::cout<<result.densities_[i]<<",";
 *     }
 *     std::cout<<std::endl;
 *
 * @endcode
 *
 * Most of the times though, the bandwidth is not known, so we provide a way of learning
 * the bandwidth with an optimization process.
 *
 * @code
 *     // if we want to learn the bandwidth then here is the code
 * 
 *     // Choose the starting (plugin) bandwidth in log scale.
 *     typedef fl::ml::KdeLscvFunction<KdeArgs> FunctionType;
 *     FunctionType lscv_function;
 *     lscv_function.Init(kde_instance, l_index_args.metric, result);
 *     bandwidth = log(lscv_function.plugin_bandwidth());
 * 
 *     // Initialize the vector to be optimized.
 *     fl::data::MonolithicPoint<double> lscv_result;
 *     lscv_result.Init(1);
 *     lscv_result[0] = bandwidth;
 * 
 *     // Initialize the LBFGS optimizer and optimize.
 *     fl::ml::Lbfgs< FunctionType > lbfgs_engine;
 *     lbfgs_engine.Init(lscv_function, 1);
 * 
 *     // Let us just do 10 re-starts.
 *     std::pair< fl::data::MonolithicPoint<double>, double >
 *     global_min_point_iterate;
 *     global_min_point_iterate.first.Init(1);
 *     global_min_point_iterate.first.SetZero();
 *     global_min_point_iterate.second = std::numeric_limits<double>::max();
 *     for (int i = 0; i < 10; i++) {
 *       fl::logger->Message() << "LBFGS Restart number: " << i;
 *       lbfgs_engine.Optimize(-1, &lscv_result);
 *       const std::pair<fl::data::MonolithicPoint<double>, double>
 *       &min_point_iterate = lbfgs_engine.min_point_iterate();
 *       if (global_min_point_iterate.second > min_point_iterate.second) {
 *         global_min_point_iterate.first.CopyValues(min_point_iterate.first);
 *         global_min_point_iterate.second = min_point_iterate.second;
 *         lscv_result[0] = min_point_iterate.first[0];
 *         fl::logger->Message() << "The optimal bandwidth currently is " <<
 *         exp(global_min_point_iterate.first[0]) << " with LSCV score of " <<
 *         min_point_iterate.second;
 *       }
 *       else {
 *         break;
 *       }
 *     }
 *   double optimal_bandwidth = exp(global_min_point_iterate.first[0]);
 *   fl::logger->Message() << "The optimal bandwidth is " <<
 *   optimal_bandwidth;
 * }
 * @endcode
 */
