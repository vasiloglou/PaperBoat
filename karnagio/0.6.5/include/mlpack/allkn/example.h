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
 * @brief This file contains examples about how to use the nearest neighbor
 *        class in the 1305 code base. For defining tables and loading them 
 *        @sa include/fastlib/table/example.h
 *
 * This is an example of how to create an object that computes all nearest neighbors:
 *
 * First you have to include the necessary headers
 * @code
 *  #include "fastlib/default_table.h"
 *  #include "mlpack/allkn/allkn_dev.h"
 *  #include "fastlib/metric_kernel/lmetric.h"
 *
 * @endcode 
 *
 * Next we create a dense table (that has a kd-tree as an index) and read data from a file
 * @code
 *  fl::table::DefaultTable table;
 *  table.Init("mydata.txt", "r");
 * @endcode
 *
 * The next commands generate a tree on the data based on eucledian metric
 * @code
 *   typedef fl::table::DefaultTable::template IndexArgs<fl::math::LMetric<2> > l_index_args;
 *   // sets the leaf size of the tree
 *   l_index_args.leaf_size=40;
 *   table.IndexData(l_index_args);
 * @endcode
 *
 * Now we need to define the template parameters of the AllKN class
 * @code
 *  struct DefaultAllKNNMap : public fl::ml::AllKNArgs {
 *    typedef fl::table::DefaultTable QueryTableType;
 *    typedef fl::table::DefaultTable ReferenceTableType;
 *    typedef boost::mpl::int_<0>::type  KNmode;
 *  };
 *
 *  fl::ml::AllKN<DefaultAllKNNMap> allknn;
 * @endcode
 * Note that if you wanted to run furthest neighbors you should have used:
 * @code
 *  struct DefaultAllKNNMap : public fl::ml::AllKNArgs {
 *    typedef fl::table::DefaultTable QueryTableType;
 *    typedef fl::table::DefaultTable ReferenceTableType;
 *    typedef boost::mpl::int_<1>::type  KNmode;
 *  };
 * @endcode
 *
 * Now we are ready to create an AllKN engine
 * @code
 *    // Note that in this case the reference table is the same as the 
 *    // query table. In that case we pass a NULL as the second argument.
 *    // This is also called monochromatic case. The algorithm will skip
 *    // the trivial solution of considering the nearest neighbor of a point
 *    // itself. If the query data was different from the reference, you would
 *    // pass it as a second arguments.
 *    allknn.Init(&table, NULL);
 *
 * @endcode
 *
 * The next step is to actually compute the nearest neighbors, with the dual tree
 * algorithm.
 *
 * @code
 *  int k_neighbors=5;
 *  std::vector<index_t> k_ind_neighbors;
 *  std::vector<double>  dist_neighbors;
 *
 *  allknn.ComputeNeighbors(
 *           "dual",
 *           l_index_args.metric,
 *           k_neighbors,
 *           &dist_neighbors,
 *           &k_ind_neighbors);
 * @endcode
 *
 * And that's it we are done!!
 *
 * Now let's see how we can get range neighbors
 * @code
 *  std::vector<std::pair<index_t, index_t> > r_ind_neighbors;
 *  std::vector<double>  dist_neighbors;
 *  double range=0.22;
 *  allknn.ComputeNeighbors(
 *           "dual",
 *           l_index_args.metric,
 *           range,
 *           &dist_neighbors,
 *           &r_ind_neighbors);
 * @endcode
 *
 * If you need to run nearest neighbors in an iterative mode 
 * you can do the following:
 * @sa http://www.ismion.com/documentation/nearest_neighbor.html#more-advanced-examples
 *
 * @code
 *  fl::ml::AllKN<DefaultAllKNNMap>::iterator<fl::math::LMetric<2>, 
 *                                            index_t, 
 *                                            std::vector<double>, 
 *                                            std::vector<index_t> > it;
 *  it.Init(
 *    &allknn,
 *    "dual",
 *     l_index_args.metric,
 *     k_neighbors,
 *     &dist_neighbors,
 *     &k_ind_neighbors);
 *
 *  int iterations=3;
 *
 *  while (iterations > 0 && *it != 0) {
 *    --iterations;
 *    ++it;
 *  };
 * @endcode
 * And that's it. This will run k-nearest neighbors in an iterative way. Every time you 
 * call ++it, it will update the indices and the distances of the nearest neighbors.
 *
 * Now that you know the mechanics if you need more examples @sa include/mlpack/allkn_defs.h
 * 
 */
