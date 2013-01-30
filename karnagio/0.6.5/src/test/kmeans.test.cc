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
// the test classes
/**
* @kmeans_test.cc
* Test file for K-Means algorithm
* author: Abhimanyu Aditya
* email: abhimanyu@analytics1305.com
*/
#undef BOOST_ALL_DYN_LINK
#include "boost/test/included/unit_test.hpp"
#include "fastlib/table/default/dense/unlabeled/kdtree/table.h"
#include "fastlib/table/default/dense/unlabeled/balltree/table.h"
#include "fastlib/table/default/dense/labeled/kdtree/table.h"
#include "fastlib/table/default/dense/labeled/balltree/table.h"
#include "fastlib/table/default/dense_sparse/labeled/balltree/table.h"
#include "mlpack/clustering/kmeans_dev.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/data/multi_dataset_dev.h"

class TestKMeans {

private:

	template <typename Table>
	struct KMeansArguments {
		typedef Table TableType;
		typedef fl::math::LMetric<2> MetricType;
		typedef fl::table::dense::unlabeled::balltree::Table CentroidTableType;
	};

	template <typename TableType>
	static bool AreSame(TableType* first_table, TableType* second_table, int num_points, int num_dim) {
		for (int i = 0; i < num_points; i++) {
			typename TableType::Point_t first;
			typename TableType::Point_t second;
			first_table->get(i, &first);
			second_table->get(i, &second);

			for (int j = 0; j < num_dim; j++) {
				if (fabs(first[j] - second[j]) > fabs(EPSILON * first[j])) {
					return false;
				}
			}
		}
		return true;
	}

	/**
	* We provide both table and input_file. The input file is used to
	* run a local naive version of kmeans.
	*/
	template <typename TableType, typename KMeansArgs>
	void RunDenseTest(TableType& table, std::string input_file, int k) {
		fl::logger->Message() << "K is "<< k;
		typedef typename TableType::Point_t Point_t;
		typedef typename fl::table::dense::unlabeled::balltree::Table::Point_t CentroidPoint_t;
		typedef TableType Table_t;

		// STARTING CENTROIDS
		CentroidPoint_t* starting_centroids = new CentroidPoint_t[k];
		fl::ml::KMeans<KMeansArgs>::AssignInitialCentroids(k, starting_centroids, &table);

		// RESULTS TABLES
		std::vector<index_t> dense_sizes(1, table.n_attributes()); 
		std::vector<index_t> sparse_sizes; 
		fl::table::dense::unlabeled::balltree::Table algo_naive_results;
		fl::table::dense::unlabeled::balltree::Table algo_tree_results;
		fl::table::dense::unlabeled::balltree::Table local_naive_results;
		algo_naive_results.Init(dense_sizes, sparse_sizes, k);
		algo_tree_results.Init(dense_sizes, sparse_sizes, k);
		local_naive_results.Init(dense_sizes, sparse_sizes, k);

		// RUN NAIVE
		fl::ml::KMeans<KMeansArgs> kmeans_;
		kmeans_.Init(k, &table, starting_centroids);
		kmeans_.RunKMeans("naive");
		kmeans_.GetCentroids(&algo_naive_results);

		// RUN TREE
		fl::ml::KMeans<KMeansArgs> kmeans_new;
		kmeans_new.Init(k, &table, starting_centroids);
		kmeans_new.RunKMeans("tree");
		kmeans_new.GetCentroids(&algo_tree_results);

		// COMPARE TREE AND NAIVE
		BOOST_ASSERT(AreSame(&algo_naive_results, &algo_tree_results, k, table.n_attributes()));
		fl::logger->Message() <<"Algorithm Naive matches Algorithm Tree.";

		// RUN LOCAL NAIVE
		TableType local_table;
		local_table.Init(input_file, "r");
		std::vector<CentroidPoint_t> local_results;
		local_results.resize(k);
		LocalNaiveKMeans<TableType, double, CentroidPoint_t>(local_table, starting_centroids,
			local_results, k, local_table.n_attributes());
		CentroidPoint_t point;
		for (int i = 0; i < k; i++) {
			local_naive_results.get(i, &point);
			point.CopyValues(local_results[i]);
		}

		// COMPARE local naive and algorithm tree
		BOOST_ASSERT(AreSame(&local_naive_results, &algo_tree_results, k, local_table.n_attributes()));
		fl::logger->Message() << "Algorithm Tree matches Local Naive.";
		delete[] starting_centroids;
	}

	template <typename TableType, typename Precision, typename CentroidPoint_t>
	void LocalNaiveKMeans(TableType& table,
		const CentroidPoint_t* starting_pts,
		std::vector<CentroidPoint_t>& results,
		int k, int dim) {
			typedef typename TableType::Point_t Point_t;
			CentroidPoint_t* centroids = new CentroidPoint_t[k];
			CentroidPoint_t* centroids_copy = new CentroidPoint_t[k];
			int* counts = new int[k];
			for (int i = 0; i < k; i++) {
				centroids[i].Copy(starting_pts[i]); // dirty way of initializing the point
				centroids_copy[i].Copy(starting_pts[i]);
			}
			std::vector<index_t> assignments;
			assignments.assign(table.n_entries(), -1);
			while (true) {
				for (int i = 0; i < k; i++) {
					counts[i] = 0;
					centroids[i].SetAll(0);
				}
				Point_t point;
				bool something_changed = false;
				for (int i = 0; i < table.n_entries(); i++) {
					Precision min_dist = INFINITY;
					int min_dist_centroid = -1;
					table.get(i, &point); // which centroid to belong to
					for (int p = 0; p < k; p++) {
						Precision dist = 0;
						for (int j = 0; j < dim; j++) {
							dist += (point[j] - centroids_copy[p][j])
								* (point[j] - centroids_copy[p][j]);
						}
						if (min_dist > dist) {
							min_dist = dist;
							min_dist_centroid = p;
						}
					}
					if(assignments[i] != min_dist_centroid) {
						something_changed = true;	
					}
					assignments[i] = min_dist_centroid;
					fl::la::AddTo(point, &centroids[min_dist_centroid]);
					counts[min_dist_centroid]++;
				}
				
				for (int i = 0; i < k; i++) {
					fl::la::SelfScale((Precision)1.0 / counts[i], &centroids[i]);
					centroids_copy[i].CopyValues(centroids[i]);
				}
				if (!something_changed) {
					break;
				}
			}// while
			for (int i = 0; i < k; i++) {
				results[i].Copy(centroids_copy[i]); // dirty way of initializing the point
			}
			delete[] counts;
			delete[] centroids;
			delete[] centroids_copy;
	}

	template <typename KMeansArgs>
	void RunDenseTestForMultipleK(std::string fullPathToDataFile) {
		typedef typename KMeansArgs::TableType::template IndexArgs<fl::math::LMetric<2> > IndexArguments;
		index_t k_count = 6;
		index_t k_values[] = {2, 3, 4, 7, 10, 21};
		typename KMeansArgs::TableType table;
		table.Init(fullPathToDataFile, "r"); // initialize table
		IndexArguments q_index_args;
		q_index_args.leaf_size = 20;
		table.IndexData(q_index_args); // build tree

		//for(int i = 0; i < k_count; i++) {
                for(int i = 2; i < 5; i++) { // ##### Note restricted range of i
			RunDenseTest<typename KMeansArgs::TableType, KMeansArgs>(
				table, fullPathToDataFile, k_values[i]);
		}
	}

	template <typename CentroidTableType> 
	void GetCentroidsFromTable(
		const CentroidTableType& table, 
		typename CentroidTableType::Point_t* points,
		int n) {
		typename CentroidTableType::Point_t point;
		for(int i = 0; i < n; i++) {
			table.get(i, &point);
			points[i].Copy(point);
		}
	}

	// will run tests comparing output results to those that
	// have been verified correct. Since kmeans has randomness
	// built in (to choose initial points) we initialize starting
	// points from a file
	template <typename KMeansArgs>
	void RunGoldenTest(
	    std::string init_from_file, 
		std::string data_file,
		std::string correct_file,
		int k) {
		fl::logger->Message() << "Running Golden Test for K=" << k << " for file: "<< data_file.c_str();
		typedef typename fl::table::dense::unlabeled::balltree::Table CentroidTable_t;
		typedef typename fl::table::dense::unlabeled::balltree::Table::Point_t CentroidPoint_t;	
		typedef typename KMeansArgs::TableType Table_t;

		// read starting centroids
		CentroidTable_t starting_centroids;
		starting_centroids.Init(init_from_file, "r");
		CentroidPoint_t centroids[k];
		GetCentroidsFromTable(starting_centroids, centroids, k);
		// read data
		Table_t table;
		table.Init(data_file, "r");
		typename KMeansArgs::TableType::template IndexArgs<fl::math::LMetric<2> > q_index_args;
		q_index_args.leaf_size = 20;
		table.IndexData(q_index_args); // build tree
		// run KMeans
		fl::ml::KMeans<KMeansArgs> kmeans_;
		kmeans_.Init(k, &table, centroids);
		kmeans_.RunKMeans("tree");
		// get the results
		CentroidTable_t results;
		std::vector<index_t> dense_sizes(1, table.n_attributes()); 
		std::vector<index_t> sparse_sizes; 
		results.Init(dense_sizes, sparse_sizes, k);
		kmeans_.GetCentroids(&results);
		// get correct results
		CentroidTable_t correct_centroids;
		correct_centroids.Init(correct_file, "r");
		BOOST_ASSERT(AreSame(&results, &correct_centroids, k, table.n_attributes()));
	}

public:

	void RunTests() {
		RunDenseTestForMultipleK<
			KMeansArguments <
				fl::table::dense::unlabeled::balltree::Table> >
				(input_files_directory_ + "/sdssdr6_38k_wolables.dat");	// plain CSV tests
		RunDenseTestForMultipleK <
			KMeansArguments <
				fl::table::dense::unlabeled::kdtree::Table> >
				(input_files_directory_ + "/sdssdr6_38k_wolables.dat");	// plain CSV tests
		// RunDenseTestForMultipleK <
		// 	KMeansArguments <
		// 		fl::table::dense::labeled::kdtree::Table> >
		// 		(input_files_directory_ + "/phy_train.fl");	// plain CSV tests
		// RunDenseTestForMultipleK <
		// 	KMeansArguments <
		// 		fl::table::dense::labeled::balltree::Table> >
		// 		(input_files_directory_ + "/phy_train.fl");	// plain CSV tests
		std::string k_values[] = { "2", "5", "11" };
		for(int i = 1; i < 2; i++) { // ##### Note restricted range of i
			RunGoldenTest<KMeansArguments <
				fl::table::dense::labeled::balltree::Table> >
				(input_files_directory_ + "/dense_" + k_values[i] + "_initial_phy", 
				  input_files_directory_ + "/phy_train.fl", 
				  input_files_directory_ + "/dense_" + k_values[i] + "_final_phy", 
				  atoi(k_values[i].c_str()));

			RunGoldenTest<KMeansArguments <
				fl::table::dense::labeled::kdtree::Table> >
				(input_files_directory_ + "/dense_" + k_values[i] + "_initial_phy", 
				  input_files_directory_ + "/phy_train.fl", 
				  input_files_directory_ + "/dense_" + k_values[i] + "_final_phy", 
				  atoi(k_values[i].c_str()));
		}
		
		// dense sparse
		k_values[0] = "3";
		k_values[1] = "6";
		k_values[2] = "10";
		for(int i = 1; i < 2; i++) { // ##### Note restricted range of i
			RunGoldenTest<KMeansArguments <
				fl::table::dense_sparse::labeled::balltree::Table> >
				(input_files_directory_ + "/dense_sparse_" + k_values[i] + "_initial_phy", 
				  input_files_directory_ + "/dense_sparse_phy", 
				  input_files_directory_ + "/dense_sparse_" + k_values[i] + "_final_phy", 
				  atoi(k_values[i].c_str()));
		}
	}

	TestKMeans(std::string input_files_directory_in) {
		input_files_directory_ = input_files_directory_in;
	}

private:

	std::string input_files_directory_;

	static const double EPSILON = 0.001;
};

class KMeansTestSuite : public boost::unit_test_framework::test_suite {
public:

	KMeansTestSuite(std::string input_files_dir_in)
		: boost::unit_test_framework::test_suite("K-Means test suite") {
			// create an instance of the test cases class
			boost::shared_ptr<TestKMeans> instance(new TestKMeans(input_files_dir_in));
			// create the test cases
			boost::unit_test_framework::test_case* kmeans_test_case
				= BOOST_CLASS_TEST_CASE(&TestKMeans::RunTests, instance);
			// add the test cases to the test suite
			add(kmeans_test_case);
	}
};


boost::unit_test_framework::test_suite*
init_unit_test_suite(int argc, char** argv) {
	fl::logger->SetLogger("debug");
	// create the top test suite
	boost::unit_test_framework::test_suite* top_test_suite
		= BOOST_TEST_SUITE("K-Means tests");
	if (argc != 2) {
		fl::logger->Message() << "Wrong number of arguments for kmeans test. Expected test input files directory. Returning NULL.";
		return NULL;
	}
	// add test suites to the top test suite
	std::string input_files_directory = argv[1];
        input_files_directory += "/kmeans/";
	top_test_suite->add(new KMeansTestSuite(input_files_directory));
	return top_test_suite;
}

