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
#ifndef MLPACK_KERNEL_PCA_MATRIX_FREE_KERNEL_PCA_DEFS_H
#define MLPACK_KERNEL_PCA_MATRIX_FREE_KERNEL_PCA_DEFS_H

#include <iostream>
#include "boost/program_options.hpp"
#include "fastlib/base/base.h"
#include "fastlib/metric_kernel/gaussian_dotprod.h"
#include "fastlib/metric_kernel/polynomial_dotprod.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "fastlib/table/default/dense/unlabeled/kdtree/compact/table.h"
#include "mlpack/kernel_pca/matrix_free_kernel_pca.h"
#include "mlpack/kernel_pca/kernel_pca_private.h"

namespace fl {
namespace ml {

template<bool do_centering>
template<typename TableType>
template < typename DataAccessType, typename MetricType, typename KernelType,
typename ResultType >
void MatrixFreeKernelPca<boost::mpl::void_, do_centering>::
Core<TableType>::Branch(
  int level, DataAccessType *data, boost::program_options::variables_map &vm,
  TableType &table, MetricType *metric, KernelType *kernel,
  int num_components, ResultType *result) {

  typedef typename TableType::CalcPrecision_t CalcPrecision_t;

  switch (level) {
    case 0:

      // Branch on the --mean_center argument.
      if (vm.count("mean_center") > 0) {
        fl::logger->Message() << "Mean-centered KPCA.";
        fl::ml::MatrixFreeKernelPca<boost::mpl::void_, true>::template
        Core<TableType>::Branch(
          level + 1, data, vm, table, metric, kernel, num_components, result);
      }
      else {
        fl::logger->Message() << "Non-meancentered KPCA.";
        fl::ml::MatrixFreeKernelPca<boost::mpl::void_, false>::template
        Core<TableType>::Branch(
          level + 1, data, vm, table, metric, kernel, num_components, result);
      }
      break;
    case 1:

      // Branch on the --metric argument.

      if (vm["metric"].as<std::string>() == std::string("l2")) {
        fl::logger->Message() << "L2 metric selected.";

        fl::math::LMetric<2> new_lmetric;

        fl::ml::MatrixFreeKernelPca<boost::mpl::void_, do_centering>::template
        Core<TableType>::Branch(
          level + 1, data, vm, table, &new_lmetric, kernel, num_components,
          result);
      }
      else if (vm["metric"].as<std::string>() == std::string("weighted_l2")) {
        fl::logger->Message() << "Weighted L2 metric selected.";

        // If it is a weighted metric, we need to read in the weights.
        if (vm.count("metric_weights_in") == 0) {
          fl::logger->Die() << "You want weighted metric and have not "
          "specified the file holding the weights!\n";
        }
        fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double> >
        wmetric;
        boost::shared_ptr<typename DataAccessType::DefaultTable_t> weights_table;
        typename DataAccessType::DefaultTable_t::Point_t weights_point;
        data->Attach(vm["metric_weights_in"].as<std::string>(), &weights_table);
        if (weights_table->n_entries() != 1) {
          fl::logger->Die() << "The file with the weights must have a point on "
          << " a single line";
        }
        weights_table->get(0, &weights_point);
        wmetric.set_weights(weights_point.template dense_point<double>());

        fl::ml::MatrixFreeKernelPca<boost::mpl::void_, do_centering>::template
        Core<TableType>::Branch(
          level + 1, data, vm, table, &wmetric, kernel, num_components,
          result);
      }

      break;
    case 2:

      // Branch on the --kernel argument.
      if (vm["kernel"].as<std::string>() == "gaussian") {
        fl::logger->Message() << "Gaussian kernel selected.";
        fl::ml::KernelInitializer < do_centering, CalcPrecision_t, MetricType,
        fl::math::GaussianDotProduct<CalcPrecision_t, MetricType>,
        MatrixFreeKernelPca<boost::mpl::void_, do_centering> >
        initializer(level, data, vm, table, metric, num_components, result);
      }
      else if (vm["kernel"].as<std::string>() == "linear") {
        fl::logger->Message() << "Linear kernel selected.";
        fl::ml::KernelInitializer < do_centering, CalcPrecision_t, MetricType,
        fl::math::PolynomialDotProduct<CalcPrecision_t, MetricType>,
        MatrixFreeKernelPca<boost::mpl::void_, do_centering> >
        initializer(level, data, vm, table, metric, num_components, result);
      }
      else {
        fl::logger->Die() << "Invalid kernel type.";
      }
      break;
    default:

      // Call the actual algorithm.
      fl::ml::MatrixFreeKernelPca<TableType, do_centering> matrix_free_kpca;
      matrix_free_kpca.Init(table);
      matrix_free_kpca.Train(*kernel, num_components, result);
      break;
  }
}

template<bool do_centering>
template<typename TableType>
template<typename DataAccessType>
int MatrixFreeKernelPca<boost::mpl::void_, do_centering>::Core<TableType>::Main(
  DataAccessType *data,
  boost::program_options::variables_map &vm) {

  // Read in the dataset.
  boost::shared_ptr<TableType> table;
  typedef typename TableType::CalcPrecision_t CalcPrecision_t;

  // The reference data file is a required parameter.
  fl::logger->Message() << "Reading in the training dataset..." << std::endl;
  std::string references_file_name = vm["references_in"].as<std::string>();
  data->Attach(references_file_name, &table);
  fl::logger->Message() << "Finished reading the training dataset..." <<
  std::endl;

  // Kernel PCA result.
  fl::ml::KernelPcaResult<TableType, typename DataAccessType::DefaultTable_t>
  result;
  int num_components = vm["k_eigenvectors"].as<int>();
  std::string kpca_components_file_name =
    vm["eigenvectors_out"].as<std::string>();
  std::string kpca_eigenvalues_file_name =
    vm["eigenvalues_out"].as<std::string>();
  result.Init(*table, num_components, kpca_components_file_name,
              kpca_eigenvalues_file_name, data);

  fl::logger->Message() << "Computing " << num_components <<
  " kernel principal components...";

  fl::ml::MatrixFreeKernelPca<boost::mpl::void_, true>::template Core
  <TableType>::Branch(
    0, data, vm, *table, (fl::math::LMetric<2> *) NULL,
    (fl::math::GaussianDotProduct<CalcPrecision_t, fl::math::LMetric<2> > *)
    NULL, num_components, &result);

  fl::logger->Message() <<
  "The kernel principal components have been computed...\n";

  // Write out the result.
  fl::logger->Message() << "Emitting the result ..." << std::endl;
  data->Purge(kpca_components_file_name);
  data->Detach(kpca_components_file_name);
  data->Purge(kpca_eigenvalues_file_name);
  data->Detach(kpca_eigenvalues_file_name);
  return 0;
}

template<bool do_centering>
bool MatrixFreeKernelPca<boost::mpl::void_, do_centering>::
ConstructBoostVariableMap(
  const std::vector<std::string> &args,
  boost::program_options::variables_map *vm) {

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////
  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Display help on greedy kernel PCA")
  ("kernel", boost::program_options::value<std::string>()->default_value("gaussian"),
   "The kernel type. One of\n"
   "  gaussian, linear"
  )
  ("algorithm", boost::program_options::value<std::string>()->default_value("matrixfree"),
   "Matrix-free KPCA."
  )
  ("bandwidth", boost::program_options::value<double>(),
   "The bandwidth for the kernel.")
  ("mean_center", "Perform the centered version of KPCA when present.")
  ("references_in", boost::program_options::value<std::string>(),
   "File containing the dataset to perform kernel PCA on.")
  ("metric",
   boost::program_options::value<std::string>()->default_value("l2"),
   "Metric function used by allkn.  One of:\n"
   "  l2, weighted_l2")
  ("metric_weights_in",
   boost::program_options::value<std::string>()->default_value(""),
   "A file containing weights for use with --metric=weighted_l2")
  ("k_eigenvectors", boost::program_options::value<int>(),
   "The number of components to extract.")
  ("eigenvectors_out",
   boost::program_options::value< std::string >()->default_value(
     "eigenvectors.txt"),
   "The output file for the kernel eigenvectors.")
  ("eigenvalues_out",
   boost::program_options::value< std::string> ()->default_value(
     "eigenvalues.txt"),
   "The output file for the extracted kernel principal components.")
  ("point",
   boost::program_options::value<std::string>()->default_value("dense"),
   "Point type used by allkn.  One of:\n"
   "  dense, sparse, dense_sparse, categorical, dense_categorical")
  ("tree",
   boost::program_options::value<std::string>()->default_value("kdtree"),
   "Tree structure used by allkn.  One of:\n"
   "  kdtree, balltree")
  ("log",
   boost::program_options::value<std::string>()->default_value(""),
   "A file to receive the log, or omit for stdout.")
  ("loglevel",
   boost::program_options::value<std::string>()->default_value("debug"),
   "Level of log detail.  One of:\n"
   "  debug: log everything\n"
   "  verbose: log messages and warnings\n"
   "  warning: log only warnings\n"
   "  silent: no logging");

  boost::program_options::store(boost::program_options::
                                command_line_parser(args).options(desc).run(),
                                *vm);
  boost::program_options::notify(*vm);
  if (vm->count("help")) {
    std::cout << fl::DISCLAIMER << "\n";
    std::cout << desc << "\n";
    return true;
  }

  // Check the validity of the parameters.
  if (vm->count("references_in") == 0) {
    fl::logger->Die() << "The --references_in is a required parameter.\n";
  }
  if (vm->count("k_eigenvectors") == 0 ||
      (*vm)["k_eigenvectors"].as<int>() <= 0) {
    fl::logger->Die() << "The --k_eigenvectors argument value needs to be "
    "non-negative.\n";
  }

  return false;
}

template<bool do_centering>
template<typename DataAccessType, typename BranchType>
int MatrixFreeKernelPca<boost::mpl::void_, do_centering>::Main(
  DataAccessType *data,
  const std::vector<std::string> &args) {

  srand(time(NULL));

  boost::program_options::variables_map vm;
  bool help_specified = ConstructBoostVariableMap(args, &vm);

  if (help_specified) {
    return 1;
  }

  return BranchType::template BranchOnTable < MatrixFreeKernelPca<boost::mpl::void_, false>, DataAccessType > (data, vm);
}
};
};

#endif
