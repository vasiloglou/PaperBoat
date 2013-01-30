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
#ifndef FL_LITE_MLPACK_KERNEL_PCA_KERNEL_PCA_PRIVATE_H
#define FL_LITE_MLPACK_KERNEL_PCA_KERNEL_PCA_PRIVATE_H

namespace fl {
namespace ml {

template < bool do_centering, typename CalcPrecision_t, typename MetricType,
typename KernelType, typename AlgorithmType >
class KernelInitializer {
};

template < bool do_centering, typename CalcPrecision_t, typename MetricType,
typename AlgorithmType >
class KernelInitializer < do_centering, CalcPrecision_t, MetricType,
      fl::math::GaussianDotProduct<CalcPrecision_t, MetricType>, AlgorithmType > {

  public:

    template<typename DataAccessType, typename TableType, typename ResultType>
    KernelInitializer(int level, DataAccessType *data,
                      boost::program_options::variables_map &vm,
                      TableType &table, MetricType *metric,
                      int num_components, ResultType *result) {

      // Declare the L2 Gaussian kernel and initialize it.
      fl::math::GaussianDotProduct<CalcPrecision_t, MetricType> l2gaussian;

      if (vm.count("bandwidth") == 0 ||
          vm["bandwidth"].as<double>() <= 0) {
        fl::logger->Die() << "--bandwidth parameter needs to be supplied with"
        " nonnegative number";
      }

      fl::logger->Message() << "Using the bandwidth value of " <<
      vm["bandwidth"].as<double>();
      l2gaussian.Init(vm["bandwidth"].as<double>(), *metric);

      // Call the branch.
      AlgorithmType::template Core<TableType>::Branch(
        level + 1, data, vm, table, metric, &l2gaussian, num_components, result);
    }
};

template < bool do_centering, typename CalcPrecision_t, typename MetricType,
typename AlgorithmType >
class KernelInitializer < do_centering, CalcPrecision_t, MetricType,
      fl::math::PolynomialDotProduct<CalcPrecision_t, MetricType>, AlgorithmType > {

  public:

    template<typename DataAccessType, typename TableType, typename ResultType>
    KernelInitializer(int level, DataAccessType *data,
                      boost::program_options::variables_map &vm,
                      TableType &table, MetricType *metric,
                      int num_components, ResultType *result) {

      // Declare the L2 Gaussian kernel and initialize it.
      fl::math::PolynomialDotProduct<CalcPrecision_t, MetricType> l2poly;
      l2poly.Init(1, metric);

      // Call the branch.
      AlgorithmType::template Core<TableType>::Branch(
        level + 1, data, vm, table, metric, &l2poly, num_components, result);
    }
};
};
};

#endif
