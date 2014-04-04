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

#ifndef FL_LITE_MLPACK_KERNEL_PCA_MATRIX_FREE_KERNEL_PCA_H
#define FL_LITE_MLPACK_KERNEL_PCA_MATRIX_FREE_KERNEL_PCA_H

#include <deque>
#include "boost/program_options.hpp"
#include "mlpack/kernel_pca/kernel_pca_result.h"

namespace fl {
namespace ml {

template < typename TableType,
bool do_centering = false >
class MatrixFreeKernelPca: boost::noncopyable {

  private:

    TableType *reference_table_;

  public:

    void Init(TableType &reference_table_in);

    template<typename KernelType, typename ResultType>
    void Train(const KernelType &kernel, int num_components,
               ResultType *result);
};

template<bool do_centering>
class MatrixFreeKernelPca<boost::mpl::void_, do_centering>
      : boost::noncopyable {
  public:
    template<typename TableType>
    struct Core {

      template < typename DataAccessType, typename MetricType,
      typename KernelType, typename ResultType >
      static void Branch(
        int level, DataAccessType *data,
        boost::program_options::variables_map &vm,
        TableType &table, MetricType *metric, KernelType *kernel,
        int num_components, ResultType *result);

      template<typename DataAccessType>
      static int Main(DataAccessType *data,
                      boost::program_options::variables_map &vm);
    };

    static bool ConstructBoostVariableMap(
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm);

    template<typename DataAccessType, typename BranchType>
    static int Main(DataAccessType *data,
                    const std::vector<std::string> &args);
};
};
};


#endif
