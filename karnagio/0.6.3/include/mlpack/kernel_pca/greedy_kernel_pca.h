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
/** @file greedy_kernel_pca.h
 *
 * @conference{ouimet2005greedy,
 *  title={{Greedy spectral embedding}},
 *  author={Ouimet, M. and Bengio, Y.},
 *  booktitle={Proceedings of the 10th International Workshop on Artificial
 *             Intelligence and Statistics},
 *  pages={253--260},
 *  year={2005}
 * }
 */
#ifndef FL_LITE_MLPACK_KERNEL_PCA_GREEDY_KERNEL_PCA_H
#define FL_LITE_MLPACK_KERNEL_PCA_GREEDY_KERNEL_PCA_H

#include <deque>
#include "boost/program_options.hpp"
#include "mlpack/kernel_pca/kernel_pca_result.h"

namespace fl {
namespace ml {

/** @brief Defines the interface for the following algorithm:
 *
 *  @conference{ouimet2005greedy,
 *   title={{Greedy spectral embedding}},
 *   author={Ouimet, M. and Bengio, Y.},
 *   booktitle={Proceedings of the 10th International Workshop on
 *              Artificial Intelligence and Statistics},
 *   pages={253--260},
 *   year={2005}
 *  }
 */
template < typename TableType,
bool do_centering = false >
class GreedyKernelPca: boost::noncopyable {

  public:
    template<typename KernelType>
    class Dictionary {

      private:

        const KernelType *kernel_;

        TableType *table_;

        std::vector<int> random_permutation_;

        std::deque<bool> in_dictionary_;

        std::vector<int> point_indices_in_dictionary_;

        std::vector<int> training_index_to_dictionary_position_;

        fl::dense::Matrix<double, false> *current_kernel_matrix_;

        fl::dense::Matrix<double, false> *current_kernel_matrix_inverse_;

        double current_kernel_matrix_inverse_sum_;

        fl::dense::Matrix<double, true>
        *current_kernel_matrix_inverse_row_sum_;

      private:

        void RandomPermutation_(std::vector<int> &permutation);

        void UpdateDictionary_(
          const int &new_point_index,
          const fl::dense::Matrix<double, true> &temp_kernel_vector,
          const double &projection_error,
          const fl::dense::Matrix<double, true>
          &inverse_times_kernel_vector);

      public:

        typedef KernelType Kernel_t;

        typedef TableType Table_t;

        bool in_dictionary(int training_point_index) const {
          return in_dictionary_[training_point_index];
        }

        const bool perform_centering() const {
          return do_centering;
        }

        ~Dictionary() {
          delete current_kernel_matrix_;
          delete current_kernel_matrix_inverse_;
          delete current_kernel_matrix_inverse_row_sum_;
        }

        int position_to_training_index_map(int position) const {
          return random_permutation_[ position ];
        }

        int training_index_to_dictionary_position(int training_index) const {
          return training_index_to_dictionary_position_[training_index];
        }

        int point_indices_in_dictionary(int nth_dictionary_point_index) const {
          return point_indices_in_dictionary_[nth_dictionary_point_index];
        }

        void ComputeProjections(
          const KernelType &args,
          fl::dense::Matrix<double, false> *projections);

        void Init(TableType &table_in,
                  const KernelType &kernel);

        void AddBasis(const int &iteration_number,
                      const KernelType &kernel);

        const std::vector<int> *basis_set() {
          return &point_indices_in_dictionary_;
        }

        const Kernel_t &kernel() const {
          return *kernel_;
        }

        const Table_t &table() const {
          return *table_;
        }

        int size() const;

        fl::dense::Matrix<double, false> *current_kernel_matrix();

        fl::dense::Matrix<double, false> *current_kernel_matrix_inverse();
    };

  public:
    template<typename KernelType, typename ResultType>
    static void Train(TableType &training_set,
                      const KernelType &kernel,
                      const int &num_components,
                      ResultType *result);
};

template<bool do_centering>
class GreedyKernelPca<boost::mpl::void_, do_centering>
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

