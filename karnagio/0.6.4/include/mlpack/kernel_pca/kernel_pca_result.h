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
#ifndef FL_LITE_MLPACK_KERNEL_PCA_KERNEL_PCA_RESULT_H
#define FL_LITE_MLPACK_KERNEL_PCA_KERNEL_PCA_RESULT_H

#include "fastlib/dense/matrix.h"
#include "boost/utility.hpp"

namespace fl {
namespace ml {
template<typename TableType1, typename TableType2>
class KernelPcaResult: boost::noncopyable {

  private:

    TableType1 *table_;

    boost::shared_ptr<TableType2> principal_components_;

    boost::shared_ptr<TableType2> principal_eigenvalues_;

  public:

    typedef TableType2 ResultTableType;

    KernelPcaResult() : table_(NULL), principal_components_(new TableType2()),
        principal_eigenvalues_(new TableType2())
    {

    }

    boost::shared_ptr<TableType2> principal_components() {
      return principal_components_;
    }

    const boost::shared_ptr<TableType2> principal_components() const {
      return principal_components_;
    }

    boost::shared_ptr<TableType2> principal_eigenvalues() {
      return principal_eigenvalues_;
    }

    const boost::shared_ptr<TableType2> principal_eigenvalues() const {
      return principal_eigenvalues_;
    }

    void Init(TableType1 &table_in,
              const int &num_components) {
      table_ = &table_in;
      principal_components_->Init(
        std::vector<index_t>(1, num_components),
        std::vector<index_t>(),
        table_->n_entries());
      principal_eigenvalues_->Init(
        std::vector<index_t>(1, num_components),
        std::vector<index_t>(),
        1);
    }

    bool IsInitialized() const {
      return table_ != NULL;
    }

    template<typename DataAccessType>
    void Init(TableType1 &table_in,
              const int &num_components,
              const std::string components_name,
              const std::string eigenvalues_name,
              DataAccessType *data) {
      table_ = &table_in;
      data->Attach(components_name,
                   std::vector<index_t>(1, num_components),
                   std::vector<index_t>(),
                   table_->n_entries(),
                   &principal_components_);
      data->Attach(eigenvalues_name,
                   std::vector<index_t>(1, 1),
                   std::vector<index_t>(),
                   num_components,
                   &principal_eigenvalues_);
    }

};
};
};

#endif
