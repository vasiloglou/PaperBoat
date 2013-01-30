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

#ifndef FL_LITE_FASTLIB_TRILINOS_WRAPPERS_KERNEL_LINEAR_PROBLEM_H
#define FL_LITE_FASTLIB_TRILINOS_WRAPPERS_KERNEL_LINEAR_PROBLEM_H

#include "fastlib/trilinos_wrappers/kernel_linear_operator.h"
#include "fastlib/trilinos_wrappers/linear_problem.h"

namespace Anasazi {

template<typename TableType>
class KernelLinearProblem: public LinearProblem {

  private:

    /** @brief The associated table of data points.
     */
    TableType *table_;

  public:

    void Init(TableType &table_in) {

      // Store the table.
      table_ = &table_in;

      // Allocate the map.
      map_ = new Epetra_Map(table_in.n_entries(), 0, comm_);

      // Allocate the associated kernel linear operator.
      linear_operator_ = new KernelLinearOperator<TableType>(
        table_in, comm_, *map_);

      // Call the initialization procedure of the parent.
      LinearProblem::Init(table_in.n_entries());
    }
};
};

#endif
