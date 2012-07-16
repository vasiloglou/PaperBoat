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

#ifndef FL_LITE_FASTLIB_TRILINOS_WRAPPERS_LINEAR_PROBLEM_H
#define FL_LITE_FASTLIB_TRILINOS_WRAPPERS_LINEAR_PROBLEM_H

#include "fastlib/trilinos_wrappers/linear_operator.h"

namespace Anasazi {
class LinearProblem {
  private:

    /** @brief The communication object.
     */
#ifdef EPETRA_MPI
    Epetra_MpiComm comm_;
#else
    Epetra_SerialComm comm_;
#endif

    /** @brief The map object.
     */
    Epetra_Map *map_;

    /** @brief The left hand side.
     */
    Epetra_MultiVector *left_hand_side_;

    /** @brief The right hand side.
     */
    Epetra_MultiVector *right_hand_side_;

    /** @brief The main linear problem.
     */
    Epetra_LinearProblem *linear_problem_;

    /** @brief The linear operator used by the linear problem.
     */
    LinearOperator *linear_operator_;

  public:

    void Init() {

      // Allocate the left hand side.
      left_hand_side_ = new Epetra_MultiVector(
        linear_operator_->OperatorDomainMap(), 1, true);

      // Allocate the right hand side.
      right_hand_side_ = new Epetra_MultiVector(
        linear_operator_->OperatorDomainMap(), 1, true);

      // Make the linear problem.
      linear_problem_ = new Epetra_LinearProblem();
      linear_problem_->SetOperator(linear_operator_);
      linear_problem_->SetLHS(left_hand_side_);
      linear_problem_->SetRHS(right_hand_side_);
    }
};
};

#endif
