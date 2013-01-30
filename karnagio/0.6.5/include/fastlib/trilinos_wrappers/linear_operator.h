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

#ifndef FL_LITE_FASTLIB_TRILINOS_WRAPPERS_LINEAR_OPERATOR_H
#define FL_LITE_FASTLIB_TRILINOS_WRAPPERS_LINEAR_OPERATOR_H

#undef F77_FUNC
#include "AnasaziEpetraAdapter.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziBlockKrylovSchurSolMgr.hpp"
#include "AnasaziBasicSort.hpp"
#include "AztecOO.h"
#include "Epetra_BlockMap.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_DataAccess.h"
#include "Epetra_LinearProblem.h"
#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Operator.h"

#ifdef EPETRA_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "Epetra_Vector.h"
#undef F77_FUNC
#include "fastlib/data/monolithic_point.h"
#include <vector>

namespace Anasazi {

class LinearOperator: public virtual Epetra_Operator {

  protected:

#ifdef EPETRA_MPI
    const Epetra_MpiComm *comm_;
#else
    const Epetra_SerialComm *comm_;
#endif

    const Epetra_Map *map_;

  public:

    virtual ~LinearOperator() {
    }

    LinearOperator() {
      comm_ = NULL;
      map_ = NULL;
    }

#ifdef EPETRA_MPI
    LinearOperator(const Epetra_MpiComm &comm_in,
                   const Epetra_Map &map_in) {
      comm_ = &comm_in;
      map_ = &map_in;
    }
#else
    LinearOperator(const Epetra_SerialComm &comm_in,
                   const Epetra_Map &map_in) {
      comm_ = &comm_in;
      map_ = &map_in;
    }
#endif

    virtual int Apply(const Epetra_MultiVector &vec,
                      Epetra_MultiVector &prod) const = 0;

    int SetUseTranspose(bool use_transpose) {
      return -1;
    }

    int ApplyInverse(const Epetra_MultiVector &X,
                     Epetra_MultiVector &Y) const {
      return -1;
    }

    double NormInf() const {
      return -1;
    }

    const char *Label() const {
      return "Generic linear operator";
    }

    bool UseTranspose() const {
      return false;
    }

    bool HasNormInf() const {
      return false;
    }

    const Epetra_Comm &Comm() const {
      return *comm_;
    }

    const Epetra_Map &OperatorDomainMap() const {
      const Epetra_Map &map_reference = *map_;
      return map_reference;
    }

    const Epetra_Map &OperatorRangeMap() const {
      const Epetra_Map &map_reference = *map_;
      return map_reference;
    }

    void PrintDebug(const char *name = "", FILE *stream = stderr) const {
      fprintf(stream, "----- MATRIX ------: %s\n", name);
      for (int r = 0; r < this->n_rows(); r++) {
        for (int c = 0; c < this->n_cols(); c++) {
          fprintf(stream, "%+3.3f ", this->get(r, c));
        }
        fprintf(stream, "\n");
      }
    }

    virtual int n_rows() const = 0;

    virtual int n_cols() const = 0;

    virtual double get(int row, int col) const = 0;
};
};

#endif
