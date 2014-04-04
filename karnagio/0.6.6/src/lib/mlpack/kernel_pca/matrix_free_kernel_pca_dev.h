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
#ifndef FL_LITE_MLPACK_KERNEL_PCA_MATRIX_FREE_KERNEL_PCA_DEV_H
#define FL_LITE_MLPACK_KERNEL_PCA_MATRIX_FREE_KERNEL_PCA_DEV_H

#include "mlpack/kernel_pca/matrix_free_kernel_pca.h"
#include "mlpack/kernel_pca/matrix_free_kernel_pca_defs.h"
#include "mlpack/kernel_pca/dense_kernel_matrix_inverse.h"
#include "fastlib/trilinos_wrappers/kernel_linear_operator.h"

#undef F77_FUNC
#include "trilinos/AnasaziEpetraAdapter.hpp"
#include "trilinos/AnasaziBasicEigenproblem.hpp"
#include "trilinos/AnasaziBlockKrylovSchurSolMgr.hpp"
#include "trilinos/AnasaziBasicSort.hpp"
#include "trilinos/AztecOO.h"
#include "trilinos/Epetra_BlockMap.h"
#include "trilinos/Epetra_CrsMatrix.h"
#include "trilinos/Epetra_DataAccess.h"
#include "trilinos/Epetra_LinearProblem.h"
#include "trilinos/Epetra_Map.h"
#include "trilinos/Epetra_MultiVector.h"
#include "trilinos/Epetra_Operator.h"
#include "fastlib/data/monolithic_point.h"

#ifdef EPETRA_MPI
#include "trilinos/Epetra_MpiComm.h"
#else
#include "trilinos/Epetra_SerialComm.h"
#endif

#include "trilinos/Epetra_Vector.h"
#include "boost/utility.hpp"
#include "fastlib/la/linear_algebra.h"

namespace fl {
namespace ml {

template<typename TableType, bool do_centering>
void MatrixFreeKernelPca<TableType, do_centering>::Init(
  TableType &reference_table_in) {

  reference_table_ = &reference_table_in;
}

template<typename TableType, bool do_centering>
template<typename KernelType, typename ResultType>
void MatrixFreeKernelPca<TableType, do_centering>::Train(
  const KernelType &kernel,
  int num_components,
  ResultType *result) {

  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef Anasazi::MultiVecTraits<double, MV> MVT;
  typedef Anasazi::KernelLinearOperator<TableType, KernelType, do_centering, false> LeftOP;

  // Allocate the result.
  if (result->IsInitialized() == false) {
    result->Init(*reference_table_, num_components);
  }

  // The matrix free linear operator for the Trilinos eigensolver.
#ifdef EPETRA_MPI
  Epetra_MpiComm comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm comm;
#endif
  Epetra_Map map(reference_table_->n_entries(), 0, comm);
  Teuchos::RCP<LeftOP> linear_operator =
  Teuchos::rcp(new LeftOP(*reference_table_, kernel, comm, map));

  // A sort manager to be passed to the solver manager.
  std::string which("LM");
  Teuchos::RCP<Anasazi::SortManager<double> > my_sort =
    Teuchos::rcp(new Anasazi::BasicSort<double>(which));

  // The parameter list to pass into solver manager.
  int block_size = num_components;
  // int num_blocks = 20;
  int max_restarts = 500;
  double tol = 1e-10;
  int verbosity = Anasazi::Errors + Anasazi::Warnings +
                  Anasazi::FinalSummary + Anasazi::TimingDetails;

  Teuchos::ParameterList parameter_list;
  parameter_list.set("Verbosity", verbosity);
  parameter_list.set("Sort Manager", my_sort);
  parameter_list.set("Block Size", block_size);
  // parameter_list.set("Num Blocks", num_blocks);
  parameter_list.set("Maximum Restarts", max_restarts);
  parameter_list.set("Convergence Tolerance", tol);

  // The set of eigenvectors that will be computed by Anasazi.
  Teuchos::RCP<Epetra_MultiVector> ivec =
    Teuchos::rcp(new Epetra_MultiVector(map, block_size));

  // Start at the random configuration for the eigenvectors.
  ivec->Random();

  Teuchos::RCP< Anasazi::BasicEigenproblem<double, MV, OP> > problem =
    Teuchos::rcp(new Anasazi::BasicEigenproblem<double, MV, OP>(
                   linear_operator, ivec));
  problem->setHermitian(true);
  problem->setNEV(num_components);
  problem->setProblem();

  // Solve the problem, and extract the answers.
  try {
    Anasazi::BlockKrylovSchurSolMgr<double, MV, OP>
    solver_mgr = Anasazi::BlockKrylovSchurSolMgr<double, MV, OP>(
                   problem, parameter_list);
    Anasazi::ReturnType return_code = solver_mgr.solve();

    if (return_code != Anasazi::Converged) {
      fl::logger->Message() <<
      "[*] Anasazi::EigensolverMgr::solve() returned unconverged.\n";
    }
    else {
      fl::logger->Message() <<
      "[*] Anasazi::EigensolverMgr::solve() converged.\n";
    }
  }
  catch (const std::exception &exception) {
    fl::logger->Message() <<
    "[*] Anasazi::EigensolverMgr bailed out.\n";
    fl::logger->Die() << "Trilinos has a bug.";
  }

  // Initialize the result to be returned and copy back from the
  // Trilinos result to the fastlib structure.
  Anasazi::Eigensolution<double, MV> sol = problem->getSolution();
  std::vector<Anasazi::Value<double> > evals = sol.Evals;
  MV *evecs = sol.Evecs.get();

  // Normalize the eigenvalues by the number of training points.
  for (int i = 0; i < num_components; i++) {
    typename ResultType::ResultTableType::Point_t point1;
    result->principal_eigenvalues()->get(i, &point1);
    point1.set(0, evals[i].realpart / ((double) reference_table_->n_entries()));
  }
  for (int i = 0; i < num_components; i++) {
    for (int j = 0; j < reference_table_->n_entries(); j++) {
      typename ResultType::ResultTableType::Point_t point;
      result->principal_components()->get(j, &point);
      point.set(i, evecs->Pointers()[i][j]);
    }
  }
}
};
};

#endif
