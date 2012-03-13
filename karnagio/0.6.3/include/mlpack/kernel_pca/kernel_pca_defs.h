/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
LLC, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE ISMION INC "AS IS" AND ANY
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
#ifndef FL_LITE_MLPACK_KERNEL_PCA_KERNEL_PCA_DEFS_H_
#define FL_LITE_MLPACK_KERNEL_PCA_KERNEL_PCA_DEFS_H_
#include "kernel_pca.h"
#include "mlpack/kernel_pca/greedy_kernel_pca_dev.h"
#include "mlpack/kernel_pca/matrix_free_kernel_pca_dev.h"
#include "boost/program_options.hpp"
#ifdef EPETRA_MPI
#include "boost/mpi/environment.hpp"
#include "boost/mpi/communicator.hpp"
#endif
#include "fastlib/workspace/task.h"

template <typename DataAccessType, typename BranchType>
int fl::ml::KernelPCA<boost::mpl::void_>::Main(DataAccessType *data, 
    const std::vector<std::string> &args) {
  // Initialize MPI.
  #ifdef EPETRA_MPI
    int argc=args.size();
    char **argv;
    argv=new char*[argc];
    for(int i=0; i<argc; ++i) {
      argv[i]=const_cast<char *>(args[i].c_str());
    }
    MPI_Init(&argc, &argv);
    delete [] argv;
  #endif

  // Use a generic file data access model
  try {

    // First find out which algorithm we are using for KPCA.
    boost::program_options::options_description desc("Available options");
    desc.add_options()
    ("help", "Kernel PCA")
    ("algorithm", boost::program_options::value<std::string>()->default_value("greedy"),
     "matrixfree: matrix-free method \n"
     "greedy: greedy method \n"
     "naive: naive method \n");

    boost::program_options::variables_map vm;
    boost::program_options::store(
      boost::program_options::command_line_parser(args).options(desc).
      allow_unregistered().run(), vm);

    boost::program_options::notify(vm);

    if (vm.count("help")) {
      std::cout << fl::DISCLAIMER << "\n";
      std::cout << desc << "\n";
#ifdef EPETRA_MPI
      MPI_Finalize();
#endif
      return 0;
    }

    if (vm["algorithm"].as<std::string>() == std::string("greedy")) {
      int return_code = fl::ml::GreedyKernelPca<boost::mpl::void_, false>::Main <
                        DataAccessType,
                        BranchType
                        > (data, args);
#ifdef EPETRA_MPI
      MPI_Finalize();
#endif
      return return_code;
    }
    else if (vm["algorithm"].as<std::string>() == std::string("matrixfree")) {
      int return_code = fl::ml::MatrixFreeKernelPca<boost::mpl::void_, false>::Main <
                        DataAccessType,
                        BranchType
                        > (data, args);
#ifdef EPETRA_MPI
      MPI_Finalize();
#endif
      return return_code;
    }
    else {
#ifdef EPETRA_MPI
      MPI_Finalize();
#endif
      fl::logger->Die() << "Unrecognized algorithm option.";
    }
  }
  catch (const fl::Exception &exception) {
#ifdef EPETRA_MPI
    MPI_Finalize();
#endif
    return EXIT_FAILURE;
  }
#ifdef EPETRA_MPI
  MPI_Finalize();
#endif
  return 0;

}

template<typename DataAccessType>
void fl::ml::KernelPCA<boost::mpl::void_>::Run(DataAccessType *data,
      const std::vector<std::string> &args) {
  fl::ws::Task<
    DataAccessType,
    &Main<
      DataAccessType, 
      typename DataAccessType::Branch_t
    > 
  > task(data, args);
  data->schedule(task); 
}


#endif
