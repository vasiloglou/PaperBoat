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
#ifndef FL_LITE_FASTLIB_TABLE_BRANCH_ON_TABLE_TINY_DEV_H_
#define FL_LITE_FASTLIB_TABLE_BRANCH_ON_TABLE_TINY_DEV_H_
//#include <omp.h>
#include "fastlib/table/branch_on_table_tiny.h"
#include "boost/program_options.hpp"
#include "fastlib/table/table_defs.h"
#include "fastlib/table/default/dense/labeled/kdtree/table.h"
#include "fastlib/data/linear_algebra.h"

namespace fl {
namespace table {
  template<typename AlgorithmType, typename DataAccessType>
  int Branch::BranchOnTable(DataAccessType *data,
                           boost::program_options::variables_map &vm) {

    std::string log = vm["log"].as<std::string>();
    std::string loglevel = vm["loglevel"].as<std::string>();
    if (vm.count("cores")) {
      fl::logger->Message() << "Using " << vm["cores"].as<int>()
      << " cores for computations";
//        omp_set_num_threads(vm["cores"].as<int>());
//        omp_set_nested(false);
    }
    else {
      fl::logger->Message() << "Using one core for computations";
    }
    if (log!="user_defined") {
      fl::logger->SetLogger(loglevel);
      fl::logger->Init(log);
    }
    
    if (!vm.count("point")) {
      try {
        fl::table::dense::labeled::kdtree::Table table;
        table.data()->TryToInit(vm["references_in"].as<std::string>());
        if (vm["tree"].as<std::string>() == "kdtree") {
          return AlgorithmType::template Core <
               fl::table::dense::labeled::kdtree::Table >::Main(data, vm);
        } else {
           return AlgorithmType::template Core <
               fl::table::dense::labeled::kdtree::Table >::Main(data, vm);
        }
      }
      catch(const fl::TypeException &e) {
        
      }
  
      fl::logger->Die() << "This option " << vm["tree"].as<std::string>()
          << " for tree is not supported for the type of the data in the file. "
          << " Most likely you are trying to use a kdtree for sparse data";
    } else {
      // not all the algorithms are using tree, or maybe they are using their own
      // custom tree like quicsvd
      if (vm["point"].as<std::string>() == "dense") {
        if (vm["tree"].as<std::string>() == "kdtree") {
          return AlgorithmType::template Core <
                 fl::table::dense::labeled::kdtree::Table >::Main(data, vm);
        }
        fl::logger->Die() << "This option " << vm["tree"].as<std::string>()
        << " for tree is not supported";
      }
  
  
    }
    return -1;
  }

}} // namespaces

#endif


