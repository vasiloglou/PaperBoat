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

#ifndef FL_LITE_FASTLIB_TABLE_R_BRANCH_ON_TABLE_H_
#define FL_LITE_FASTLIB_TABLE_R_BRANCH_ON_TABLE_H_
#include "boost/program_options.hpp"
#include "fastlib/table/default/dense/labeled/kdtree/table.h"
#include "fastlib/table/default/dense/labeled/balltree/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/table.h"
#include "fastlib/data/linear_algebra.h"

namespace fl {
namespace table {
class Branch {
  public:
    template<typename AlgorithmType, typename DataAccessType>
    static int BranchOnTable(DataAccessType *data,
                             boost::program_options::variables_map &vm) {

      std::string log = vm["log"].as<std::string>();
      std::string loglevel = vm["loglevel"].as<std::string>();
      if (vm.count("cores")) {
        fl::logger->Message() << "Using " << vm["cores"].as<int>()
        << " cores for computations";
      }
      else {
        fl::logger->Message() << "Using one core for computations";
      }
      // not all the algorithms are using tree, or maybe they are using their own
      // custom tree like quicsvd

      if (vm["point"].as<std::string>() == "dense") {
        if (vm["tree"].as<std::string>() == "balltree" ||
            vm["tree"].as<std::string>() == "cosine") {
          return AlgorithmType::template Core <
                 fl::table::dense::labeled::balltree::Table >::Main(data, vm);
        }
        if (vm["tree"].as<std::string>() == "kdtree") {
          return AlgorithmType::template Core <
                 fl::table::dense::labeled::kdtree::Table >::Main(data, vm);
        }
        fl::logger->Die() << "This option " << vm["tree"].as<std::string>()
        << " for tree is not supported";
      }

      if (vm["point"].as<std::string>() == "sparse") {
        if (vm["tree"].as<std::string>() == "balltree") {
          return AlgorithmType::template Core <
                 fl::table::sparse::labeled::balltree::Table >::Main(data, vm);
        }
        if (vm["tree"].as<std::string>() == "kdtree") {
          fl::logger->Die() << "KdTree is not supported for sparse data";
        }
        fl::logger->Die() << "This option " << vm["tree"].as<std::string>()
        << " for tree is not supported";
      }


      fl::logger->Die() << "This option " << vm["point"].as<std::string>()
      << " for data point is not supported for R";
      return -1;
    }
}; // class Branch

}
} // namespaces

#endif

