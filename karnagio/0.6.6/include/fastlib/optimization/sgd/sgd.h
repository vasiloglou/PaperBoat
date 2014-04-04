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

#ifndef PAPERBOAT_FASTLIB_OPTIMIZATION_SGD_SGD_H_
#define PAPERBOAT_FASTLIB_OPTIMIZATION_SGD_SGD_H_
#include <vector>
#include <string>
#include "fastlib/data/monolithic_point.h"

namespace fl {namespace ml {
  template <typename FunctionType>
  class StochasticGradientDescent {
    public:
      void set_objective(FunctionType *function);
      void set_initial_learning_rate(double eta0);
      void set_optimization_parameters(const std::vector<std::string> &args);
      void set_iterations(int32 iterations);
      void set_epochs(int32 epochs);
      template<typename WorkSpaceType, typename TableType>
      bool Optimize(WorkSpaceType *ws,
                    const std::vector<std::string> &tables,
                    fl::data::MonolithicPoint<double> *model);
    private:
      FunctionType *function_;
      double eta0_;
      int32 iterations_;
      int32 epochs_;
      int32 max_trials_;
  };
}}
#endif
