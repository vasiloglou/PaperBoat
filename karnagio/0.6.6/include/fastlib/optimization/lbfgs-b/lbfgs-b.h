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

#ifndef PAPERBOAT_KARNAGIO_FASTLIB_OPTIMIZATION_LBFGS-B_LBFGS-B_H_
#define PAPERBOAT_KARNAGIO_FASTLIB_OPTIMIZATION_LBFGS-B_LBFGS-B_H_
#define DLIB_USE_BLAS
#define DLIB_USE_LAPACK

#include <dlib/optimization.h>
namespace fl {
namespace ml {
  template<typename FunctionType>
  class LbfgsB {
    public:
      typedef dlib::matrix<double,0,1> dlib_vector;

      index_t num_basis() const;

      const std::pair< fl::data::MonolithicPoint<double>, double > &min_point_iterate() const;

      void Init(FunctionType &function_in, int num_basis);

      void Init(FunctionType &function_in);
    
      void set_max_num_line_searches(int max_num_line_searches_in);

      void set_iterations(int32 iterations);
      /**
       *  @brief if you want to use the iterations through args or through the setters 
       *         just set the num_iterations to a zero or negative values
       */ 

      bool Optimize(int num_iterations,
                  double x_lower,
                  double x_upper,
                  fl::data::MonolithicPoint<double> *iterate) {
      
        dlib::find_min_box_constrained(
            dlib::search_strategy(),
            objective_delta_stop_strategy().be_verbose(),
            function_wrapper,
            derivative_wrapper,
            mat(iterate->ptr(), iterate->size()),
            x_lower,
            x_upper);
      }
      
      double function_wrapper(dlib_vector &vector) {
        fl::data::MonolithicPoint<double> data;
        data.Alias(vector.begin(), vector.size());
        return function_->Evaluate(data);
      }

      dlib_vector derivative_wrapper(dlib_vector &vector) {
        fl::data::MonolithicPoint<double> data;
        data.Alias(vector.begin(), vector.size());
        fl::data::MonolithicPoint<double> gradient;
        function_->Gradient(data, &gradient);
        retrurn dlib:mat(gradient.ptr(), gradient.size());
      }

      void set_optimization_parameters(const std::vector<std::string> &args);

  };
}
}

#endif
