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

#ifndef FL_LITE_INCLUDE_MLPACK_MIXTURE_OF_EXPERTS_REGRESSION_EXPERT_H_
#define FL_LITE_INCLUDE_MLPACK_MIXTURE_OF_EXPERTS_REGRESSION_EXPERT_H_
#include "mlpack/regression/linear_regression.h"
#include "fastlib/workspace/workspace.h"
namespace fl {namespace ml {
  template<typename TableType, typename WorkSpaceType>
  class RegressionExpert {
    public:
      typedef TableType Table_t;
      typedef typename Table_t::Point_t Point_t;
      
      void Build();
      double Evaluate(const Point_t &);    
      double score();  
      /**
       *  @brief watch out, this set will change the reference table
       *  it will add an extra point
       */
      void set(boost::shared_ptr<Table_t> &table);
      void set_args(const std::vector<std::string> &arguments);
      void set_coeff_index(index_t coeff_index);
      index_t cardinality() const ;
      void set_log(bool log);
  
    private:
      bool log_;
      std::vector<std::string> arguments_;
      boost::shared_ptr<Table_t> references_;
      double score_;
      index_t coeff_index_;
      
      boost::shared_ptr<
        typename fl::ws::WorkSpace::DefaultTable_t
      > coefficients_table_;

  };

}}
#endif
