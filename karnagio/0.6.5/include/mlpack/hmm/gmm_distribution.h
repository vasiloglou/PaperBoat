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

#ifndef PAPERBOAT_INCLUDE_MLPACK_GMM_DISTRIBUTION_H_
#define PAPERBOAT_INCLUDE_MLPACK_GMM_DISTRIBUTION_H_

#include "boost/mpl/if.hpp"
#include "fastlib/table/default_table.h"

namespace fl { namespace ml {

  /**
   *  @brief This is supposed to be for one dimensional
   *         integer distributions
   */
  template<typename TableType, bool IS_DIAGONAL>
  class GmmDistribution {
    public:
      typedef TableType Table_t;
      typedef typename TableType::Point_t Point_t;
      typedef typename boost::mpl::if_c<
        IS_DIAGONAL,
        typename fl::table::DefaultTable::Point_t,
        fl::table::DefaultTable
      >::type Covariance_t;
        

      void Init(
          const std::vector<std::string> &args,
          int32 id,
          const std::vector<index_t> &dense_sizes, 
          const std::vector<index_t> &sparse_sizes); 
      void ResetData();
      void AddPoint(Point_t &point);
      double Eval(const Point_t &point);
      double LogDensity(const Point_t &point);
      void Train();
      template<typename WorkSpaceType>
      void Export(const std::vector<std::string> &args, WorkSpaceType *ws);
      index_t count() const;

    private:
      std::string args_;
      boost::shared_ptr<TableType> references_;
      TableType *references_ptr_;
      int32 n_gaussians_;
      boost::shared_ptr<
        std::vector<
          Covariance_t> 
      > covariances_;
      boost::shared_ptr<
        std::vector<
          typename fl::table::DefaultTable::Point_t> 
      > means_;
      boost::shared_ptr<std::vector<double> > priors_;
      std::vector<fl::data::MonolithicPoint<double> > sigma_times_mu_;
      std::vector<double> mutrans_w_mu_;
      int32 id_;
  };
  

}}

#endif
