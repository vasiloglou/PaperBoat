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

#ifndef FL_LITE_INCLUDE_MLPACK_MIXTURE_OF_EXPERTS_MOE_H
#define FL_LITE_INCLUDE_MLPACK_MIXTURE_OF_EXPERTS_MOE_H

#include "fastlib/base/base.h"
#include "fastlib/math/fl_math.h"
#include "boost/program_options.hpp"

/**
 *  @file moe.h
 */

namespace fl {
namespace ml {

template<typename ExpertType>
class Moe {
  public:
    /** The table that contains the data for computing Moe */
    typedef ExpertType  Expert_t;
    typedef typename Expert_t::Table_t Table_t;
    typedef typename Table_t::Point_t Point_t;

  public:
    Moe();
    void Compute(std::vector<index_t> *memberships,
                 std::vector<double> *cluster_scores);
    void set_predefined_memberships(std::map<index_t, int32> &predefined_memberships);
    void set_references(boost::shared_ptr<Table_t> references);
    void set_expert_args(const std::vector<std::string> &args);
    void set_expert_log(bool log);
    void set_k_clusters(int32 k_clusters);
    void set_iterations(int32 iterations);
    void set_n_restarts(int32 n_restarts);
    void set_error_tolerance(double error_tolerance);
    void set_initial_clusters(std::vector<boost::shared_ptr<Table_t> > 
       &initial_clusters);
    void set_initial_clusters();

  private:
    boost::shared_ptr<Table_t> reference_table_;
    std::map<index_t, int32> predefined_memberships_;
    std::vector<boost::shared_ptr<Expert_t> > experts_;
    int32 k_clusters_;
    int32 iterations_;
    int32 n_restarts_;
    double error_tolerance_;
    std::vector<std::string> expert_args_;
    bool log_;

};

template<>
class Moe<boost::mpl::void_> {
  public:
    template<typename TableType1>
    class Core {
      public:
        template<typename WorkSpaceType>
        static int Main(WorkSpaceType *data,
                        boost::program_options::variables_map &vm);
    };

    /**
     * @brief This is the main driver function that the user has to
     *        call.
     */
    template<typename DataAccessType, typename Branch>
    static int Main(DataAccessType *data,
                    const std::vector<std::string> &args);

    template<typename DataAccessType>
    static void Run(DataAccessType *data,
        const std::vector<std::string> &args);

};



}
} // namespaces

#endif
