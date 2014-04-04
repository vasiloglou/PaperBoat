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
#ifndef FL_LITE_MLPACK_NMF_LBFGS_NMF_H_
#define FL_LITE_MLPACK_NMF_LBFGS_NMF_H_
#include "boost/mpl/void.hpp"
#include "boost/utility.hpp"
#include "fastlib/base/base.h"

namespace fl {
namespace ml {
/**
 * @brief This class is used in LBFGS NMF. It has all the necessary 
 *        information NMF needs to optimize W.
 */
template<typename NmfArgsType>
class NmfWFactorFunction {

  public:
    typedef typename NmfArgsType::InputTable_t InputTable_t;
    typedef typename NmfArgsType::FactorsTable_t FactorsTable_t;
    typedef typename FactorsTable_t::CalcPrecision_t CalcPrecision_t;

  private:
    int num_dimensions_;
    InputTable_t *table_;
    int k_rank_;
    const fl::data::MonolithicPoint<CalcPrecision_t> *current_h_factor_;


  public:

    NmfWFactorFunction();

    void Init(
      InputTable_t *table_in, int k_rank_in,
      const fl::data::MonolithicPoint<CalcPrecision_t> *current_h_factor);

    CalcPrecision_t Evaluate(
      const fl::data::MonolithicPoint<CalcPrecision_t> &x);

    void Project(fl::data::MonolithicPoint<double> *data) {}
    void Gradient(const fl::data::MonolithicPoint<CalcPrecision_t> &x,
                  fl::data::MonolithicPoint<CalcPrecision_t> *gradient);

    int num_dimensions() const;

};

/**
 * @brief This class is used in LBFGS NMF. It has all the necessary 
 *        information NMF needs to optimize H.
 */
template<typename NmfArgsType>
class NmfHFactorFunction {

  public:
    typedef typename NmfArgsType::InputTable_t InputTable_t;
    typedef typename NmfArgsType::FactorsTable_t FactorsTable_t;
    typedef typename FactorsTable_t::CalcPrecision_t CalcPrecision_t;

  private:
    int num_dimensions_;
    InputTable_t *table_;
    int k_rank_;
    const fl::data::MonolithicPoint<CalcPrecision_t> *current_w_factor_;

  public:

    NmfHFactorFunction();

    void Init(
      InputTable_t *table_in, int k_rank_in,
      const fl::data::MonolithicPoint<CalcPrecision_t> *current_w_factor);

    CalcPrecision_t Evaluate(
      const fl::data::MonolithicPoint<CalcPrecision_t> &x);

    void Project(fl::data::MonolithicPoint<double> *data) {}

    void Gradient(const fl::data::MonolithicPoint<CalcPrecision_t> &x,
                  fl::data::MonolithicPoint<CalcPrecision_t> *gradient);

    int num_dimensions() const;

};

/**
 * @brief Default arguments for Sparse NMF
 */
class NmfDefaultArgs {
    typedef boost::mpl::void_ InputTable_t;
    typedef boost::mpl::void_ FactorsTable_t;
};

/**
 *  @brief SparseNmf is a class that does NMF on sparse data. By sparse
 *         we mean a data matrix that some of the entries are undefined.
 *         The class uses LBFGS or Stochastic gradient descent, or both.
 */
template<typename NmfArgsType>
class SparseNmf : boost::noncopyable {
  public:
    typedef typename NmfArgsType::InputTable_t InputTable_t;
    typedef typename NmfArgsType::FactorsTable_t FactorsTable_t;
    typedef typename FactorsTable_t::CalcPrecision_t CalcPrecision_t;

    SparseNmf();

    void Init(InputTable_t *table,
              FactorsTable_t *w_factor,
              FactorsTable_t *h_factor);
    void Train(const std::string &mode);
    CalcPrecision_t Evaluate(index_t i, index_t j);
    template<typename ContainerType>
    void Evaluate(std::vector<std::pair<index_t, index_t> > &indices,
                  ContainerType *values);
    void set_rank(index_t k_rank);
    void set_iterations(index_t iterations);
    void set_w_sparsity_factor(CalcPrecision_t sparsity_factor);
    void set_h_sparsity_factor(CalcPrecision_t sparsity_factor);
    void set_lbfgs_rank(index_t lbfgs_rank);
    void set_lbfgs_steps(index_t lbfgs_steps);
    void set_epochs(index_t epochs);
    void set_step0(CalcPrecision_t step0);
    FactorsTable_t *get_w();
    FactorsTable_t *get_h();

  private:

    CalcPrecision_t L1Norm_(
      const fl::data::MonolithicPoint<CalcPrecision_t> &v) const;
    CalcPrecision_t L1Norm_(const CalcPrecision_t *v, int length) const;
    void SparseProjection_(CalcPrecision_t *point, index_t length,
                           CalcPrecision_t sparsity_factor);
    void BatchSparseProjection_(
      fl::data::MonolithicPoint<CalcPrecision_t> &factor,
      int step_length,
      CalcPrecision_t sparsity_factor);

  private:
    InputTable_t *table_;
    int k_rank_;
    CalcPrecision_t w_sparsity_factor_;
    CalcPrecision_t h_sparsity_factor_;
    FactorsTable_t *w_factor_;
    FactorsTable_t *h_factor_;
    int iterations_;
    index_t lbfgs_rank_;
    index_t lbfgs_steps_;
    CalcPrecision_t step0_;
    index_t epochs_;
};

template<>
class SparseNmf<boost::mpl::void_> {
  public:
    template<typename TableType>
    class Core {
      public:
        template<typename FactorsTableType>
        struct Args : public NmfDefaultArgs {
          typedef TableType InputTable_t;
          typedef FactorsTableType FactorsTable_t;
        };
        template<typename DataAccessType>
        static int Main(DataAccessType *data,
                        boost::program_options::variables_map &vm);
    };
    template<typename DataAccessType, typename BranchType>
    static int Main(DataAccessType *data,
                    const std::vector<std::string> &args);


};

}
} //namespaces


#endif
