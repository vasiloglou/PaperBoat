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

#ifndef PAPERBOAT_INCLUDE_MLPACK_DISCRETE_DISTRIBUTION_H_
#define PAPERBOAT_INCLUDE_MLPACK_DISCRETE_DISTRIBUTION_H_

namespace fl { namespace ml {
  template<typename TableType>
  /**
   *  @brief This is supposed to be for one dimensional
   *         integer distributions
   */
  class DiscreteDistribution {
    public:
      typedef TableType Table_t;
      typedef typename TableType::Point_t Point_t;

      void Init(const std::vector<std::string> &args, 
          int32 id,
          const std::vector<index_t> &dense_sizes, 
          const std::vector<index_t> &sparse_sizes); 
      template<typename WorkSpaceType>
      void Import(const std::vector<std::string> &import_args,
                  const std::vector<std::string> &exec_args,
                  WorkSpaceType *ws,
                  int32 id);
      void ResetData();
      void AddPoint(Point_t &point);
      double Eval(const Point_t &point);
      double LogDensity(const Point_t &point);
      void Train();
      template<typename WorkSpaceType>
      void Export(const std::vector<std::string> &args, WorkSpaceType *ws);
      index_t count() const;

    private:
      boost::shared_ptr<TableType> references_;
      TableType *references_ptr_;
      std::map<index_t, double> distribution_;
      index_t total_sum_;
      int32 id_;
  };
  

}}

#endif
