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
#ifndef FL_LITE_MLPACK_NNCLASSIFIER_NNCLASSIFIER_H
#define FL_LITE_MLPACK_NNCLASSIFIER_NNCLASSIFIER_H

#include <vector>
#include <map>
#include "fastlib/base/base.h"
#include "mlpack/allkn/allkn.h"
#include "mlpack/mnnclassifier/auc.h"
/**
 *  @brief This function converts nearest neighbor results to 
 *         classification scores. It accepts as inputs three vectors:
 *
 *  @param ind maps point indices to neighbor point indices for each
 *         query.
 *  @param dist maps point indices to neighbor point distances
 *         for each query.
 *  @param point_labels maps each reference point index to a class label
 *  @param score The classificatino accuracy for the reference set 
 *         in case the query is equal to the reference.
 */
namespace fl {
namespace ml {
  class MNNClassifier {
    public:
      template<typename TableType>
      struct Core {
        struct MySqrt {
          MySqrt() {
            sum=0;
          }
          template<typename T>
          void operator()(T *t) {
            sum+=*t;
            *t=sqrt((typename TableType::CalcPrecision_t)*t);  
          }
          double sum;
        };
        struct Norm {
          Norm(double s) {
            s_=s;
          }
          template<typename T>
          void operator()(T *t) {
            *t/=s_;  
          }
          double s_;
        };

        struct DefaultAllKNNMap : public AllKNArgs {
          typedef TableType QueryTableType;
          typedef TableType ReferenceTableType;
          typedef boost::mpl::int_<0>::type  KNmode;
        };
        typedef AllKN<DefaultAllKNNMap> DefaultAllKNN;
        template<typename DataAccessType>
        static int Main(DataAccessType *data,
                        boost::program_options::variables_map &vm);
      };

      /**
       * @brief This is the main driver function that the user has to
       *        call.
       */
      template<typename DataAccessType, typename BranchType> 
      static int Main(DataAccessType *data,
                      const std::vector<std::string> &args);
 
      template<typename IndTableType,
               typename DistTableType,
               typename ClassLabelTableType,
               typename ResultTableType>
      static void ComputeNNClassification(const fl::table::Table<IndTableType> &ind,
      			     const fl::table::Table<DistTableType> &dist,
      			     const fl::table::Table<ClassLabelTableType> &reference_labels,
      			     fl::table::Table<ResultTableType> *result,
      			     bool compute_score,
      			     double *score);

      template<typename IndTableType,
               typename DistTableType,
               typename ClassLabelTableType,
               typename ResultTableType>
      static void ComputeNNClassification(const fl::table::Table<IndTableType> &ind,
      			     const fl::table::Table<DistTableType> &dist,
      			     const fl::table::Table<ClassLabelTableType> &reference_labels,
                 const fl::table::Table<ClassLabelTableType> &query_labels,
      			     fl::table::Table<ResultTableType> *result,
      			     bool compute_score,
                 int auc_label,
      			     double *score,
                 double *auc,
                 std::vector<std::pair<double, double> > *roc,
                 std::map<int, int> *points_per_class,
                 std::map<int, double> *partial_score);

  };

}
} // namespaces

#endif
