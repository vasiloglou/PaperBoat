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
#ifndef FL_LITE_FASTLIB_DATA_LINEAR_ALGEBRA_H_
#define FL_LITE_FASTLIB_DATA_LINEAR_ALGEBRA_H_
#include "boost/mpl/front.hpp"
#include "boost/mpl/for_each.hpp"
#include "boost/mpl/equal_to.hpp"
#include "boost/mpl/size.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/and.hpp"
#include "boost/mpl/empty.hpp"
#include "mixed_point.h"
#include "monolithic_point.h"
#include "sparse_point.h"
#include "fastlib/la/linear_algebra_defs.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/dense/linear_algebra.h"

namespace fl {
namespace data {
class mixed_ops {
  public:
    template<typename MatrixType>
    struct DenseLengthEuclideanOperators {
     public:
      DenseLengthEuclideanOperators(const MatrixType &matrix,
                                    typename MatrixType::CalcPrecision_t *result) : matrix_(matrix) {
        result_ = result;
      }
      template<typename T>
      void operator()(T) {
        *result_ += fl::la::Dot(matrix_.template dense_point<T>(),
                                matrix_.template dense_point<T>());
      }
     private:
      const MatrixType &matrix_;
      typename MatrixType::CalcPrecision_t *result_;
    };

    template<typename MatrixType>
    struct SparseLengthEuclideanOperators {
     public:
      SparseLengthEuclideanOperators(const MatrixType &matrix,
                                     double *result) : matrix_(matrix) {
        result_ = result;
      }
      template<typename T>
      void operator()(T) {
        *result_ += fl::la::Dot(matrix_.template sparse_point<T>(),
                                matrix_.template sparse_point<T>());
      }
     private:
      const MatrixType &matrix_;
      double *result_;
    };

    template<typename MatrixType>
    static inline typename MatrixType::CalcPrecision_t LengthEuclidean(
      const MatrixType &x) {
      typename MatrixType::CalcPrecision_t result = 0;
      boost::mpl::for_each<typename MatrixType::DenseTypes_t>(
        DenseLengthEuclideanOperators<MatrixType>(x, &result));
      boost::mpl::for_each<typename MatrixType::SparseTypes_t>(
        SparseLengthEuclideanOperators<MatrixType>(x, &result));
      return fl::math::Pow<typename MatrixType::CalcPrecision_t, 1, 2>(result);
    }

    template<typename MatrixType>
    struct DenseDotOperators {
     public:
      DenseDotOperators(const MatrixType &matrix1, const MatrixType &matrix2,
                        double *result) :
          matrix1_(matrix1), matrix2_(matrix2) {
        result_ = result;
      }
      template<typename T>
      void operator()(T) {
        *result_ += fl::la::Dot(matrix1_.template dense_point<T>(),
                                matrix2_.template dense_point<T>());
      }
     private:
      const MatrixType &matrix1_;
      const MatrixType &matrix2_;
      double *result_;
    };

    template<typename MatrixType>
    struct SparseDotOperators {
     public:
      SparseDotOperators(const MatrixType &matrix1,
                         const MatrixType &matrix2,
                         double *result) :
          matrix1_(matrix1), matrix2_(matrix2) {
        result_ = result;
      }
      template<typename T>
      void operator()(T) {
        *result_ += fl::la::Dot(matrix1_.template sparse_point<T>(),
                                matrix2_.template sparse_point<T>());
      }
     private:
      const MatrixType &matrix1_;
      const MatrixType &matrix2_;
      double *result_;
    };

    template<typename MatrixType1, typename MatrixType2>
    static inline double Dot(const MatrixType1 &x,
        const MatrixType2 &y) {
      return boost::mpl::if_<
          boost::mpl::and_<
            boost::mpl::equal_to<
              boost::mpl::size<
                typename MatrixType1::DenseTypes_t
              >,
              boost::mpl::int_<1>
            >,
            boost::mpl::empty<
              typename MatrixType1::SparseTypes_t
            >
          >,
          Dot2,
          Dot1
        >::type::Dot(x,y);
    }

    struct Dot1 {
      template<typename MatrixType>
      static inline double Dot(const MatrixType &x,
          const MatrixType &y) {
        double result = 0;
        boost::mpl::for_each<typename MatrixType::DenseTypes_t>(
          DenseDotOperators<MatrixType>(x, y, &result));
        boost::mpl::for_each<typename MatrixType::SparseTypes_t>(
          SparseDotOperators<MatrixType>(x, y, &result));
        return result;
      }
    };
    
    struct Dot2 {
      template<typename MatrixType1, typename MatrixType2>
      static inline double Dot(const MatrixType1 &x,
          const MatrixType2 &y) {
        typedef typename 
          boost::mpl::at_c<typename MatrixType1::DenseTypes_t, 0>::type Precision_t;
        return (double)(
            MonolithicPoint<Precision_t>::Dot(
            x. template dense_point<Precision_t>(), y));
      }
    };


    template<typename MatrixType>
    struct DenseSelfScaleOperator {
      public:
        DenseSelfScaleOperator(MatrixType *matrix,
                               double alpha) {
          matrix_  = matrix;
          alpha_ = alpha;
        }
        template<typename T>
        void operator()(T) {
          fl::la::SelfScale(alpha_, &(matrix_->template dense_point<T>()));
        }
      private:
        MatrixType *matrix_;
        double alpha_;
    };

    template<typename MatrixType>
    struct SparseSelfScaleOperator {
      public:
        SparseSelfScaleOperator(MatrixType *matrix,
                                double alpha) {
          matrix_  = matrix;
          alpha_ = alpha;
        }
        template<typename T>
        void operator()(T) {
          fl::la::SelfScale(alpha_, &(matrix_->template sparse_point<T>()));
        }
      private:
        MatrixType *matrix_;
        double alpha_;
    };

    template<typename MatrixType>
    static inline void SelfScale(double alpha,
                                 MatrixType *x) {
      boost::mpl::for_each<typename MatrixType::DenseTypes_t>(
        DenseSelfScaleOperator<MatrixType>(x, alpha));
      boost::mpl::for_each<typename MatrixType::SparseTypes_t>(
        SparseSelfScaleOperator<MatrixType>(x, alpha));

    }

    template<typename MatrixType1, 
             typename MatrixType2,
             fl::la::MemoryAlloc M>
    struct DenseScaleOperator {
      public:
        DenseScaleOperator(double alpha,
                           const MatrixType1 &matrix1,
                           MatrixType2 *matrix2) : alpha_(alpha),
        matrix1_(matrix1), matrix2_(matrix2)
        {} 
        template<typename T>
        void operator()(T) {
          fl::la::Scale<M>(alpha_, 
                        matrix1_.template dense_point<T>(),
                        &(matrix2_->template dense_point<T>()));
        }
      private:
        double alpha_;
        const MatrixType1 &matrix1_;
        MatrixType2 *matrix2_;
    };

    template<typename MatrixType1, 
      typename MatrixType2, 
      fl::la::MemoryAlloc M>
    struct SparseScaleOperator {
      public:
        SparseScaleOperator(double alpha,
                            const MatrixType1 &matrix1,
                            MatrixType2 *matrix2) :
        alpha_(alpha), matrix1_(matrix1), matrix2_(matrix2)
        {}
        template<typename T>
        void operator()(T) {
          fl::la::Scale<M>(alpha_, 
                        matrix1_.template sparse_point<T>(),
                        &(matrix2_->template sparse_point<T>()));
        }
      private:
        const double alpha_;
        const MatrixType1 &matrix1_;
        MatrixType2 *matrix2_;
    };


    template<fl::la::MemoryAlloc M>
    class Scale {
      public:
        template<typename MatrixType1, typename MatrixType2>
        Scale(double alpha,
              const MatrixType1 &x,
              MatrixType2 *y) {
         boost::mpl::for_each<typename MatrixType1::DenseTypes_t>(
           DenseScaleOperator<MatrixType1, MatrixType2, M>(alpha, x, y));
         boost::mpl::for_each<typename MatrixType1::SparseTypes_t>(
           SparseScaleOperator<MatrixType1, MatrixType2, M>(alpha, x, y)); 
        }
    };

    template<typename MatrixType>
    struct DenseAddExpertOperator {
      public:
        DenseAddExpertOperator(double alpha,
                               const MatrixType &x,
                               MatrixType *y) :alpha_(alpha), x_(x), y_(y) {
        }
        template<typename T>
        void operator()(T) {
          fl::la::AddExpert(alpha_,
              x_.template dense_point<T>(), 
              &(y_->template dense_point<T>()));
        }
      private:
        const double alpha_;
        const MatrixType &x_;
        MatrixType *y_;
    };

    template<typename MatrixType>
    struct SparseAddExpertOperator {
      public:
        SparseAddExpertOperator(double alpha,
                                const MatrixType &x,
                                MatrixType *y) : alpha_(alpha), x_(x), y_(y) {
        }
        template<typename T>
        void operator()(T) {
          fl::la::AddExpert(alpha_,
                            x_.template sparse_point<T>(),
                            &(y_->template sparse_point<T>()));
        }
      private:
        const double alpha_;
        const MatrixType &x_;
        MatrixType *y_;
    };

    template<typename MatrixType>
    static inline void AddExpert(const typename MatrixType::CalcPrecision_t alpha,
                                 const MatrixType &x,
                                 MatrixType * const y) {
      boost::mpl::for_each<typename MatrixType::DenseTypes_t>(
          DenseAddExpertOperator<MatrixType>(alpha, x, y));
      boost::mpl::for_each<typename MatrixType::SparseTypes_t>(
        SparseAddExpertOperator<MatrixType>(alpha, x, y));
    }

    template<typename MatrixType>
    struct DenseAddToOperator {
      public:
        DenseAddToOperator(const MatrixType &x,
                           MatrixType *y): x_(x), y_(y) {
        }
        template<typename T>
        void operator()(T) {
          fl::la::AddTo(x_.template dense_point<T>(), &(y_->template dense_point<T>()));
        }
      private:
        const MatrixType &x_;
        MatrixType *y_;
    };

    template<typename MatrixType>
    struct SparseAddToOperator {
      public:
        SparseAddToOperator(const MatrixType &x,
                            MatrixType *y): x_(x), y_(y) {
        }
        template<typename T>
        void operator()(T) {
          fl::la::AddTo(x_.template sparse_point<T>(),
                        &(y_->template sparse_point<T>()));
        }
      private:
        const MatrixType &x_;
        MatrixType *y_;
    };

    template<typename MatrixType>
    static inline void AddTo(const MatrixType &x,
                             MatrixType * const y) {
      boost::mpl::for_each<typename MatrixType::DenseTypes_t>(
        DenseAddToOperator<MatrixType>(x, y));
      boost::mpl::for_each<typename MatrixType::SparseTypes_t>(
        SparseAddToOperator<MatrixType>(x, y));

    }

    template<fl::la::MemoryAlloc M>
    class Add {
      public:
        template<typename MatrixType>
        Add(const MatrixType &x,
            const MatrixType &y,
            MatrixType * const z);
    };

    template<typename MatrixType>
    struct DenseSubFromOperator {
public:
      DenseSubFromOperator(const MatrixType &x,
                           MatrixType *y): x_(x), y_(y) {
      }
      template<typename T>
      void operator()(T) {
        fl::la::SubFrom(x_.template dense_point<T>(), &(y_->template dense_point<T>()));
      }
private:
      const MatrixType &x_;
      MatrixType *y_;
    };

    template<typename MatrixType>
    struct SparseSubFromOperator {
public:
      SparseSubFromOperator(const MatrixType &x,
                            MatrixType *y): x_(x), y_(y) {
      }
      template<typename T>
      void operator()(T) {
        fl::la::SubFrom(x_.template sparse_point<T>(),
                        &(y_->template sparse_point<T>()));
      }
private:
      const MatrixType &x_;
      MatrixType *y_;
    };

    template<typename MatrixType1, typename MatrixType2>
    static inline void SubFrom(const MatrixType1 &x,
                               MatrixType2 * const y) {
      boost::mpl::for_each<typename MatrixType2::DenseTypes_t>(
        DenseSubFromOperator<MatrixType2>(x, y));
      boost::mpl::for_each<typename MatrixType2::SparseTypes_t>(
        SparseSubFromOperator<MatrixType2>(x, y));
    }

    template<fl::la::MemoryAlloc M>
    class Sub {
      public:
        template<typename MatrixType1, typename MatrixType2>
        Sub(const MatrixType1 &x,
            const MatrixType2 &y,
            MatrixType1 * const z);
    };


    template<typename MatrixType1, typename MatrixType2>
    static inline typename MatrixType1::CalcPrecision_t DistanceSqEuclidean(const MatrixType1& x,
        const MatrixType2& y) {
      typename MatrixType1::CalcPrecision_t result;
      RawLMetric<2>(x, y, &result);
      return result;
    }

    template<typename MatrixType, int t_pow>
    struct DenseRawLMetricOperators {
      public:
        DenseRawLMetricOperators(const MatrixType &matrix1, const MatrixType &matrix2,
                               typename MatrixType::CalcPrecision_t *result) :
          matrix1_(matrix1), matrix2_(matrix2) {
          result_ = result;
        }
        template<typename T>
        void operator()(T) {
          typename MatrixType::CalcPrecision_t temp;
          fl::la::RawLMetric<t_pow>(matrix1_.template dense_point<T>(),
                                    matrix2_.template dense_point<T>(), &temp);
          (*result_) += temp;
        }
      private:
        const MatrixType &matrix1_;
        const MatrixType &matrix2_;
        typename MatrixType::CalcPrecision_t *result_;
    };

    template<typename MatrixType, int t_pow>
    struct SparseRawLMetricOperators {
      public:
        SparseRawLMetricOperators(const MatrixType &matrix1,
                                  const MatrixType &matrix2,
                                  typename MatrixType::CalcPrecision_t *result) :
            matrix1_(matrix1), matrix2_(matrix2) {
          result_ = result;
        }
        template<typename T>
        void operator()(T) {
          typename MatrixType::CalcPrecision_t temp;
          fl::la::RawLMetric<t_pow>(matrix1_.template sparse_point<T>(),
                                    matrix2_.template sparse_point<T>(), &temp);
          (*result_) += temp;
        }
      private:
        const MatrixType &matrix1_;
        const MatrixType &matrix2_;
        typename MatrixType::CalcPrecision_t *result_;
    };

    template<typename MatrixType, typename PrecisionType, int t_pow>
    struct DenseWeightsRawLMetricOperators {
public:
      DenseWeightsRawLMetricOperators(
        const fl::data::MonolithicPoint<PrecisionType> &weights,
        const MatrixType &matrix1, const MatrixType &matrix2,
        index_t * const offset,
        typename MatrixType::CalcPrecision_t *result) : weights_(weights),
          matrix1_(matrix1), matrix2_(matrix2), offset_(offset) {
        result_ = result;
      }
      template<typename T>
      void operator()(T) {
        typename MatrixType::CalcPrecision_t temp;
        fl::data::MonolithicPoint<PrecisionType> w;
        index_t size = matrix1_.template dense_point<T>().size();
        DEBUG_ASSERT(*offset_ + size <= weights_.size());
        w.Alias(const_cast<fl::data::MonolithicPoint<PrecisionType> &>(weights_).ptr()
                + *offset_, size);
        fl::la::RawLMetric<t_pow>(weights_, matrix1_.template dense_point<T>(),
                                  matrix2_.template dense_point<T>(), &temp);
        (*result_) += temp;
        *offset_ += size;
      }
private:
      const fl::data::MonolithicPoint<PrecisionType> &weights_;
      const MatrixType &matrix1_;
      const MatrixType &matrix2_;
      index_t * const offset_;
      typename MatrixType::CalcPrecision_t *result_;
    };

    template<typename MatrixType, typename PrecisionType, int t_pow>
    struct SparseWeightsRawLMetricOperators {
public:
      SparseWeightsRawLMetricOperators(
        const fl::data::MonolithicPoint<PrecisionType> &weights,
        const MatrixType &matrix1,
        const MatrixType &matrix2,
        index_t * const offset,
        typename MatrixType::CalcPrecision_t *result) :
          weights_(weights),
          matrix1_(matrix1), matrix2_(matrix2), offset_(offset) {
        result_ = result;
      }
      template<typename T>
      void operator()(T) {
        typename MatrixType::CalcPrecision_t temp;
        fl::data::MonolithicPoint<PrecisionType> w;
        index_t size = matrix1_.template sparse_point<T>().size();
        DEBUG_ASSERT(*offset_ + size <= weights_.size());
        w.Alias(const_cast<fl::data::MonolithicPoint<PrecisionType> &>(weights_).ptr()
                + *offset_, size);
        fl::la::RawLMetric<t_pow>(weights_, matrix1_.template sparse_point<T>(),
                                  matrix2_.template sparse_point<T>(), &temp);
        (*result_) += temp;
        *offset_ += size;

      }
private:
      const fl::data::MonolithicPoint<PrecisionType> &weights_;
      const MatrixType &matrix1_;
      const MatrixType &matrix2_;
      index_t * const offset_;
      typename MatrixType::CalcPrecision_t *result_;
    };

    template<int t_pow>
    class RawLMetric {
      public:
        template<typename PointType>
        RawLMetric(const PointType &x,
                   const PointType &y, typename PointType::CalcPrecision_t *result) {
          typename PointType::CalcPrecision_t result1 = 0;
          typename PointType::CalcPrecision_t result2 = 0;
          boost::mpl::for_each<typename PointType::DenseTypes_t>(
            DenseRawLMetricOperators<PointType, t_pow>(x, y, &result1));
          boost::mpl::for_each<typename PointType::SparseTypes_t>(
            SparseRawLMetricOperators<PointType, t_pow>(x, y, &result2));
          *result = result1 + result2;
        }

        template<typename PointType, typename T>
        RawLMetric(const MonolithicPoint<T> &weights,
                   const PointType &x,
                   const PointType &y, typename PointType::CalcPrecision_t *result) {
          typename PointType::CalcPrecision_t result1 = 0;
          typename PointType::CalcPrecision_t result2 = 0;
          index_t offset = 0;
          boost::mpl::for_each<typename PointType::DenseTypes_t>(
            DenseWeightsRawLMetricOperators<PointType, T, t_pow>(weights, x, y, &offset, &result1));
          boost::mpl::for_each<typename PointType::SparseTypes_t>(
            SparseWeightsRawLMetricOperators<PointType, T, t_pow>(weights, x, y, &offset, &result2));
          *result = result1 + result2;
        }


        template < typename PrecisionType1,
        typename MixedPointType,
        typename PrecisionType2 >
        RawLMetric(
          const MixedPointType &x,
          const fl::data::MonolithicPoint<PrecisionType1> &y,
          PrecisionType2 *result) {
          typename fl::data::MonolithicPoint<PrecisionType1>::template RawLMetric<t_pow>(
            y, x, result);
        }
    };

    template<int t_pow>
    class LMetric {
      public:
        template<typename MatrixType1, typename MatrixType2>
        LMetric(const MatrixType1 &x, const MatrixType2 &y,
                typename MatrixType1::Precision_t * const result);
    };

    template<typename MatrixType, fl::la::MemoryAlloc MemAlloc>
    struct DenseDotMulOperators {
      public:
        DenseDotMulOperators(const MatrixType &matrix1,
                              const MatrixType &matrix2,
                              MatrixType *result) :
            matrix1_(matrix1), matrix2_(matrix2) {
          result_ = result;
        }
        template<typename T>
        void operator()(T) {
          typename MatrixType::CalcPrecision_t temp;
          fl::la::DotMul<MemAlloc>(matrix1_.template dense_point<T>(),
                                   matrix2_.template dense_point<T>(), result_);
        }
      private:
        const MatrixType &matrix1_;
        const MatrixType &matrix2_;
        MatrixType *result_;
    };

    template<typename MatrixType, fl::la::MemoryAlloc MemAlloc>
    struct SparseDotMulOperators {
      public:
        SparseDotMulOperators(const MatrixType &matrix1,
                              const MatrixType &matrix2,
                              MatrixType *result) :
            matrix1_(matrix1), matrix2_(matrix2) {
          result_ = result;
        }
        template<typename T>
        void operator()(T) {
          typename MatrixType::CalcPrecision_t temp;
          fl::la::DotMul<MemAlloc>(matrix1_.template sparse_point<T>(),
                                   matrix2_.template sparse_point<T>(), result_);
        }
      private:
        const MatrixType &matrix1_;
        const MatrixType &matrix2_;
        MatrixType *result_;
    };


    template<fl::la::MemoryAlloc MemAlloc>
    class DotMul {
      public:
        template<typename PointType>
        DotMul(const PointType &a,
               const PointType &b,
               PointType* const c) {
          boost::mpl::for_each<typename PointType::DenseTypes_t>(
            DenseDotMulOperators<PointType, MemAlloc>(a, b, c));
          boost::mpl::for_each<typename PointType::SparseTypes_t>(
            SparseDotMulOperators<PointType, MemAlloc>(a, b, c));
        }
    };

    template<typename MatrixType1, typename MatrixType2>
    static void DotMulTo(MatrixType1 * const a,
                         const MatrixType2 &b);

    class Sum {
      public:
        template<typename PointType>
        struct DenseSumOperators {
          public:
            DenseSumOperators(const PointType &point,
                              double *result) :
               point_(point), result_(result) {
         
            }
            template<typename T>
            void operator()(T) {
              fl::la::Sum(point_.template dense_point<T>(), result_);
            }
          private:
            const PointType &point_;
            double *result_;
        };

        template<typename PointType>
        struct SparseSumOperators {
          public:
            SparseSumOperators(const PointType &point,
                              double *result) :
               point_(point), result_(result) {
         
            }
            template<typename T>
            void operator()(T) {
              fl::la::Sum(point_.template sparse_point<T>(), result_);
            }
          private:
            const PointType &point_;
            double *result_;
        };
        template<typename PointType>
        Sum(const PointType &a, 
            typename PointType::CalcPrecision_t* const sum) {
          *sum=0;
          boost::mpl::for_each<typename PointType::DenseTypes_t>(
            DenseSumOperators<PointType>(a, sum));
          boost::mpl::for_each<typename PointType::SparseTypes_t>(
            SparseSumOperators<PointType>(a, sum));
       
        }
    };

};
}
} // namespaces
#endif
