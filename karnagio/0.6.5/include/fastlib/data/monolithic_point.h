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
#ifndef FASTLIB_DATA_MONOLITHIC_POINT_H_
#define FASTLIB_DATA_MONOLITHIC_POINT_H_
#include <iostream>
#include <string>
#include <vector>
#include "boost/mpl/for_each.hpp"
#include "boost/mpl/vector.hpp"
#include "fastlib/base/base.h"
#include "fastlib/dense/matrix.h"
#include "sparse_point.h"
#include "mixed_point.h"
#include "boost/mpl/assert.hpp"
#include "boost/type_traits.hpp"
#include "boost/serialization/base_object.hpp"

namespace fl {
namespace data {
/* Forward Declarations */
template<typename CalcPrecisionType>
class MonolithicPoint :
      public fl::dense::Matrix<CalcPrecisionType, true> {
  public:
    typedef CalcPrecisionType CalcPrecision_t;
    typedef boost::mpl::vector1<CalcPrecisionType> DenseMemberTypes_t;
    typedef boost::mpl::vector0<> SparseMemberTypes_t;
    friend class boost::serialization::access;

    class iterator {
      public:

        iterator() {
          ptr_ = NULL;
          ind_ = 0;
        }

        iterator(const iterator &other) {
          ptr_ = other.ptr_;
          ind_ = other.ind_;
        }
        
        iterator(CalcPrecision_t * const ptr, const index_t ind) {
          ptr_=ptr;
          ind_=ind;
        }

        const iterator& operator=(const iterator &other) {
          ptr_ = other.ptr_;
          ind_ = other.ind_;
          return *this;
        }
     
        CalcPrecision_t &operator*() {
          return *ptr_;
        }

        const CalcPrecision_t &operator*() const {
          return *ptr_;
        }

        CalcPrecision_t *operator->() {
          return ptr_;
        }

        const CalcPrecision_t *operator->() const {
          return ptr_;
        }

        const iterator &operator++() {
          ++ptr_;
          ++ind_;
          return *this;
        }

        bool operator==(const iterator& other) const {
          return ptr_ == other.ptr_;
        }

        bool operator!=(const iterator& other) const {
          return ptr_ != other.ptr_;
        }

        CalcPrecision_t &value() {
          return *ptr_;
        }

        const CalcPrecision_t &value() const {
          return *ptr_;
        }

        index_t &attribute() {
          return ind_;
        }

        const index_t &attribute() const {
          return ind_;
        }

      private:
        CalcPrecision_t *ptr_;
        index_t ind_;
    };
    
    virtual ~MonolithicPoint() {
    };

    void Init(index_t size) {
      fl::dense::Matrix<CalcPrecisionType, true>::Init(size);
    }

    iterator begin() const {
      iterator it(this->ptr_, 0);
      return it;
    }

    iterator end() const {
      iterator it(this->ptr_+this->size(), this->size());
      return it;
    }

    template<class ContainerType>
    void Init(ContainerType &sizes) {
      BOOST_ASSERT(sizes.size() == 1);
      this->Init(sizes[0]);
    }

    std::vector<index_t> dense_sizes() const {
      return std::vector<index_t>(1, this->size());
    }

    std::vector<index_t> sparse_sizes() const {
      return std::vector<index_t>();
    }
    
    bool IsZero() const {
      for(index_t i=0; i<this->size(); ++i) {
        if (this->get(i)!=0) {
          return false;
        }
      }
      return true;
    }
    /**
     * @brief number of non zero points for monolithic point
     *        is the size of the point
     */
    index_t nnz() const {
      return this->size();
    }
    /**
     * @brief for batch operations on a point
     */
    template<typename FunctionType>
    void Transform(FunctionType &f) {
      for(index_t i=0; i<this->size(); ++i) {
        f(i, &(this->ptr_[i]));
      }
    }


    /**
     * @brief The Matrix Copy repeated here because of a C++ aliasing
     */
    inline void Copy(const MonolithicPoint<CalcPrecisionType> &other) {
      fl::dense::Matrix<CalcPrecisionType, true>::Copy(other);
    }
    /**
     * @brief The Matrix CopyValues repeated here because of a C++ aliasing
     */
    template<typename PrecisionType>
    inline void Copy(const PrecisionType *other, index_t length) {
      fl::dense::Matrix<CalcPrecisionType, true>::Copy(other, length);
    }

    /**
     * @brief The Matrix CopyValues repeated here because of a C++ aliasing
     */
    inline void CopyValues(const MonolithicPoint<CalcPrecisionType> &other) {
      fl::dense::Matrix<CalcPrecisionType, true>::CopyValues(other);
    }

    /**
     * @brief The assignment operator
     */
    inline MonolithicPoint<CalcPrecisionType> & operator=(
      const MonolithicPoint<CalcPrecisionType> &other) {
      fl::dense::Matrix<CalcPrecisionType, true>::CopyValues(other);
      return *this;
    }

    /**
     * @brief Copy a sparse point
     *
     */
    template<typename PrecisionType>
    inline void Copy(const SparsePoint<PrecisionType> &other) {
      Init(other.size());
      CopyValues(other);
    }
    /**
     * @brief CopyValues, copies the values from a sparse point
     *
     */
    template<typename PrecisionType>
    inline void CopyValues(const SparsePoint<PrecisionType> &other) {
      DEBUG_ASSERT(this->size()==other.size());
      typename SparsePoint<PrecisionType>::iterator it, end;
      end = other.end();
      for (it = other.begin(); it != end; ++it) {
        this->operator[](it->first) = it->second;
      }
    }
    /**
     * @brief Copy a mixed point
     *
     */
    template<typename ParamList>
    inline void Copy(const MixedPoint<ParamList> &other) {
      Init(index_t(other.size()));
      CopyValues(other);
    }
    /**
     * @brief CopyValues, copies the values from a mixed point
     *
     */
    template<typename ParamList>
    inline void CopyValues(const MixedPoint<ParamList> &other) {
      typedef fl::data::MonolithicPoint<CalcPrecisionType> MonolithicPoint_t;
      typedef fl::data::MixedPoint<ParamList> MixedPoint_t;
      typedef typename MixedPoint_t::DenseTypes_t  DenseTypes_t;
      typedef typename MixedPoint_t::SparseTypes_t SparseTypes_t;
      index_t current_index = 0;
      boost::mpl::for_each<DenseTypes_t>(
        CopyValuesOperatorsDense<MonolithicPoint_t, MixedPoint_t>(
          this, other, &current_index));
      boost::mpl::for_each<SparseTypes_t>(
        CopyValuesOperatorsSparse<MonolithicPoint_t, MixedPoint_t>(
          this, other, &current_index));

    }
    
    void SetRandom(const CalcPrecision_t low, const CalcPrecision_t hi) {
      for(index_t i=0; i<this->size(); ++i) {
        set(i, fl::math::Random(low, hi));
      }
    }
    
    /**
     * Loading a sparse point
     * This method assumes that the indices
     * between it1 and it2 are in ascending order
     */
    template<typename IteratorType>
    void Load(const IteratorType &it1, const IteratorType &it2) {
      this->SetAll(0.0);
      for(IteratorType it=it1; it!=it2; ++it) {
        set(it->first, it->second);
      }
    }
    template<typename PointType1, typename PointType2>
    struct CopyValuesOperatorsDense {
     public:
      CopyValuesOperatorsDense(
        PointType1 * const x, const PointType2 &y,
        index_t *current_index) :
          x_(x), y_(y), ind_(current_index) {
      }

      template<typename T>
      void operator()(T) {
        PointType1 temp;
        temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(
                     x_->ptr()) + *ind_,
                   y_.template dense_point<T>().size());
        temp.CopyValues(y_.template dense_point<T>());
        *ind_ += y_.template dense_point<T>().size();
      }

private:
      PointType1 * const x_;
      const PointType2 &y_;
      index_t *ind_;
    };

    template<typename PointType1, typename PointType2>
    struct CopyValuesOperatorsSparse {
public:
      CopyValuesOperatorsSparse(
        PointType1 * const x, const PointType2 &y,
        index_t *current_index) :
          x_(x), y_(y), ind_(current_index) {
      }

      template<typename T>
      void operator()(T) {
        PointType1 temp;
        temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(
                     x_->ptr()) + *ind_,
                   y_.template sparse_point<T>().size());
        temp.CopyValues(y_.template sparse_point<T>());
        *ind_ += y_.template sparse_point<T>().size();
      }

private:
      typename PointType2::CalcPrecision_t alpha_;
      PointType1 * const x_;
      const PointType2 &y_;
      index_t *ind_;
    };


    template<typename StreamType>
    void Print(StreamType &stream, std::string delim) const {
      for (size_t i = 0; i < this->size(); i++) {
        stream << this->operator[](i) << delim;
      }
    }
    // serialization
    template<typename Archive>
    void save(Archive &ar,
              const unsigned int file_version) const {
      ar << boost::serialization::make_nvp("matrix",  
          boost::serialization::base_object<
            fl::dense::Matrix<CalcPrecisionType, true> >(*this));
    }

    template<typename Archive>
    void load(Archive &ar,
              const unsigned int file_version) {
      ar >> boost::serialization::make_nvp("matrix", 
          boost::serialization::base_object<
            fl::dense::Matrix<CalcPrecisionType, true> >(*this));

    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()


    /**
     *  @brief we need to reintroduce the dot product because the monolithic, sparse dot product
     *         shadows the one in the base class
     *
     */
    static inline long double Dot(const MonolithicPoint<CalcPrecision_t> &x,
                                      const MonolithicPoint<CalcPrecision_t> &y) {
      return fl::dense::ops::Dot(x, y);
    }

    static inline long double Dot(const fl::dense::Matrix<CalcPrecision_t, true> &x,
                                      const fl::dense::Matrix<CalcPrecision_t, true> &y) {
      return fl::dense::ops::Dot(x, y);
    }

    static inline long double Dot(const MonolithicPoint<CalcPrecision_t> &x,
                                  const MonolithicPoint<CalcPrecision_t> &W,      
                                  const MonolithicPoint<CalcPrecision_t> &y) {
      return fl::dense::ops::Dot(x, W, y);
    }

    static inline long double Dot(const MonolithicPoint<CalcPrecision_t> &x,
                                  const fl::dense::Matrix<CalcPrecision_t, false> &W,      
                                  const MonolithicPoint<CalcPrecision_t> &y) {
      return fl::dense::ops::Dot(x, W, y);
    }
    /**
     *  @brief the outer product between two points V x P
     */
    template<typename ResultMatrixType>
    void UpdateOuterProd(const MonolithicPoint<CalcPrecision_t> &x,
                          ResultMatrixType *res) const {
      for(index_t i=0; i<this->size(); ++i) {
        for(index_t j=0; j<x.size(); ++j) {
          res->set(i, j, res->get(i,j)
              +this->operator[](i)*static_cast<double>(x[j]));
        }
      }
    }

    /**
     *  @brief the outer product between two points
     *         the only difference is that we add an offset
     *         on the second point
     */
    template<typename ResultMatrixType>
    void UpdateOuterProd(const MonolithicPoint<CalcPrecision_t> &x,
                         index_t offset,
                         ResultMatrixType *res) const {
      for(index_t i=0; i<this->size(); ++i) {
        for(index_t j=0; j<x.size(); ++j) {
          res->set(i, j+offset, res->get(i, j+offset)
              +this->operator[](i)*static_cast<double>(x[j]));
        }
      }
    }

    /**
     *  @brief the outer product between this point and its self
     */
    template<typename ResultMatrixType>
    void UpdateSelfOuterProd(ResultMatrixType *res) const {
      UpdateSelfOuterProd(res, 0);
    }
    template<typename ResultMatrixType>
    void UpdateSelfOuterProd(ResultMatrixType *res,
        index_t offset) const {
      for(index_t i=0; i<this->size(); ++i) {
        for(index_t j=0; j<i; ++j) {
          double val=this->operator[](i) * 
            static_cast<double>(this->operator[](j));
          res->set(i+offset, 
                   j+offset, 
                   res->get(i+offset, j+offset)+val);
          res->set(j+offset, 
                   i+offset, 
                   res->get(j+offset, i+offset)+val);

        }
      }
      for(index_t i=0; i<this->size(); ++i) {
        double val=this->operator[](i)* 
          static_cast<double>(this->operator[](i));
          res->set(i+offset, 
                   i+offset, 
                   res->get(i+offset, i+offset)+val);

      }
    }

    /**
     *  @brief the outer product between a monolithic point and a sparse one
     */
    template<typename ResultMatrixType, typename PrecisionType2>
    void UpdateOuterProd(const SparsePoint<PrecisionType2> &x,
                          ResultMatrixType *res) const {
      typedef fl::data::SparsePoint<PrecisionType2> SparsePoint_t;
      typename SparsePoint_t::Container_t::iterator it;
      for(it=x.begin(); it!=x.end(); ++it) {
        for(index_t i=0; i<this->size(); ++i) {
          res->set(i, it->first(),
              res->get(i, it->first)
              +this->operator[](i)*static_cast<double>(it->second));
        }
      }
    }
    
    /**
     *  @brief we introduce an outer product between a monolithic point
     *         and a sparse one
     */
    template<typename ResultMatrixType,
             typename PointType1, 
             typename PointType2>
    struct UpdateOuterProdOperatorsDense {
      public:
        UpdateOuterProdOperatorsDense(
            const PointType1 &x, 
            const PointType2 &y,
            ResultMatrixType *result,
            index_t *current_index) :
            x_(x), y_(y), result_(result), ind_(current_index) {
        }

        template<typename T>
        void operator()(T) {
          PointType1 temp;
          x_.UpdateOuterProject(
              y_.template dense_point<T>(),
              ind_,
              result_);
          *ind_ += y_.template dense_point<T>().size();
        } 

      private:
        const PointType1 &x_;
        const PointType2 &y_;
        index_t *ind_;
        ResultMatrixType *result_;
    };

    template<typename ResultMatrixType,
             typename PointType1, 
             typename PointType2>
    struct UpdateOuterProdOperatorsSparse {
      public:
        UpdateOuterProdOperatorsSparse(
            const PointType1 &x, 
            const PointType2 &y,
            ResultMatrixType *result,
            index_t *current_index) :
            x_(x), y_(y), result_(result), ind_(current_index) {
        }

        template<typename T>
        void operator()(T) {
          PointType1 temp;
          x_.UpdateOuterProject(
              y_.template sparse_point<T>(),
              ind_,
              result_);
          *ind_ += y_.template sparse_point<T>().size();
        } 

      private:
        const PointType1 &x_;
        const PointType2 &y_;
        index_t *ind_;
        ResultMatrixType *result_;
    };

    template<typename ResultMatrixType, typename MixedPointArgs>
    void UpdateOuterProd(
        const fl::data::MixedPoint<MixedPointArgs> &x,
        ResultMatrixType *result) const {

      typedef MixedPoint<MixedPointArgs> MixedPoint_t; 
      typedef typename MixedPoint_t::DenseTypes_t  DenseTypes_t;
      typedef typename MixedPoint_t::SparseTypes_t SparseTypes_t;
      index_t current_index=0;
      boost::mpl::for_each<DenseTypes_t>(
        UpdateOuterProdOperatorsDense<
            ResultMatrixType,
            MonolithicPoint<CalcPrecision_t>, 
            MixedPoint_t>(*this, x, result, &current_index));
      boost::mpl::for_each<SparseTypes_t>(
        UpdateOuterProdOperatorsSparse<
            ResultMatrixType,
            MonolithicPoint<CalcPrecision_t>, 
            MixedPoint_t>(*this, x, result, &current_index));
    
    }

    /**
     *  @brief monolithic multiplied with a sparse
     *
     */
    template<typename PrecisionType1, typename PrecisionType2>
    static inline double  Dot(const fl::data::MonolithicPoint<PrecisionType1> &x,
        const fl::data::SparsePoint<PrecisionType2> &y) {
      typedef fl::data::SparsePoint<PrecisionType2> SparsePoint_t;
      typedef fl::data::MonolithicPoint<PrecisionType1> MonolithicPoint_t;
      typename SparsePoint_t::Container_t::iterator it2 = y.begin();
      typename SparsePoint_t::Container_t::iterator end2 = y.end();
      typename MonolithicPoint::CalcPrecision_t result = 0;
      while (it2 != end2) {
        BOOST_ASSERT(static_cast<size_t>(it2->first) < x.size() &&  it2->first >= 0);
        if (boost::is_same<PrecisionType2, bool>::value) {
          result += x[it2->first];
        }
        else {
          result += x[it2->first] * static_cast<double>(it2->second);
        }

        ++it2;
      }
      return result;
    }

    /**
     *  @brief we introduce a specialization for dense versus mixed_point
     */
    template<typename PrecisionType, typename MixedPointArgs>
    static inline double Dot(const fl::data::MonolithicPoint<PrecisionType> &x,
        const fl::data::MixedPoint<MixedPointArgs> &y) {

      typedef fl::data::MonolithicPoint<PrecisionType> MonolithicPoint_t;
      typedef fl::data::MixedPoint<MixedPointArgs> MixedPoint_t;
      typedef typename MixedPoint_t::DenseTypes_t  DenseTypes_t;
      typedef typename MixedPoint_t::SparseTypes_t SparseTypes_t;
      CalcPrecision_t result = 0;
      index_t current_index = 0;
      //BOOST_MPL_ASSERT((boost::is_same<SparseTypes_t, int>));
      boost::mpl::for_each<DenseTypes_t>(
        DotOperatorsDense<MonolithicPoint_t, MixedPoint_t>(x, y, &current_index, &result));
      boost::mpl::for_each<SparseTypes_t>(
        DotOperatorsSparse<MonolithicPoint_t, MixedPoint_t>(x, y, &current_index, &result));

      return result;
    }

    template<typename PointType1, typename PointType2>
    struct DotOperatorsDense {
      public:
        DotOperatorsDense(const PointType1 &x, const PointType2 &y,
                          index_t *current_index, typename PointType1::CalcPrecision_t *result) :
            x_(x), y_(y), ind_(current_index), result_(result) {
        }

        template<typename T>
        void operator()(T) {
          PointType1 temp;
          temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(x_.ptr()) + *ind_,
                   y_.template dense_point<T>().size());
          *result_ += fl::dense::ops::Dot(temp, y_.template dense_point<T>());
          *ind_ += y_.template dense_point<T>().size();
        } 

      private:
        const PointType1 &x_;
        const PointType2 &y_;
        index_t *ind_;
        typename PointType1::CalcPrecision_t *result_;
    };

    template<typename PointType1, typename PointType2>
    struct DotOperatorsSparse {
      public:
        DotOperatorsSparse(const PointType1 &x, const PointType2 &y,
                         index_t *current_index, typename PointType1::CalcPrecision_t *result) :
          x_(x), y_(y), ind_(current_index), result_(result) {
        }

        template<typename T>
        void operator()(T) {
          PointType1 temp;
          temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(x_.ptr()) + *ind_,
                     y_.template sparse_point<T>().size());
          *result_ += PointType1::Dot(temp, y_.template sparse_point<T>());
          *ind_ += y_.template sparse_point<T>().size();

        } 

      private:
        const PointType1 &x_;
        const PointType2 &y_;
        index_t *ind_;
        typename PointType1::CalcPrecision_t *result_;
    };
    /**
     *  @brief we need to reintroduce the AddExpert product because the monolithic,
     *  sparse AddExpert product shadows the one in the base class
     *
     */
    static inline void AddExpert(const CalcPrecision_t alpha,
                                 const MonolithicPoint<CalcPrecision_t> &x,
                                 MonolithicPoint<CalcPrecision_t>* const y) {
      fl::dense::ops::AddExpert(alpha, x, y);
    }

    /**
     *  @brief monolithic added with a sparse
     *
     */
    template<typename PrecisionType1, typename PrecisionType2>
    static inline void AddExpert(const PrecisionType1 alpha,
                                 const fl::data::SparsePoint<PrecisionType2> &x,
                                 fl::data::MonolithicPoint<PrecisionType1> *y) {
      typedef fl::data::SparsePoint<PrecisionType2> SparsePoint_t;
      typedef fl::data::MonolithicPoint<PrecisionType1> MonolithicPoint_t;
      typename SparsePoint_t::Container_t::iterator it1 = x.begin();
      typename SparsePoint_t::Container_t::iterator end1 = x.end();

      while (it1 != end1) {
        BOOST_ASSERT(static_cast<size_t>(it1->first) < y->size() &&  it1->first >= 0);
        if (boost::is_same<PrecisionType2, bool>::value) {
          (*y)[it1->first] += alpha;
        }
        else {
          (*y)[it1->first] += it1->second * alpha;
        }
        ++it1;
      }
    }

    /**
     *  @brief we introduce a specialization for dense versus mixed_point
     */
    template<typename PrecisionType, typename MixedPointArgs>
    static inline void AddExpert(const PrecisionType alpha,
                                 const fl::data::MixedPoint<MixedPointArgs> &x,
                                 const fl::data::MonolithicPoint<PrecisionType> *y) {

      typedef fl::data::MonolithicPoint<PrecisionType> MonolithicPoint_t;
      typedef fl::data::MixedPoint<MixedPointArgs> MixedPoint_t;
      typedef typename MixedPoint_t::DenseTypes_t  DenseTypes_t;
      typedef typename MixedPoint_t::SparseTypes_t SparseTypes_t;
      index_t current_index = 0;
      boost::mpl::for_each<DenseTypes_t>(
        AddExpertOperatorsDense<MixedPoint_t, MonolithicPoint_t>(alpha, x, y, &current_index));
      boost::mpl::for_each<SparseTypes_t>(
        AddExpertOperatorsSparse<MixedPoint_t, MonolithicPoint_t>(alpha, x, y, &current_index));

    }

    template<typename PointType1, typename PointType2>
    struct AddExpertOperatorsDense {
public:
      AddExpertOperatorsDense(const typename PointType2::CalcPrecision_t alpha,
                              const PointType1 &x, const PointType2 *y,
                              index_t *current_index) :
          alpha_(alpha), x_(x), y_(y), ind_(current_index) {
      }

      template<typename T>
      void operator()(T) {
        PointType2 temp;
        temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(y_->ptr()) + *ind_,
                   x_.template dense_point<T>().size());
        fl::dense::ops::AddExpert(alpha_, x_.template dense_point<T>(), &temp);
        *ind_ += x_.template dense_point<T>().size();
      }

private:
      typename PointType2::CalcPrecision_t alpha_;
      const PointType1 &x_;
      const PointType2 *y_;
      index_t *ind_;
    };

    template<typename PointType1, typename PointType2>
    struct AddExpertOperatorsSparse {
public:
      AddExpertOperatorsSparse(const typename PointType2::CalcPrecision_t alpha,
                               const PointType1 &x, const PointType2 *y,
                               index_t *current_index) :
          alpha_(alpha), x_(x), y_(y), ind_(current_index) {
      }

      template<typename T>
      void operator()(T) {
        PointType2 temp;
        temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(y_->ptr()) + *ind_,
                   x_.template sparse_point<T>().size());
        fl::la::AddExpert(alpha_, x_.template sparse_point<T>(), &temp);
        *ind_ += x_.template sparse_point<T>().size();
      }

private:
      typename PointType2::CalcPrecision_t alpha_;
      const PointType1 &x_;
      const PointType2 *y_;
      index_t *ind_;
    };

    /**
     *  @brief we need to reintroduce the RawLMetric because the monolithic,
     *  sparse RawLMetric shadows the one in the base class
     *
     */
    template<int t_pow>
    class RawLMetric {
      public:
        template < typename PrecisionType1,
        typename PrecisionType2,
        typename PrecisionType3 >
        RawLMetric(
          const MonolithicPoint<PrecisionType1> &x,
          const MonolithicPoint<PrecisionType2> &y,
          PrecisionType3 *result) {
          fl::dense::ops::RawLMetric<t_pow>(x, y, result);
        }

        template < typename PrecisionType1,
        typename PrecisionType2,
        typename PrecisionType3,
        typename PrecisionType4 >
        RawLMetric(
          const MonolithicPoint<PrecisionType3> &w,
          const MonolithicPoint<PrecisionType1> &x,
          const MonolithicPoint<PrecisionType2> &y,
          PrecisionType4 *result) {
          fl::dense::ops::RawLMetric<t_pow>(w, x, y, result);
        }

        /**
         *  @brief monolithic added with a sparse
         *
         */
        template < typename PrecisionType1, typename PrecisionType2,
        typename PrecisionType3 >
        RawLMetric(
          const fl::data::MonolithicPoint<PrecisionType1> &y,
          const fl::data::SparsePoint<PrecisionType2> &x,
          PrecisionType3 *result) {
          typedef fl::data::SparsePoint<PrecisionType2> SparsePoint_t;
          typedef fl::data::MonolithicPoint<PrecisionType1> MonolithicPoint_t;
          typename SparsePoint_t::Container_t::iterator it1 = x.begin();
          typename SparsePoint_t::Container_t::iterator end1 = x.end();
          if (t_pow == 2) {
            int i=0;
            if(it1 != end1) {
              for(i=0; i<y.size(); ++i) {
                if (i==it1->first) {
                  *result += fl::math::Sqr(y[i]-it1->second);
                  ++it1;
                  if (it1==end1) {
                    break;
                  }
                } else {
                  *result+=y[i]*y[i];
                }
              } // for
            } else { --i; }
            for(int j=i+1; j<y.size(); ++j) {
              *result+=y[j]*y[j];           
            }
            
          } else {
            // This hasn't been implemented yet
            BOOST_ASSERT(false);
          }
        }
        template < typename PrecisionType1, typename PrecisionType2,
        typename PrecisionType3 >
        RawLMetric(
          const fl::data::MonolithicPoint<PrecisionType1> &y,
          const fl::data::MonolithicPoint<PrecisionType1> &w,
          const fl::data::SparsePoint<PrecisionType2> &x,
          PrecisionType3 *result) {
          typedef fl::data::SparsePoint<PrecisionType2> SparsePoint_t;
          typedef fl::data::MonolithicPoint<PrecisionType1> MonolithicPoint_t;
          typename SparsePoint_t::Container_t::iterator it1 = x.begin();
          typename SparsePoint_t::Container_t::iterator end1 = x.end();
          if (t_pow == 2) {
            int i=0;
            if(it1 != end1) {
              for(i=0; i<y.size(); ++i) {
                if (i==it1->first) {
                  *result += w[i]*fl::math::Sqr(y[i]-it1->second);
                  ++it1;
                  if (it1==end1) {
                    break;
                  }
                } else {
                  *result+=w[i]*y[i]*y[i];
                }
              }
            } else { --i; }
            for(int j=i+1; j<y.size(); ++j) {
              *result+=w[j]*y[j]*y[j];           
            }
          }
          else {
            // This hasn't been implemented yet
            BOOST_ASSERT(false);
          }
        }
  
        /**
         *  @brief we introduce a specialization for dense versus mixed_point
         */
        template < typename PrecisionType1, typename MixedPointArgs,
        typename PrecisionType2 >
        RawLMetric(
          const fl::data::MixedPoint<MixedPointArgs> &x,
          const fl::data::MonolithicPoint<PrecisionType1> &y,
          PrecisionType2 *result) {
          RawLMetric(y, x, result);
        }

        template < typename PrecisionType1, typename MixedPointArgs,
        typename PrecisionType2 >
        RawLMetric(
          const fl::data::MonolithicPoint<PrecisionType1> &y,
          const fl::data::MixedPoint<MixedPointArgs> &x,
          PrecisionType2 *result) {

          typedef fl::data::MonolithicPoint<PrecisionType1> MonolithicPoint_t;
          typedef fl::data::MixedPoint<MixedPointArgs> MixedPoint_t;
          typedef typename MixedPoint_t::DenseTypes_t  DenseTypes_t;
          typedef typename MixedPoint_t::SparseTypes_t SparseTypes_t;
          index_t current_index = 0;
          *result = 0;
          boost::mpl::for_each<DenseTypes_t>(
            RawLMetricDense < MixedPoint_t,
            MonolithicPoint_t,
            PrecisionType2 > (x, y, &current_index, result));
          boost::mpl::for_each<SparseTypes_t>(
            RawLMetricSparse < MixedPoint_t,
            MonolithicPoint_t,
            PrecisionType2 > (x, y, &current_index, result));

        }

        template < typename PrecisionType1, typename MixedPointArgs,
        typename PrecisionType2 >
        RawLMetric(
          const fl::data::MonolithicPoint<PrecisionType1> &y,
          const fl::data::MonolithicPoint<PrecisionType2> &w,
          const fl::data::MixedPoint<MixedPointArgs> &x,
          PrecisionType2 *result) {

          typedef fl::data::MonolithicPoint<PrecisionType1> MonolithicPoint_t;
          typedef fl::data::MixedPoint<MixedPointArgs> MixedPoint_t;
          typedef typename MixedPoint_t::DenseTypes_t  DenseTypes_t;
          typedef typename MixedPoint_t::SparseTypes_t SparseTypes_t;
          index_t current_index = 0;
          *result = 0;
          boost::mpl::for_each<DenseTypes_t>(
            RawLMetricDenseW < MixedPoint_t,
            MonolithicPoint_t,
            PrecisionType2 > (x, w, y, &current_index, result));
          boost::mpl::for_each<SparseTypes_t>(
            RawLMetricSparseW < MixedPoint_t,
            MonolithicPoint_t,
            PrecisionType2 > (x, w, y, &current_index, result));

        }

        template<typename PointType1, typename PointType2, typename PrecisionType>
        struct RawLMetricDense {
         public:
          RawLMetricDense(
            const PointType1 &x, const PointType2 &y,
            index_t *current_index, PrecisionType *result) :
              x_(x), y_(y), ind_(current_index), result_(result)  {
          }

          template<typename T>
          void operator()(T) {
            PointType2 temp;
            temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(y_.ptr()) + *ind_,
                       x_.template dense_point<T>().size());
            PrecisionType temp_result = 0;
            RawLMetric<t_pow>(temp, x_.template dense_point<T>(), &temp_result);
            (*result_) += temp_result;
            *ind_ += x_.template dense_point<T>().size();
          }

          private:

          typename PointType2::CalcPrecision_t alpha_;
          const PointType1 &x_;
          const PointType2 &y_;
          index_t *ind_;
          PrecisionType *result_;
        };

        template<typename PointType1, typename PointType2, typename PrecisionType>
        struct RawLMetricSparse {
         public:
          RawLMetricSparse(
            const PointType1 &x, const PointType2 &y,
            index_t *current_index, PrecisionType *result) :
              x_(x), y_(y), ind_(current_index), result_(result)  {
          }

          template<typename T>
          void operator()(T) {
            PointType2 temp;
            PrecisionType temp_result=0;
            temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(y_.ptr()) + *ind_,
                       x_.template sparse_point<T>().size());
            RawLMetric<t_pow>(temp, x_.template sparse_point<T>(), &temp_result);
            (*result_) += temp_result;
            *ind_ += x_.template sparse_point<T>().size();
          }

         private:
          typename PointType2::CalcPrecision_t alpha_;
          const PointType1 &x_;
          const PointType2 &y_;
          index_t *ind_;
          PrecisionType *result_;
        };
  
        template<typename PointType1, typename PointType2, typename PrecisionType>
        struct RawLMetricDenseW {
         public:
          RawLMetricDenseW(
            const PointType1 &x, const fl::data::MonolithicPoint<PrecisionType> &w, 
            const PointType2 &y,
            index_t *current_index, PrecisionType *result) :
              x_(x), w_(w), y_(y), ind_(current_index), result_(result)  {
          }

          template<typename T>
          void operator()(T) {
            PointType2 temp;
            temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(y_.ptr()) + *ind_,
                       x_.template dense_point<T>().size());
            PrecisionType temp_result = 0;
            RawLMetric<t_pow>(temp, w_, x_.template dense_point<T>(), &temp_result);
            (*result_) += temp_result;
            *ind_ += x_.template dense_point<T>().size();
          }

          private:

          typename PointType2::CalcPrecision_t alpha_;
          const PointType1 &x_;
          const fl::data::MonolithicPoint<PrecisionType> &w_;
          const PointType2 &y_;
          index_t *ind_;
          PrecisionType *result_;
        };

        template<typename PointType1, typename PointType2, typename PrecisionType>
        struct RawLMetricSparseW {
         public:
          RawLMetricSparseW(
            const PointType1 &x, 
            const fl::data::MonolithicPoint<PrecisionType> &w,
            const PointType2 &y,
            index_t *current_index, PrecisionType *result) :
              x_(x), w_(w), y_(y), ind_(current_index), result_(result)  {
          }

          template<typename T>
          void operator()(T) {
            PointType2 temp;
            PrecisionType temp_result=0;
            temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(y_.ptr()) + *ind_,
                       x_.template sparse_point<T>().size());
            RawLMetric<t_pow>(temp, w_, x_.template sparse_point<T>(), &temp_result);
            (*result_) += temp_result;
            *ind_ += x_.template sparse_point<T>().size();
          }

         private:
          typename PointType2::CalcPrecision_t alpha_;
          const PointType1 &x_;
          const fl::data::MonolithicPoint<PrecisionType> &w_;
          const PointType2 &y_;
          index_t *ind_;
          PrecisionType *result_;
        };
    }; // RawLMetric


    /**
     *  @brief we need to reintroduce the AddTo product because the monolithic,
     *  sparse AddTo product shadows the one in the base class
     *
     */
    static inline void AddTo(
      const MonolithicPoint<CalcPrecision_t> &x,
      MonolithicPoint<CalcPrecision_t>* const y) {
      fl::dense::ops::AddTo(x, y);
    }

    /**
     *  @brief monolithic added with a sparse
     *
     */
    template<typename PrecisionType1, typename PrecisionType2>
    static inline void AddTo(
      const fl::data::SparsePoint<PrecisionType2> &x,
      fl::data::MonolithicPoint<PrecisionType1> *y) {
      AddExpert(PrecisionType1(1), x, y);
    }

    /**
     *  @brief we introduce a specialization for dense versus mixed_point
     */
    template<typename PrecisionType, typename MixedPointArgs>
    static inline void AddTo(
      const fl::data::MixedPoint<MixedPointArgs> &x,
      fl::data::MonolithicPoint<PrecisionType> *y) {

      AddExpert(PrecisionType(1), x, y);
    }
    /**
     *  @brief we need to reintroduce the MulTo product because the monolithic,
     *  sparse DotMulTo product shadows the one in the base class
     *
     */
    static inline void DotMulTo(
      const MonolithicPoint<CalcPrecision_t> &x,
      MonolithicPoint<CalcPrecision_t>* const y) {
      fl::dense::ops::DotMulTo(x, y);
    }

    /**
     *  @brief monolithic dotmulto with a sparse
     *
     */
    template<typename PrecisionType1, typename PrecisionType2>
    static inline void DotMulTo(
      const fl::data::MonolithicPoint<PrecisionType2> &x,
      fl::data::SparsePoint<PrecisionType1> *y) {
      typename fl::data::SparsePoint<PrecisionType1>::Container_t::iterator it, it1;
      it = y->begin();
      while (it != y->end()) {
        if (x[it->first] == 0) {
          it1 = it;
          ++it;
          y->set(it1->first, 0);
        }
        else {
          it->second *= x[it->first];
          ++it;
        }
      }
    }

    /**
     *  @brief we introduce a specialization for dense versus mixed_point
     */
    template<typename PointType1, typename PointType2>
    struct DotMulToDense {
      public:
        DotMulToDense(const PointType1 &x,  PointType2 *y,
                      index_t *current_index) :
            x_(x), y_(y), ind_(current_index) {
        }

        template<typename T>
        void operator()(T) {
          PointType1 temp;
          temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(x_.ptr()) + *ind_,
                     y_->template dense_point<T>().size());
          fl::dense::ops::DotMulTo(temp, &(y_->template dense_point<T>()));
          *ind_ += y_->template dense_point<T>().size();
        }

      private:
        const PointType1 &x_;
        PointType2 *y_;
        index_t *ind_;
    };
    template<typename PointType1, typename PointType2>
    struct DotMulToSparse {
      public:
        DotMulToSparse(const PointType1 &x,  PointType2 *y,
                       index_t *current_index) :
            x_(x), y_(y), ind_(current_index) {
        }

        template<typename T>
        void operator()(T) {
          PointType1 temp;
          temp.Alias(const_cast<typename PointType1::CalcPrecision_t*>(x_.ptr()) + *ind_,
                     y_->template sparse_point<T>().size());
          fl::la::DotMulTo(temp, &(y_->template sparse_point<T>()));
          *ind_ += y_->template sparse_point<T>().size();
        }

      private:
        const PointType1 &x_;
        PointType2 *y_;
        index_t *ind_;
    };

    template<typename PrecisionType, typename MixedPointArgs>
    static inline void DotMulTo(
      const fl::data::MonolithicPoint<PrecisionType> &x,
      fl::data::MixedPoint<MixedPointArgs> *y) {

      typedef fl::data::MonolithicPoint<PrecisionType> MonolithicPoint_t;
      typedef fl::data::MixedPoint<MixedPointArgs> MixedPoint_t;
      typedef typename MixedPoint_t::DenseTypes_t  DenseTypes_t;
      typedef typename MixedPoint_t::SparseTypes_t SparseTypes_t;
      index_t current_index = 0;
      boost::mpl::for_each<DenseTypes_t>(
        DotMulToDense<MonolithicPoint_t, MixedPoint_t>(x, y, &current_index));
      boost::mpl::for_each<SparseTypes_t>(
        DotMulToSparse<MonolithicPoint_t, MixedPoint_t>(x, y, &current_index));
    }

    /**
     *  @brief we need to reintroduce the DotMul product because the monolithic,
     *  sparse DotMulTo product shadows the one in the base class
     *
     */

    template<fl::la::MemoryAlloc MemAlloc>
    class DotMul {
      public:
        template<typename PrecisionType, typename MixedPointArgs>
        DotMul(const fl::data::MonolithicPoint<PrecisionType> &x,
               const fl::data::MixedPoint<MixedPointArgs> &y,
               fl::data::MixedPoint<MixedPointArgs> *z) {
          if (MemAlloc==fl::la::Init) {
            z->Copy(y);
          } else {
            z->CopyValues(y);
          }
          DotMulTo(x,z);
        }

        DotMul(
            const MonolithicPoint<CalcPrecision_t> &x,
            const MonolithicPoint<CalcPrecision_t> &y,
            MonolithicPoint<CalcPrecision_t>* const z) {
          fl::dense::ops::DotMul<MemAlloc>(x, y, z);
        }
    };

};

} // namespace dense
}  // namespace fl

#endif
