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
#ifndef FASTLIB_DATA_SPARSE_POINT_H_
#define FASTLIB_DATA_SPARSE_POINT_H_
#include <new>
#include <map>
#include <set>
#include <vector>
#include "monolithic_point.h"
#include "mixed_point.h"
#include "boost/mpl/if.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/type_traits.hpp"
#include "boost/serialization/split_member.hpp"
#include "boost/serialization/nvp.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/map.hpp"
#include "boost/serialization/tracking.hpp"
#include "fastlib/base/base.h"
#include "fastlib/la/linear_algebra_defs.h"

namespace fl {
namespace data {
template<typename CalcPrecisionType>
class SparsePoint {
  public:
    friend class boost::serialization::access;
    typedef boost::mpl::vector0<> DenseMemberTypes_t;
    typedef boost::mpl::vector1<CalcPrecisionType> SparseMemberTypes_t;
    template<typename T> friend  class MultiDataset;
    typedef CalcPrecisionType CalcPrecision_t;
    typedef std::vector<std::pair<index_t, CalcPrecision_t> > Container_t;
    //typedef std::map<index_t, CalcPrecision_t> Container_t;
    typedef typename Container_t::iterator Iterator;
    class iterator : public Iterator {
      public:

        iterator() {
        }

        iterator(const Iterator &other) : Iterator(other) {
        }

        const iterator& operator=(const iterator& other) {
           Iterator::operator=(static_cast<Iterator>(other));
           return *this;
        }

        const iterator& operator=(const Iterator& other) {
          Iterator::operator=(other);
          return *this;
        }

        bool operator==(const iterator &other) const {
          return Iterator::operator==(static_cast<Iterator>(other));
        }

        bool operator==(const Iterator &other) const {
          // return Iterator::operator==(other);
          return static_cast<Iterator>(*this)==other;
        }

        bool operator!=(const iterator &other) const {
          //return !Iterator::operator==(static_cast<Iterator>(other));
          return !this->operator==(static_cast<Iterator>(other));

        }

        bool operator!=(const Iterator &other) const {
          return static_cast<Iterator>(*this)!=(other);
        }

        CalcPrecision_t &operator*()  {
          return (*this)->second;
        }

        const CalcPrecision_t &operator*() const {
          return (*this)->second;
        }


        CalcPrecision_t &value() {
          return (*this)->second;
        }

        const CalcPrecision_t &value() const {
          return (*this)->second;
        }

        const index_t &attribute() const {
          return (*this)->first;
        }

        const index_t &attribute() {
          return (*this)->first;
        }

    };
    SparsePoint() : elem_(NULL),
        size_(0), should_free_(false) {};

    SparsePoint(index_t size) : elem_(new (std::nothrow) Container_t()),
        size_(size), should_free_(true) {
      if (elem_==NULL) {
        fl::logger->Die() << "There was a problem allocating memory. "
          << "Either your dataset doesn't fit in the RAM, or"
          << "you are using a 32bit platform that limits the process "
          << "address space to 4GB";
      }    
    };

    SparsePoint(const SparsePoint &other) {
      size_ = other.size_ ;
      should_free_ = other.should_free_;
      if (other.elem_ != NULL) {
        try {
          elem_ = new Container_t();
        }
        catch(const std::bad_alloc &e) {
          fl::logger->Die() << "There was a problem allocating memory. "
            << "Either your dataset doesn't fit in the RAM, or"
            << "you are using a 32bit platform that limits the process "
            << "address space to 4GB";
        }
        *elem_ = *(other.elem_);
      }
    }

    virtual ~SparsePoint() {
      if (should_free_ == true) {
        delete elem_;
      }
    }

    void Init(index_t size) {
      should_free_ = true;
      try {
        elem_ = new Container_t();
      }
      catch(const std::bad_alloc &e) {
        fl::logger->Die() << "There was a problem allocating memory. "
          << "Either your dataset doesn't fit in the RAM, or"
          << "you are using a 32bit platform that limits the process "
          << "address space to 4GB";
      }
      size_ = size;
    }

    template<typename ContainerType>
    void Init(ContainerType &sizes) {
      BOOST_ASSERT(sizes.size() == 1);
      Init(sizes[0]);
    }

    void Copy(const SparsePoint& other) {
      size_ = other.size_;
      if (should_free_ == true) {
        delete elem_;
      }
      if (other.elem_ != NULL) {
        try {
          elem_ = new Container_t();
        }
        catch(const std::bad_alloc &e) {
          fl::logger->Die() << "There was a problem allocating memory. "
            << "Either your dataset doesn't fit in the RAM, or"
            << "you are using a 32bit platform that limits the process "
            << "address space to 4GB";
        }
        *elem_ = *(other.elem_);
      }
      should_free_ = true;
    }

    void CopyValues(const SparsePoint& other) {
      size_ = other.size_;
      *elem_ = *(other.elem_);
    }

    template<typename PrecisionType>
    void Copy(const MonolithicPoint<PrecisionType>& other) {
      Init(other.size());
      CopyValues(other);
    }

    template<typename PrecisionType>
    void CopyValues(const MonolithicPoint<PrecisionType>& other) {
      DEBUG_ASSERT(size()==other.size());
      elem_->clear();
      typename MonolithicPoint<PrecisionType>::iterator it, end;
      end = other.end();
      for (it = other.begin(); it != end; ++it) {
        this->set(it->first, it->second);
      }
    }

    template<typename ParamsType>
    void Copy(const MixedPoint<ParamsType>& other) {
      Init(other.size());
      CopyValues(other);
    }

    template<typename ParamsType>
    void CopyValues(const MixedPoint<ParamsType>& other) {
      fl::logger->Die() << "Copying from MixedPoint to Sparse is not "
                        << "supported yet\n";
    }

    SparsePoint& operator=(const SparsePoint &other) {
      CopyValues(other);
      return *this;
    }

    iterator begin()  {
      return elem_->begin();
    }

    const Iterator begin()  const {
      return elem_->begin();
    }

    Iterator end() {
      return elem_->end();
    }

    const Iterator end() const {
      return elem_->end();
    }
    bool IsZero() const {
      for(Iterator it=begin();it!=end(); ++it) {
        if (it->second!=0) {
          return false;
        }
      }
      return true;
    }
    template<typename FunctionType>
    void Transform(FunctionType &f) {
      for(Iterator it=elem_->begin(); it!=elem_->end(); ++it) {
        f(it->first, &(it->second));
      }
    }
    void SetAll(CalcPrecision_t value) {
      elem_->clear();
      if (value != CalcPrecision_t(0)) {
        for (index_t i = 0 ; i < size_; i++) {
          boost::mpl::if_<
            boost::is_same<
              Container_t,
              std::map<index_t, CalcPrecision_t>
            >,
            InsertElement1,
            InsertElement2
          >::type::Set(elem_, i, value);
          // elem_->insert(std::make_pair(i, value));
        }
      }
    }

    struct SetRandom1 {
      struct type {
        static void Set(Container_t *elem, CalcPrecision_t low, 
            CalcPrecision_t hi, 
            index_t num_non_zeros, int size) {
          elem->clear();
          int i=0;
          std::set<index_t> unique_ids;
          while(i<num_non_zeros) {
            index_t id = fl::math::Random(0, size-1);
            if (unique_ids.find(id)==unique_ids.end()) {
              elem->push_back(std::make_pair(id, fl::math::Random(low, hi)));
              unique_ids.insert(id);
              ++i;
            }
          }
          std::sort(elem->begin(), elem->end());
        }
      };
    };

    struct SetRandom2 {
      struct type {
        static void Set(Container_t *elem, CalcPrecision_t low, 
            CalcPrecision_t hi, 
            index_t num_non_zeros, int size) {
          elem->clear();
          for(index_t i=0; i<num_non_zeros; ++i) {
            index_t id = fl::math::Random(0, size-1);
            elem->insert(std::make_pair(id, fl::math::Random(low, hi)));          
          }
        }
      };
    };

    void SetRandom(const CalcPrecision_t low, 
                   const CalcPrecision_t hi,
                   const double sparsity) {
      if (sparsity <0 || sparsity>1) {
        fl::logger->Die() << "Sparsity must be between 0 and 1, but you gave "<<
          sparsity;
      }
      index_t num_non_zeros=(1-sparsity)*size();
      boost::mpl::eval_if<
        boost::is_same<
          Container_t,
          std::map<index_t, CalcPrecision_t>
        >,
        SetRandom2,
        SetRandom1
      >::type::Set(elem_, low, hi, num_non_zeros, size());
    }

    struct SetGetNullaryMetafunction1 {
      struct type {
        static void set(Container_t *elem, index_t i, CalcPrecision_t value) {
          typename Container_t::iterator it;
          it = elem->find(i);
          if (unlikely(it != elem->end())) {
            if (value != 0) {
              it->second = value;
            }
            else {
              elem->erase(it);
            }
          }
          else {
            if (value != 0) {
              elem->insert(std::make_pair(i, value));
            }
          }
        }
        template<typename IteratorType>
        static void set(Container_t *elem, const IteratorType &it1,
            const IteratorType &it2) {
          for(const IteratorType it=it1; it!=it2; ++it) {  
            elem->operator[](it->first)=it->second;
          }
        }

        static CalcPrecision_t get(Container_t *elem, index_t i) {
          typename Container_t::iterator it;
          it = elem->find(i);
          if (likely(it != elem->end())) {
            return it->second;
          }
          else {
            return CalcPrecision_t(0.0);
          }
        }
      };
    };

    struct SetGetNullaryMetafunction2 {
      struct type {
        static void set(Container_t *elem, index_t i, CalcPrecision_t value) {
          typename Container_t::iterator it;
          for (it = elem->begin(); it != elem->end(); ++it) {
            if (it->first == i) {
              break;
            }
          }
          if (unlikely(it != elem->end())) {
            it->second = value;
          }
          else {
            elem->push_back(std::make_pair(i, value));
            std::sort(elem->begin(), elem->end());
          }
        }
        template<typename IteratorType>
        static void set(Container_t *elem, const IteratorType &it1,
            const IteratorType &it2) {
          for(IteratorType it=it1; it!=it2; ++it) {  
            elem->push_back(std::make_pair(it->first, it->second));
          }
        }

        static CalcPrecision_t get(Container_t *elem, index_t i) {
          typename Container_t::iterator it;
          for (it = elem->begin(); it != elem->end(); ++it) {
            if (it->first == i) {
              break;
            }
          }
          if (likely(it != elem->end())) {
            return it->second;
          }
          else {
            return CalcPrecision_t(0.0);
          }
        }
      };
    };

    void set(index_t i, CalcPrecision_t value) {
      DEBUG_BOUNDS(i, size_);
      boost::mpl::eval_if <
      boost::is_same <
      Container_t,
      std::map<index_t, CalcPrecision_t>
      > ,
      SetGetNullaryMetafunction1,
      SetGetNullaryMetafunction2
      >::type::set(elem_, i, value);
    }

    struct LoadNullaryMetafunction1 {
      struct type {
        template<typename IteratorType>
        static void set(Container_t *elem, const IteratorType &it1,
            const IteratorType &it2) {
          elem->clear();
          for(const IteratorType it=it1; it!=it2; ++it) {  
            elem->operator[](it->first)=it->second;
          }
        }
      };
    };

    struct LoadNullaryMetafunction2 {
      struct type {
        template<typename IteratorType>
        static void Load(Container_t *elem, const IteratorType &it1,
            const IteratorType &it2) {
          elem->clear();
          for(IteratorType it=it1; it!=it2; ++it) {  
            elem->push_back(std::make_pair(it->first, it->second));
          }
        }
      };
    };

    /**
     * Loading a sparse point
     * This method assumes that the indices
     * between it1 and it2 are in ascending order
     */
    template<typename IteratorType>
    void Load(const IteratorType &it1, const IteratorType &it2) {
      boost::mpl::eval_if <
        boost::is_same <
          Container_t,
          std::map<index_t, CalcPrecision_t>
        > ,
        LoadNullaryMetafunction1,
        LoadNullaryMetafunction2
      >::type::Load(elem_, it1, it2);
    }


    CalcPrecision_t get(index_t i) const {
      DEBUG_BOUNDS(i, size_);
      return boost::mpl::eval_if <
             boost::is_same <
             Container_t,
             std::map<index_t, CalcPrecision_t>
             > ,
             SetGetNullaryMetafunction1,
             SetGetNullaryMetafunction2
             >::type::get(elem_, i);
    }

    CalcPrecision_t operator[](const index_t i) const {
      return get(i);
    }

    void SwapValues(SparsePoint *other) {
      std::swap(*elem_, *(other->elem_));
      std::swap(size_, other->size_);
    }
    void Alias(const SparsePoint &other) {
      DEBUG_ASSERT(should_free_ == false);
      elem_  = other.elem_;
      size_ = other.size_;
    }
    const index_t size() const {
      return size_;
    }

    const index_t length() const {
      return size_;
    }

    const Container_t *elem() const {
      return elem_;
    }

    index_t nnz() const {
      return elem_->size();
    }

    virtual Container_t *&elem() {
      return elem_;
    }

    template<typename StreamType>
    void Print(StreamType &stream, std::string delim) const {
      typename  Container_t::iterator it = elem_->begin();
      while (it != elem_->end()) {
        stream << it->first << ":" << it->second << delim;
        ++it;
      }
    }
    // serialization
    template<typename Archive>
    void save(Archive &ar,
              const unsigned int file_version) const {
      ar << boost::serialization::make_nvp("elem_", elem_);
      ar << boost::serialization::make_nvp("size_", size_);
      bool temp=true;
      ar << boost::serialization::make_nvp("should_free_",temp);
    }

    template<typename Archive>
    void load(Archive &ar,
              const unsigned int file_version) {
      delete elem_;
      ar >> boost::serialization::make_nvp("elem_", elem_);
      ar >> boost::serialization::make_nvp("size_", size_);
      ar >> boost::serialization::make_nvp("should_free_", should_free_);
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /**
     *  @brief Linear Algebra
     */
    static void Sum(
      const SparsePoint<CalcPrecisionType> &a,
      const SparsePoint<CalcPrecisionType> &b,
      SparsePoint<CalcPrecisionType> * const result);

    static void Subtract(
      const SparsePoint<CalcPrecisionType> &a,
      const SparsePoint<CalcPrecisionType> &b,
      SparsePoint<CalcPrecisionType> * const result);

    static void DotMultiply(
      const SparsePoint<CalcPrecisionType> &a,
      const SparsePoint<CalcPrecisionType> &b,
      SparsePoint<CalcPrecisionType> * const result);

    static double LengthEuclidean(
      const SparsePoint<CalcPrecisionType> &a);

    static double Dot(
      const SparsePoint<CalcPrecisionType> &a,
      const SparsePoint<CalcPrecisionType> &b);

    template<typename PrecisionType1, typename PrecisionType2>
    static double Dot(
      const SparsePoint<PrecisionType1> &a,
      const MonolithicPoint<PrecisionType2> &b);
    
    template<typename PrecisionType, typename MixedPointArgs>
    static double Dot(const fl::data::SparsePoint<PrecisionType> &x,
                      const fl::data::MixedPoint<MixedPointArgs> &y); 
    
    template<typename CalcPrecisionType1> 
    static long double Dot(
      const SparsePoint<CalcPrecisionType> &a,
      const fl::dense::Matrix<CalcPrecisionType1, false> &W,
      const SparsePoint<CalcPrecisionType> &b);
    
    template<typename CalcPrecisionType1>   
    static long double Dot(
      const SparsePoint<CalcPrecisionType> &a,
      const fl::dense::Matrix<CalcPrecisionType1, true> &W,
      const SparsePoint<CalcPrecisionType> &b);

    template<typename CalcPrecisionType1>
    static long double Dot(
        const SparsePoint<CalcPrecisionType> &a,
        const fl::data::MonolithicPoint<CalcPrecisionType1> &W,
        const SparsePoint<CalcPrecisionType> &b);

    template<typename ResultMatrixType>
    void UpdateSefOuterProd(
        ResultMatrixType *result) const;

    template<typename ResultMatrixType>
    void UpdateSefOuterProd(
        index_t offset,
        ResultMatrixType *result) const;


    template<typename ResultMatrixType, 
             typename PrecisionType>
    void UpdateOuterProd(
        const fl::data::SparsePoint<PrecisionType> &x,
        ResultMatrixType *result) const;
 
    template<typename ResultMatrixType,
             typename MixedPointArgsType>
    void UpdateOuterProd(
        const fl::data::MixedPoint<MixedPointArgsType> &x,
        ResultMatrixType *result) const;

    static CalcPrecisionType DistanceSq(
      const SparsePoint<CalcPrecisionType> &a,
      const SparsePoint<CalcPrecisionType> &b);

    template<int t_pow>
    class RawLMetric {
      public:
        template<typename CalcPrecisionType1>
        RawLMetric(
          const SparsePoint<CalcPrecisionType> &a,
          const SparsePoint<CalcPrecisionType> &b,
          CalcPrecisionType1 *result);

        template<typename CalcPrecisionType1>
        RawLMetric(
          const fl::data::MonolithicPoint<CalcPrecisionType1> &w,
          const SparsePoint<CalcPrecisionType> &a,
          const SparsePoint<CalcPrecisionType> &b,
          CalcPrecisionType1 *result);

    };

    static inline void SelfScale(
      const CalcPrecisionType alpha,
      SparsePoint<CalcPrecisionType> *const a);

    template<fl::la::MemoryAlloc M>
    class Scale {
      public:
        Scale(
          const CalcPrecisionType alpha,
          const SparsePoint<CalcPrecisionType> &x,
          SparsePoint<CalcPrecisionType> * const y);
    };

    static void AddExpert(CalcPrecisionType alpha,
                          const SparsePoint<CalcPrecisionType> &x,
                          SparsePoint<CalcPrecisionType> *y);

    static void AddTo(
      const SparsePoint<CalcPrecisionType> &x,
      SparsePoint<CalcPrecisionType> * const y);

    template<fl::la::MemoryAlloc M>
    class Add {
      public:
        Add(const SparsePoint<CalcPrecisionType> &x,
            const SparsePoint<CalcPrecisionType> &y,
            SparsePoint<CalcPrecisionType> * const z);
    };

    static void SubFrom(
      const SparsePoint<CalcPrecisionType> &x,
      SparsePoint<CalcPrecisionType> * const y);

    template<fl::la::MemoryAlloc M>
    class Sub {
      public:
        Sub(
          const SparsePoint<CalcPrecisionType> &x,
          const SparsePoint<CalcPrecisionType> &y,
          SparsePoint<CalcPrecisionType> * const z);
    };

#ifdef WIN32
	static bool SparseFabs(bool value);
	static double SparseFabs(double value);
	static long double SparseFabs(long double value);
	static float SparseFabs(float value);
	static unsigned char SparseFabs(unsigned char value);
    static unsigned short SparseFabs(unsigned short value);

#else
    static CalcPrecisionType SparseFabs(CalcPrecisionType value);
#endif
  struct InsertElement1 {
    static void Set(std::map<index_t, CalcPrecision_t> *p,
        index_t ind, CalcPrecision_t val) {
      p->insert(std::make_pair(ind, val));
    } 
    
    static void Set(std::map<index_t, CalcPrecision_t> *p,
        const std::pair<index_t, CalcPrecision_t> &val) {
      p->insert(val);
    }
  };

  struct InsertElement2 {
    static void Set(std::vector<std::pair<index_t, CalcPrecision_t> > *p,
        index_t ind, CalcPrecision_t val) {
      p->push_back(std::make_pair(ind, val));
    }
    static void Set(std::vector<std::pair<index_t, CalcPrecision_t> > *p,
        const std::pair<index_t, CalcPrecision_t> &val) {
      p->push_back(val);
    }
  }; 
  private:
    Container_t* elem_;
    index_t size_;
    bool should_free_;
};

template<typename CalcPrecisionType>
void SparsePoint<CalcPrecisionType>::Sum(
  const SparsePoint<CalcPrecisionType> &a,
  const SparsePoint<CalcPrecisionType> &b,
  SparsePoint<CalcPrecisionType> * const result) {
  DEBUG_ASSERT(a.size_ == b.size_);
  typename  Container_t::const_iterator it1 = a.elem()->begin();
  typename  Container_t::const_iterator it2 = b.elem()->begin();
  typename  Container_t::const_iterator end1 = a.elem()->end();
  typename  Container_t::const_iterator end2 = b.elem()->end(); 

  result->Init(a.size_);
  Container_t * const r_elem =result->elem();

  while (likely(it1 != end1 && it2 != end2)) {
    while (it1->first < it2->first) {
      boost::mpl::if_<
        boost::is_same<
          Container_t,
          std::map<index_t, CalcPrecision_t>
        >,
        InsertElement1,
        InsertElement2
      >::type::Set(r_elem, *it1);
      //r_elem->insert(*it1);
      ++it1;
      if unlikely((it1 == end1)) {
        break;
      }
    }
    if (likely(it1 != end1) && it1->first == it2->first) {
      boost::mpl::if_<
        boost::is_same<
          Container_t,
          std::map<index_t, CalcPrecision_t>
        >,
        InsertElement1,
        InsertElement2
      >::type::Set(r_elem, it1->first, it1->second + it2->second);
      // r_elem->insert(std::make_pair(it1->first, it1->second + it2->second));
      ++it1;
      ++it2;
    }
    else {
      boost::mpl::if_<
        boost::is_same<
          Container_t,
          std::map<index_t, CalcPrecision_t>
        >,
        InsertElement1,
        InsertElement2
      >::type::Set(r_elem, *it2); 
      // r_elem->insert(*it2);
      ++it2;
    }
  }
  while (it1 != end1) {
    boost::mpl::if_<
      boost::is_same<
        Container_t,
        std::map<index_t, CalcPrecision_t>
      >,
      InsertElement1,
      InsertElement2
    >::type::Set(r_elem, *it1);    
    // r_elem->insert(*it1);
    ++it1;
  }
  while (it2 != end2) {
    boost::mpl::if_<
      boost::is_same<
        Container_t,
        std::map<index_t, CalcPrecision_t>
      >,
      InsertElement1,
      InsertElement2
    >::type::Set(r_elem, *it2);
    // r_elem->insert(*it2);
    ++it2;
  }
}

template<typename CalcPrecisionType>
void SparsePoint<CalcPrecisionType>::Subtract(
  const SparsePoint<CalcPrecisionType> &a,
  const SparsePoint<CalcPrecisionType> &b,
  SparsePoint<CalcPrecisionType> * const result) {
  DEBUG_ASSERT(a.size_ == b.size_);
  typename Container_t::const_iterator it1 = a.elem()->begin();
  typename Container_t::const_iterator it2 = b.elem()->begin();
  typename Container_t::const_iterator end1 = a.elem()->end();
  typename Container_t::const_iterator end2 = b.elem()->end();
  Container_t * const r_elem =result->elem();

  result->Init(a.size_);
  while (likely(it1 != end1 && it2 != end2)) {
    while (it1->first < it2->first) {
      boost::mpl::if_<
        boost::is_same<
          Container_t,
          std::map<index_t, CalcPrecision_t>
        >,
        InsertElement1,
        InsertElement2
      >::type::Set(r_elem, *it1);

      // r_elem->insert(*it1);
      ++it1;
      if unlikely((it1 == end1)) {
        break;
      }
    }
    if (likely(it1 != end1) && it1->first == it2->first) {
      boost::mpl::if_<
        boost::is_same<
          Container_t,
          std::map<index_t, CalcPrecision_t>
        >,
        InsertElement1,
        InsertElement2
      >::type::Set(r_elem, it1->first, it1->second - it2->second);

      // r_elem->insert(std::make_pair(it1->first, it1->second - it2->second));
      ++it1;
      ++it2;
    }
    else {
      boost::mpl::if_<
        boost::is_same<
          Container_t,
          std::map<index_t, CalcPrecision_t>
        >,
        InsertElement1,
        InsertElement2
      >::type::Set(r_elem, *it2);
//      r_elem->insert(*it2);
      ++it2;
    }
  }
  while (it1 != end1) {
    boost::mpl::if_<
      boost::is_same<
        Container_t,
          std::map<index_t, CalcPrecision_t>
        >,
        InsertElement1,
        InsertElement2
      >::type::Set(r_elem, *it1);

    // r_elem->insert(it1, end1);
    ++it1;
  }
  if (it2 != end2) {
    while (it2 != end2) {
      boost::mpl::if_<
        boost::is_same<
          Container_t,
          std::map<index_t, CalcPrecision_t>
        >,
        InsertElement1,
        InsertElement2
      >::type::Set(r_elem, it2->first, -it2->second);
//      r_elem->insert(std::make_pair(it2->first, -it2->second));
      ++it2;
    }
  }
}

template<typename CalcPrecisionType>
void SparsePoint<CalcPrecisionType>::DotMultiply(
  const SparsePoint<CalcPrecisionType> &a,
  const SparsePoint<CalcPrecisionType> &b,
  SparsePoint<CalcPrecisionType> * const result) {
  DEBUG_ASSERT(a.size_ == b.size_);
  typename Container_t::const_iterator it1 = a.elem()->begin();
  typename Container_t::const_iterator it2 = b.elem()->begin();
  typename Container_t::const_iterator end1 = a.elem()->end();
  typename Container_t::const_iterator end2 = b.elem()->end();
  Container_t * const r_elem =result->elem();

  result->Init(a.size_);
  while (likely(it1 != end1 && it2 != end2)) {
    while (it1->first < it2->first) {
      ++it1;
      if unlikely((it1 == end1)) {
        break;
      }
    }
    if (likely(it1 != end1) && it1->first == it2->first) {
      boost::mpl::if_<
        boost::is_same<
          Container_t,
          std::map<index_t, CalcPrecision_t>
        >,
        InsertElement1,
        InsertElement2
      >::type::Set(r_elem, it1->first, it1->second*it2->second);

     // r_elem->insert(std::make_pair(it1->first, it1->second*it2->second));
    }
    ++it2;
  }
}

/**
 * The L2 norm
 */
template<typename CalcPrecisionType>
double SparsePoint<CalcPrecisionType>::LengthEuclidean(
  const SparsePoint<CalcPrecisionType> &a) {
  double result = 0;
  typename Container_t::const_iterator it1 = a.elem()->begin();
  typename Container_t::const_iterator end1 = a.elem()->end();
  while (it1 != end1) {
    result += static_cast<double>(it1->second) 
      * static_cast<double>(it1->second);
    ++it1;
  }
  return fl::math::Pow<double, 1, 2>(result);
}



template<typename CalcPrecisionType>
double SparsePoint<CalcPrecisionType>::Dot(
  const SparsePoint<CalcPrecisionType> &a,
  const SparsePoint<CalcPrecisionType> &b) {
  DEBUG_ASSERT(a.size_ == b.size_);
  typename Container_t::const_iterator it1 = a.elem()->begin();
  typename Container_t::const_iterator it2 = b.elem()->begin();
  typename Container_t::const_iterator end1 = a.elem()->end();
  typename Container_t::const_iterator end2 = b.elem()->end();

  double result = 0;
  while (likely(it1 != end1 && it2 != end2)) {
    while (it1->first < it2->first) {
      ++it1;
      if unlikely((it1 == end1)) {
        break;
      }
    }
    if (likely(it1 != end1) && it1->first == it2->first) {
      result += static_cast<double>(it1->second) * static_cast<double>(it2->second);
    }
    ++it2;
  }
  return result;
}

template<typename CalcPrecisionType>
template<typename PrecisionType1, typename PrecisionType2>
double SparsePoint<CalcPrecisionType>::Dot(
      const SparsePoint<PrecisionType1> &a,
      const MonolithicPoint<PrecisionType2> &b) {
  DEBUG_ASSERT(a.size_ == b.size_);
  typename Container_t::const_iterator it1 = a.elem()->begin();
  typename Container_t::const_iterator end1 = a.elem()->end();
  double result=0;
  while(it1!=end1) {
    result+=static_cast<double>(it1->second)*static_cast<double>(b[it1->first]);
    ++it1;
  }
  return result;
}
 
template<typename PointType1, typename PointType2>
struct DotOperatorsDense {
  public:
    DotOperatorsDense(const PointType1 &x, const PointType2 &y,
                      index_t *current_index, 
                      typename PointType1::Container_t::const_iterator &it1,
                      double *result) :
        x_(x), y_(y), ind_(current_index), it1_(it1), result_(result) {
    }

    template<typename T>
    void operator()(T) {
      typename PointType1::Container_t::const_iterator end1 = x_.elem()->end();
      while(it1_!=end1) {
        *result_+=static_cast<double>(it1_->second)
          *static_cast<double>(y_.template dense_point<T>()[it1_->first-*ind_]);
        ++it1_;
      }
      *ind_ += y_.template dense_point<T>().size();
    } 

  private:
    const PointType1 &x_;
    const PointType2 &y_;
    index_t *ind_;
    typename PointType1::Container_t::const_iterator &it1_;
    double *result_;
};

template<typename PointType1, typename PointType2>
struct DotOperatorsSparse {
  public:
    DotOperatorsSparse(const PointType1 &x, const PointType2 &y,
                     index_t *current_index, 
                     typename PointType1::Container_t::const_iterator &it1,
                     double *result) :
      x_(x), y_(y), ind_(current_index), it1_(it1), result_(result) {
    }

    template<typename T>
    void operator()(T) {
      typename SparsePoint<T>::Container_t::const_iterator 
        it2 = y_.template sparse_point<T>().elem()->begin();
      typename PointType1::Container_t::const_iterator end1 = 
        x_.elem()->end();
      typename SparsePoint<T>::Container_t::const_iterator 
        end2 = y_.template sparse_point<T>().elem()->end();

      while (likely(it1_ != end1 && it2 != end2)) {
        while (it1_->first-*ind_ < it2->first) {
          ++it1_;
          if unlikely((it1_ == end1)) {
            break;
          }
        }
        if (likely(it1_ != end1) && it1_->first-*ind_ == it2->first) {
          *result_ += static_cast<double>(it1_->second) * static_cast<double>(it2->second);
        }
        ++it2;
      }
      *ind_ += y_.template sparse_point<T>().size();
    } 

  private:
    const PointType1 &x_;
    const PointType2 &y_;
    index_t *ind_;
    typename PointType1::Container_t::const_iterator &it1_;
    double *result_;
};

   
template<typename CalcPrecisionType>
template<typename PrecisionType, typename MixedPointArgs>
double SparsePoint<CalcPrecisionType>::Dot(
    const fl::data::SparsePoint<PrecisionType> &x,
    const fl::data::MixedPoint<MixedPointArgs> &y) {
  typedef fl::data::SparsePoint<PrecisionType> SparsePoint_t;
  typedef fl::data::MixedPoint<MixedPointArgs> MixedPoint_t;
  typedef typename MixedPoint_t::DenseTypes_t  DenseTypes_t;
  typedef typename MixedPoint_t::SparseTypes_t SparseTypes_t;
  double result = 0;
  index_t current_index = 0;
  typename SparsePoint_t::Container_t::const_iterator it1=
    x.elem()->begin();
  boost::mpl::for_each<DenseTypes_t>(
      DotOperatorsDense<SparsePoint_t, MixedPoint_t>(x, y, 
        &current_index, 
        it1,
        &result));
  boost::mpl::for_each<SparseTypes_t>(
      DotOperatorsSparse<SparsePoint_t, MixedPoint_t>(x, y, 
        &current_index, 
        it1,
        &result));
  return result;
}

template<typename CalcPrecisionType>
template<typename CalcPrecisionType1>
long double SparsePoint<CalcPrecisionType>::Dot(
    const SparsePoint<CalcPrecisionType> &a,
    const fl::dense::Matrix<CalcPrecisionType1, false> &W,
    const SparsePoint<CalcPrecisionType> &b) {
  DEBUG_ASSERT(a.size_==W.n_rows());
  DEBUG_ASSERT(b.size()==W.n_cols());
  typename Container_t::const_iterator it1 = a.elem()->begin();
  typename Container_t::const_iterator it2 = b.elem()->begin();
  typename Container_t::const_iterator end1 = a.elem()->end();
  typename Container_t::const_iterator end2 = b.elem()->end();

  long double result = 0;
  fl::dense::Matrix<double>  temp(W.n_cols());
  temp.SetAll(0.0);
  for(;it2!=end2; ++it2) {
    for(; it1!=end1; ++it1) {
      temp[it2->first]+=static_cast<double>(it1->second)*W.get(it1->first, it2->first);
    }
  }
  for(;it2!=end2; ++it2) {
    result+=static_cast<double>(it2->second)*temp[it2->first];
  }
  return result;
}

template<typename CalcPrecisionType>
template<typename CalcPrecisionType1>
long double SparsePoint<CalcPrecisionType>::Dot(
    const SparsePoint<CalcPrecisionType> &a,
    const fl::data::MonolithicPoint<CalcPrecisionType1> &W,
    const SparsePoint<CalcPrecisionType> &b) {
  return SparsePoint<CalcPrecisionType>::Dot(
      a,
      W,
      b); 
}

template<typename CalcPrecisionType>
template<typename CalcPrecisionType1>
long double SparsePoint<CalcPrecisionType>::Dot(
    const SparsePoint<CalcPrecisionType> &a,
    const fl::dense::Matrix<CalcPrecisionType1, true> &W,
    const SparsePoint<CalcPrecisionType> &b) {
  DEBUG_ASSERT(a.size_ == b.size_);
  DEBUG_ASSERT(a.size_ == W.size());
  typename Container_t::const_iterator it1 = a.elem()->begin();
  typename Container_t::const_iterator it2 = b.elem()->begin();
  typename Container_t::const_iterator end1 = a.elem()->end();
  typename Container_t::const_iterator end2 = b.elem()->end();

  long double result = 0;
  while (likely(it1 != end1 && it2 != end2)) {
    while (it1->first < it2->first) {
      ++it1;
      if unlikely((it1 == end1)) {
        break;
      }
    }
    if (likely(it1 != end1) && it1->first == it2->first) {
      result += static_cast<double>(it1->second) 
        * W.get(it1->second)
        * static_cast<double>(it2->second);
    }
    ++it2;
  }
  return result;
}

template<typename CalcPrecisionType>
CalcPrecisionType SparsePoint<CalcPrecisionType>::DistanceSq(
  const SparsePoint<CalcPrecisionType> &a,
  const SparsePoint<CalcPrecisionType> &b) {
  CalcPrecision_t result;
  RawLMetric<2>(a, b, &result);
  return result;
}

////////////////////////////////////////////////////////////////////////
// The following 2 functions exist due to a VC++ issue. All places here
// where fabs has been called on CalcPrecisionType is after a check 
// to see that the CalcPrecisionType is not a bool. Still VC++ does
// not recognize this (essentially VC++ doesn't recognize a constant
// in the if statement which checks that CalcPrecisionType is not
// a bool before calling fabs. These 2 functions easily correct that
// by using a overloaded version for bools.
////////////////////////////////////////////////////////////////////////
#ifndef WIN32

template<typename CalcPrecisionType>
CalcPrecisionType SparsePoint<CalcPrecisionType>::SparseFabs(CalcPrecisionType value) {
	return fabs(value);
}

#else

template<typename CalcPrecisionType>
long double SparsePoint<CalcPrecisionType>::SparseFabs(long double value) {
	return fabs(value);
}

template<typename CalcPrecisionType>
float SparsePoint<CalcPrecisionType>::SparseFabs(float value) {
	return fabs(value);
}

template<typename CalcPrecisionType>
double SparsePoint<CalcPrecisionType>::SparseFabs(double value) {
	return fabs(value);
}

template<typename CalcPrecisionType>
bool SparsePoint<CalcPrecisionType>::SparseFabs(bool value) {
	return value;
}

template<typename CalcPrecisionType>
unsigned char SparsePoint<CalcPrecisionType>::SparseFabs(unsigned char value) {
	return value;
}

template<typename CalcPrecisionType>
unsigned short SparsePoint<CalcPrecisionType>::SparseFabs(unsigned short value) {
	return value;
}

#endif
/// done with SparseFabs ///

template<typename CalcPrecisionType>
template<int t_pow>
template<typename CalcPrecisionType1>
SparsePoint<CalcPrecisionType>::RawLMetric<t_pow>::RawLMetric(
  const SparsePoint<CalcPrecisionType> &a,
  const SparsePoint<CalcPrecisionType> &b,
  CalcPrecisionType1 *result) {
  DEBUG_ASSERT(a.size_ == b.size_);
  typename Container_t::const_iterator it1 = a.elem()->begin();
  typename Container_t::const_iterator it2 = b.elem()->begin();
  typename Container_t::const_iterator end1 = a.elem()->end();
  typename Container_t::const_iterator end2 = b.elem()->end();

  *result = 0;
  while (likely(it1 != end1 && it2 != end2)) {
    while (it1->first < it2->first) {
      if (boost::is_same<CalcPrecisionType, bool>::type::value == false) {
        if (t_pow % 2 == 0) {
          (*result) += fl::math::Pow<CalcPrecisionType, t_pow, 1>(it1->second);
        }
        else {
          (*result) += fl::math::Pow<CalcPrecisionType, t_pow, 1>(SparseFabs(it1->second));
        }
      }
      else {
        *result += it1->second;
      }
      ++it1;
      if unlikely((it1 == end1)) {
        break;
      }
    }

    if (likely(it1 != end1) && it1->first == it2->first) {
      if (boost::is_same<CalcPrecisionType, bool>::type::value == false) {
        CalcPrecisionType diff = it1->second - it2->second;
        if (t_pow % 2 == 1) {
          diff = SparseFabs(diff);
        }
        (*result) += fl::math::Pow<CalcPrecisionType, t_pow, 1>(diff);
      }
      else {
        (*result) += it1->first == it2->second ? 0 : 1;
        // (*result)+=true & (it1->second ^ it2->second);
      }
      ++it1;
      ++it2;
    }
    else {
      if (boost::is_same<CalcPrecisionType, bool>::type::value == false) {
        if (t_pow % 2 == 0) {
          (*result) += fl::math::Pow<CalcPrecisionType, t_pow, 1>(it2->second);
        }
        else {
          (*result) += fl::math::Pow<CalcPrecisionType, t_pow, 1>(SparseFabs(it2->second));
        }
      }
      else {
        (*result) += it2->second;
      }
      ++it2;
    }
  }
  while (it1 != end1) {
    if (t_pow % 2 == 0) {
      (*result) += fl::math::Pow<CalcPrecisionType, t_pow, 1>(it1->second);
    }
    else {
      (*result) += fl::math::Pow<CalcPrecisionType, t_pow, 1>(SparseFabs(it1->second));
    }
    ++it1;
  }
  while (it2 != end2) {
    if (t_pow % 2 == 0) {
      (*result) += fl::math::Pow<CalcPrecisionType, t_pow, 1>(it2->second);
    }
    else {
      (*result) += fl::math::Pow<CalcPrecisionType, t_pow, 1>(SparseFabs(it2->second));
    }
    ++it2;
  }
}

template<typename CalcPrecisionType>
template<int t_pow>
template<typename CalcPrecisionType1>
SparsePoint<CalcPrecisionType>::RawLMetric<t_pow>::RawLMetric(
  const fl::data::MonolithicPoint<CalcPrecisionType1> &w,
  const SparsePoint<CalcPrecisionType> &a,
  const SparsePoint<CalcPrecisionType> &b,
  CalcPrecisionType1 *result) {
  DEBUG_ASSERT(a.size_ == b.size_);
  DEBUG_ASSERT(a.size() == static_cast<index_t>(w.size()));
  typename Container_t::const_iterator it1 = a.elem()->begin();
  typename Container_t::const_iterator it2 = b.elem()->begin();
  typename Container_t::const_iterator end1 = a.elem()->end();
  typename Container_t::const_iterator end2 = b.elem()->end();

  *result = 0;
  while (likely(it1 != end1 && it2 != end2)) {
    while (it1->first < it2->first) {
      if (boost::is_same<CalcPrecisionType, bool>::type::value == false) {
        if (t_pow % 2 == 0) {
          (*result) += w[it1->first] * fl::math::Pow<CalcPrecisionType, t_pow, 1>(it1->second);
        }
        else {
          (*result) += w[it1->first] * fl::math::Pow<CalcPrecisionType, t_pow, 1>(SparseFabs(it1->second));
        }
      }
      else {
        *result += w[it1->first] * it1->second;
      }
      ++it1;
      if unlikely((it1 == end1)) {
        break;
      }
    }

    if (likely(it1 != end1) && it1->first == it2->first) {
      if (boost::is_same<CalcPrecisionType, bool>::type::value == false) {
        CalcPrecisionType diff = it1->second - it2->second;
        if (t_pow % 2 == 1) {
          diff = SparseFabs(diff);
        }
        (*result) += w[it1->first] * fl::math::Pow<CalcPrecisionType, t_pow, 1>(diff);
      }
      else {
        (*result) += w[it1->first] * it1->first == it2->second ? 0 : 1;
        // (*result)+=true & (it1->second ^ it2->second);
      }
      ++it1;
      ++it2;
    }
    else {
      if (boost::is_same<CalcPrecisionType, bool>::type::value == false) {
        if (t_pow % 2 == 0) {
          (*result) += w[it2->first] * fl::math::Pow<CalcPrecisionType, t_pow, 1>(it2->second);
        }
        else {
          (*result) += w[it2->first] * fl::math::Pow<CalcPrecisionType, t_pow, 1>(SparseFabs(it2->second));
        }
      }
      else {
        (*result) += w[it2->first] * it2->second;
      }
      ++it2;
    }
  }
  while (it1 != end1) {
    if (t_pow % 2 == 0) {
      (*result) += w[it1->first] * fl::math::Pow<CalcPrecisionType, t_pow, 1>(it1->second);
    }
    else {
      (*result) += w[it1->first] * fl::math::Pow<CalcPrecisionType, t_pow, 1>(SparseFabs(it1->second));
    }
    ++it1;
  }
  while (it2 != end2) {
    if (t_pow % 2 == 0) {
      (*result) += w[it2->first] * fl::math::Pow<CalcPrecisionType, t_pow, 1>(it2->second);
    }
    else {
      (*result) += w[it2->first] * fl::math::Pow<CalcPrecisionType, t_pow, 1>(SparseFabs(it2->second));
    }
    ++it2;
  }
}

/**
 * (\f$ A \gets a A\f$)
 */
template<typename CalcPrecisionType>
void SparsePoint<CalcPrecisionType>::SelfScale(
  const CalcPrecisionType alpha,
  SparsePoint<CalcPrecisionType> * const a) {
  typename Container_t::iterator it1 = a->elem()->begin();
  typename Container_t::iterator end1 = a->elem()->end();
  while (it1 != end1) {
    it1->second *= alpha;
    ++it1;
  }
}

/**
 * (\f$ B \gets a A\f$)
 */

template<typename CalcPrecisionType>
template<fl::la::MemoryAlloc M>
SparsePoint<CalcPrecisionType>::Scale<M>::Scale(
  const CalcPrecisionType alpha,
  const SparsePoint<CalcPrecisionType> &x,
  SparsePoint<CalcPrecisionType> * const y) {
  if (M == fl::la::Overwrite) {
    y->CopyValues(x);
  } else {
    y->Copy(x);
  }
  SelfScale(alpha, y);
}

/**
 *  (\f$ y \gets  ax + y \f$)
 */
template<typename CalcPrecisionType>
void SparsePoint<CalcPrecisionType>::AddExpert(CalcPrecisionType alpha,
    const SparsePoint<CalcPrecisionType> &x,
    SparsePoint<CalcPrecisionType> *y) {
  SparsePoint<CalcPrecisionType> z;
  z.Copy(x);
  SelfScale(alpha, &z);
  AddTo(z, y);
}

/**
 *  (\f$ y \gets y + x\f$)
 */
template<typename CalcPrecisionType>
void SparsePoint<CalcPrecisionType>::AddTo(
  const SparsePoint<CalcPrecisionType> &x,
  SparsePoint<CalcPrecisionType> * const y) {
  SparsePoint<CalcPrecisionType> z;
  Sum(x, *y, &z);
  y->Copy(z);
}

/**
 * (\f$ z = x + y \f$)
 */
template<typename CalcPrecisionType>
template<fl::la::MemoryAlloc M>
SparsePoint<CalcPrecisionType>::Add<M>::Add(const SparsePoint<CalcPrecisionType> &x,
    const SparsePoint<CalcPrecisionType> &y,
    SparsePoint<CalcPrecisionType> * const z) {
  if (M == fl::la::Overwrite) {
    z->Destruct();
  }
  Sum(x, y, z);
}

/**
 *  (\f$ y = y - x \f$)
 */
template<typename CalcPrecisionType>
void SparsePoint<CalcPrecisionType>::SubFrom(
  const SparsePoint<CalcPrecisionType> &x,
  SparsePoint<CalcPrecisionType> * const y) {
  SparsePoint<CalcPrecisionType> z;
  Subtract(*y, x, &z);
  y->Copy(z);
}

/**
 *  (\f$ z = y - x \f$)
 */
template<typename CalcPrecisionType>
template<fl::la::MemoryAlloc M>
SparsePoint<CalcPrecisionType>::Sub<M>::Sub(
  const SparsePoint<CalcPrecisionType> &x,
  const SparsePoint<CalcPrecisionType> &y,
  SparsePoint<CalcPrecisionType> * const z) {
  if (M == la::Overwrite) {
    z->Destruct();
  }
  Subtract(y, x, z);
}
}
}

#endif
