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
#ifndef FL_LITE_FASTLIB_BASE_MPL_H_
#define FL_LITE_FASTLIB_BASE_MPL_H_
#include "boost/mpl/aux_/na.hpp"
#include "boost/mpl/for_each.hpp"
#include "boost/mpl/inherit.hpp"
#include "boost/mpl/inherit_linearly.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/mpl/placeholders.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/size.hpp"
#include "boost/mpl/pair.hpp"
#include "boost/mpl/int.hpp"
#include "boost/mpl/transform.hpp"
#include "boost/mpl/assert.hpp"
#include "boost/mpl/range_c.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/serialization/nvp.hpp"
#include "boost/serialization/split_member.hpp"

namespace fl {
/**
 * @brief Many times we need to put a template inside an mpl container such as mpl::map
 *        or mpl::vector. This wrapper enables us to do so. For example
 *        template<typename T>
 *        class Foo;
 *
 *        boost::mpl::vector2<int, Foo> is not correct Foo is not a type
 *        but
 *        boost::mpl::vector2<int, WrapTemplate1<Foo> > is ok
 *
 *        Notice that our current implementation we support up to 3 template
 *        arguments. It is possible that with the boost preprocessor mpl it
 *        will be eaiser to automate that
 *
 *        Now if you want to acceess the teplate inside WrapTemplate you have
 *        to do the this:
 *        typedef WrapTemplate<Foo> FooRapped;
 *        FooRapped::type<int> a;
 *        Notice FooRapped::type<int> has exactly the same functionality as Foo<int>
 *        but they are not the same types, since FooRapped::type<int> inherits from
 *        Foo<int>.
 *        If you want to get the exact same type you should use
 *        FooRapped::type<int>::original which is exactly the same type as Foo<int>
 *
 */
template<template <typename T1, typename T2, typename T3>  class U >
struct WrapTemplate3 {
  template<typename T1, typename T2, typename T3>
  class type : public U<T1, T2, T3> {
      typedef U<T1, T2, T3> original;
  };
};
template<template <typename T1, typename T2>  class U >
struct WrapTemplate2 {
  template<typename T1, typename T2>
  class type : public U<T1, T2> {
      typedef U<T1, T2> original;
  };
};
template<template <typename T1>  class U>
struct WrapTemplate1 {
  template<typename T1>
  class type : public U<T1> {
      typedef U<T1> original;
  };
};

/**
 * @brief We need this struct so that we can easily use boost::mpl::eval_if
 *        eval_if expects a NullaryMetafunction which is nothing else than a struct
 *        that define a class  called type inside it
 */
template<typename T>
struct EmbedInType {
  typedef T type;
};

/**
 * @brief we use this to wrap a pointer to a class (which is a valid template argument)
 *        inside a container
 */
template<typename T, T* object>
struct WrapPointer {
  typedef T type;
  static const T* obj;
};

template<typename T, T* object>
const T* WrapPointer<T, object>::obj = object;

/**
 *  @brief this one generates a struct with members all the types in the class
 */
template<typename TypeList>
struct MakeTypeIndexedStruct {
  template<typename T>
  struct wrap {
    T value;
  };
  class  Generated :  public
        boost::mpl::inherit_linearly <
        TypeList,
        boost::mpl::inherit <
        wrap<boost::mpl::placeholders::_2>,
        boost::mpl::placeholders::_1
        >
        >::type {
    public:
      template<typename T>
      static  T &get(Generated &box) {
        return static_cast<wrap<T> &>(box).value;
      }
      template<typename T>
      T &get() {
        return get<T>(*this);
      }
  };
};
/**
 *  @brief This template generates a struct that can have the same type
 *         more than once. The TypeList must be an mpl::vector
 *         where the key is an enum (int) and the value is the type
 */

template<typename TypeList>
struct MakeIntIndexedStruct {
  template<typename T>
  struct wrap {
    typename T::first value;
  };
  static const int size = boost::mpl::size<TypeList>::value;
  typedef typename boost::mpl::transform <
  TypeList,
  boost::mpl::range_c<int, 0, size>,
  boost::mpl::pair <
  boost::mpl::placeholders::_1,
  boost::mpl::placeholders::_2
  >
  >::type TypeListAux_t;

  class  Generated :  public
        boost::mpl::inherit_linearly < TypeListAux_t,
        boost::mpl::inherit < wrap<boost::mpl::placeholders::_2>,
        boost::mpl::placeholders::_1 > >::type {
    public:
      typedef TypeList TypeList_t;
      // BOOST_MPL_ASSERT((boost::is_same<TypeList_t, char>));
      static const int size = boost::mpl::size<TypeList>::value;

      template<int field>
      static typename boost::mpl::at < TypeList,
      boost::mpl::int_<field> >::type &
      get(Generated &box) {


        typedef typename boost::mpl::at_c <
        TypeListAux_t,
        field
        >::type type_aux;
        return static_cast<wrap<type_aux> &>(box).value;
      }

      template<int field>
      static const typename boost::mpl::at < TypeList,
      boost::mpl::int_<field> >::type &
      get(const Generated &box) {


        typedef typename boost::mpl::at_c <
        TypeListAux_t,
        field
        >::type type_aux;
        return static_cast<const wrap<type_aux> &>(box).value;
      }

      template<int field>
      typename boost::mpl::at<TypeList, boost::mpl::int_<field> >::type &get() {
        return get<field>(*this);
      }

      template<int field>
      const typename boost::mpl::at<TypeList, boost::mpl::int_<field> >::type &get() const {
        return get<field>(*this);
      }

      Generated &operator=(const Generated &other) {
        boost::mpl::for_each<boost::mpl::range_c<int, 0, size> >(
          AssignOperator(this, &other));
        return *this;

      }
      Generated &Copy(const Generated &other) {
        boost::mpl::for_each<boost::mpl::range_c<int, 0, size> >(
          AssignOperator(this, &other));
        return *this;
      }
      // serialization
      friend class boost::serialization::access;
      template<typename Archive>
      void save(Archive &ar,
                const unsigned int file_version) const {
        boost::mpl::for_each<boost::mpl::range_c<int, 0, size> > (
            SerializationOperator<Archive>(this, ar));
      }

      template<typename Archive>
      void load(Archive &ar,
                const unsigned int file_version) {
         boost::mpl::for_each<boost::mpl::range_c<int, 0, size> >(
            DeSerializationOperator<Archive>(this, ar));
      }

      BOOST_SERIALIZATION_SPLIT_MEMBER()
  };
  template<typename Archive>
  struct SerializationOperator {
      SerializationOperator(const Generated *p, Archive &ar) :
        p_(p), ar_(ar) {};
      template<typename T>
      void operator()(T) {
        ar_ << boost::serialization::make_nvp(
            boost::lexical_cast<std::string>(T::type::value).c_str(), 
              p_->template get<T::type::value>());
      }  
    private:
      const Generated *p_;
      Archive &ar_;
  };

  template<typename Archive>
  struct DeSerializationOperator {
      DeSerializationOperator(Generated *p, Archive &ar) :
        p_(p), ar_(ar) {};
      template<typename T>
      void operator()(T) {
        ar_ >> boost::serialization::make_nvp(
            boost::lexical_cast<std::string>(T::type::value).c_str(), 
              p_->template get<T::type::value>());
      }
    
    private:
      Generated *p_;
      Archive &ar_;
  
  };

  struct AssignOperator {
    AssignOperator(Generated *p1, const Generated *p2):
        p1_(p1), p2_(p2) {
    }
    template<typename T>
    void operator()(T) {
      p1_->template get<T::type::value>() = p2_->template get<T::type::value>();
    }
    private:
      Generated *p1_;
      const Generated *p2_;
  };

};

}
#endif
