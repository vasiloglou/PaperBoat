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
#ifndef FASTLIB_TRAITS_FL_TRAITS_H_
#define FASTLIB_TRAITS_FL_TRAITS_H_

#include <vector>
#include <map>
#include "boost/type_traits/is_same.hpp"
namespace fl {

/**
 *  Traits for getting the Precision and other features
 *  from classes, specially from STL containers.
 *  Now inside a class that takes as a template argument
 *  a class you should use it as follows:
 *  template<typename Container>
 *  MyClass {
 *   public:
 *     typedef Container Container_t;
 *     typedef typename fl::TypeInfo<Container_t>::Precision_t Precision_t
 *  };
 */

template<typename T>
class TypeInfo {
  public:
    typedef typename T::Precision_t Precision_t;
};

template<typename T>
class TypeInfo<std::vector<T> > {
  public:
    typedef T Precision_t;
};

template<typename T>
class TypeInfo<T*> {
  public:
    typedef T Precision_t;
};

#define FL_TRAIT_PRECISION(type) typename fl::TypeInfo<type>::Precision_t

/**
* @brief Sometimes we want to Copy a GenMatrix of floats
*        to a GenMatrix of doubles, which is a valid thing to do
*        The following static assertions are trying to prevent the
*        user from doing invalid operations such as copy a
*        GenMatrix of doubles to a GenMatrix of floats, where precision
*        is lost
*        So you can always copy to larger precision
*        And you can always copy int to float double and long double
*
*/
template<typename Precision1, typename Precision2> struct
      You_have_a_precision_conflict;
template<typename Precision> struct
      You_have_a_precision_conflict<Precision, Precision> {};
template<> struct
      You_have_a_precision_conflict<double, float> {};
template<> struct
      You_have_a_precision_conflict<long double, float> {};
template<> struct
      You_have_a_precision_conflict<long double, double> {};
template<> struct
      You_have_a_precision_conflict<float, int> {};
template<> struct
      You_have_a_precision_conflict<double, int> {};
template<> struct
      You_have_a_precision_conflict<long double, int> {};


}
#endif
