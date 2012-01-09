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
#ifndef FL_LITE_FASTLIB_DATA_MULTIDATASET_ITERATORS_H_
#define FL_LITE_FASTLIB_DATA_MULTIDATASET_ITERATORS_H_


// Iterators are critical for linear access of the data
template < typename TypeList,
template <typename> class PointType,
template <typename> class ContainerType >
class Iterators {
  public:
    template<typename T>
    struct wrap {
      typedef  typename ContainerType<PointType<T> >::iterator Iterator_t;
      Iterator_t value;
    };
    class  Generated :  public
          boost::mpl::inherit_linearly < TypeList,
          boost::mpl::inherit < wrap<boost::mpl::placeholders::_2>,
          boost::mpl::placeholders::_1 > >::type {
      public:
        template<typename T>
        class  wrap {
          public:
            typedef typename ContainerType<PointType<T> >::iterator Iterator_t;
        };

        template<typename T>
        static typename wrap<T>::Iterator_t& get(Generated &iterators) {
          return static_cast<typename Iterators::template wrap<T> &>(iterators).value;
        }

        template<typename T>
        typename wrap<T>::Iterator_t& get() {
          return static_cast<typename Iterators::template wrap<T> &>(*this).value;

        }
    };
};
#endif
