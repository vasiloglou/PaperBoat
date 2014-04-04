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

#ifndef FL_LITE_FASTLIB_DATA_MIXED_ITERATOR_H_
#define FL_LITE_FASTLIB_DATA_MIXED_ITERATOR_H_
  struct NullaryIterator1 {
    template<typename PointType>
    static typename MixedPoint<ParamList>::iterator begin(const PointType *p) {
      return  p->template sparse_point<typename boost::mpl::front<typename MixedPoint<ParamList>::SparseTypes_t>::type >().begin();  
    }
    template<typename PointType>
    static typename MixedPoint<ParamList>::iterator end(const PointType *p) {
      return p->template sparse_point<typename boost::mpl::front<typename MixedPoint<ParamList>::SparseTypes_t>::type >().end();  
    }
  };

  struct NullaryIterator2 {
    template<typename PointType>
    static typename MixedPoint<ParamList>::iterator begin(const PointType *p) {
      return p->template dense_point<typename boost::mpl::front<typename MixedPoint<ParamList>::DenseTypes_t>::type >().begin();  
    }
    template<typename PointType>
    static typename MixedPoint<ParamList>::iterator end(const PointType *p) {
      return p->template dense_point<typename boost::mpl::front<typename MixedPoint<ParamList>::DenseTypes_t>::type >().end();  
    }
  };

  struct NullaryIterator3 {
    template<typename PointType>
    static typename MixedPoint<ParamList>::MixedIterator begin(PointType *p) {
      fl::logger->Die() << "mixed iterators " << fl::NOT_SUPPORTED_MESSAGE; 
      return typename MixedPoint<ParamList>::MixedIterator();
    }
    template<typename PointType>
    static typename MixedPoint<ParamList>::MixedIterator end(PointType *p) {
      fl::logger->Die() << "mixed iterators " << fl::NOT_SUPPORTED_MESSAGE; 
      return typename MixedPoint<ParamList>::MixedIterator();
    }
  };

#endif
