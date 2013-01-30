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
#ifndef FL_LITE_FASTLIB_DATA_MULTIDATASET_PUSH_BACK_OPERATOR_H_
#define FL_LITE_FASTLIB_DATA_MULTIDATASET_PUSH_BACK_OPERATOR_H_
/**
 * @brief The following struncts are used for pushing points in the dataset
 */
template<typename BoxType, typename Iterators>
struct PushBackOperator {
  struct DenseCase {
    template<typename T>
    static  MonolithicPoint<T> &get(Point_t *point); 

    struct Get1 {
      template<typename T>
      static  MonolithicPoint<T> &get(Point_t *point); 
    };

    struct Get2 {
      template<typename T>
      static  MonolithicPoint<T> &get(MonolithicPoint<T> *point); 
    };
  };

  struct SparseCase {
    template<typename T>
    static  SparsePoint<T> &get(Point_t *point); 

    struct Get1 {
      template<typename T>
      static  SparsePoint<T> &get(Point_t *point); 
    };

    struct Get2 {
      template<typename T>
      static  SparsePoint<T> &get(SparsePoint<T> *point); 
    };
  };

  PushBackOperator(BoxType *box, Iterators *its, Point_t *point);

  template<typename T>
  void operator()(T);

private:
  BoxType *box_;
  Iterators *its_;
  Point_t *point_;
};

struct SetMetaDataOperator1 {
  struct type {
    template<typename MetaDataBoxType, typename PointType>
    static void Set(PointType &point, MetaDataBoxType *box); 
  };
};

struct SetMetaDataOperator2 {
  struct type {
    template<typename MetaDataBoxType, typename PointType>
    static void Set(PointType &point, MetaDataBoxType *box); 
  };
};

#endif
