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
#ifndef FL_LITE_FASTLIB_DATA_MULTIDATASET_ALIAS_OPERATORS_H_
#define FL_LITE_FASTLIB_DATA_MULTIDATASET_ALIAS_OPERATORS_H_
/**
 * @brief the following structs are used in conjuction with for_each so that
 *      we can do alias between points. It sounds wierd since all the points
 *      implement Alias from another point. Keep in mind that here the
 *      different types are kept in different containers, so the Alias
 *      has to be done differently. This is what the following classed do
 *      with the appropriate mpl tricks
 */

struct MetaDataAlias1 {
  struct type {
    template<typename MetaDataType>
    static inline void Alias(MetaDataType *value, Point_t *entry);
  };
};

struct MetaDataAlias2 {
  struct type {
    template<typename MetaDataType>
    static inline void Alias(MetaDataType *value, Point_t *entry);
  };
};

struct DenseAlias {
  DenseAlias(DenseBox *box, index_t ind, Point_t *entry);

  struct MonolithicPointOperator {
    template<typename T>
    static MonolithicPoint<T> &Do(MonolithicPoint<T> *point); 
  };

  struct MixedPointOperator {
    template<typename T>
    static MonolithicPoint<T> &Do(Point_t *point);
  };

  struct CompactOperator {
    template<typename T>
    static void Do(DenseBox *box, index_t i, MonolithicPoint<T> &point);
  };

  struct ExtendableOperator {
    template<typename T>
    static void Do(DenseBox *box, index_t i, MonolithicPoint<T>  &point);
  };

  template<typename T>
  void operator()(T); 

private:
  DenseBox *box_;
  index_t ind_;
  Point_t *entry_;

};


// we need this metafunctions to set the size for points. For sparse and dense
// we do nothing but for mixed we have to
struct NullaryMetaFunctionSetSize1 {
  struct type {
    template<typename PointType>
    static void set_size(PointType *p, index_t size);
  };
};

struct NullaryMetaFunctionSetSize2 {
  struct type {
    template<typename PointType>
    static void set_size(PointType *p, index_t size);
  };
};

struct SparseAlias {
  SparseAlias(SparseBox *box, index_t ind, Point_t *entry);
  
  struct SparsePointOperator {
    template<typename T>
    static Point_t &Do(Point_t *point);
  };

  struct MixedPointOperator {
    template<typename T>
    static SparsePoint<T> &Do(Point_t *point);   
  };

  template<typename T>
  void operator()(T); 

private:
  SparseBox *box_;
  index_t ind_;
  Point_t *entry_;
};



#endif
