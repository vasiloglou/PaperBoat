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

#ifndef FASTLIB_DENSE_LINEAR_ALGEBRA_AUX_H_
#define FASTLIB_DENSE_LINEAR_ALGEBRA_AUX_H_
#include "fastlib/la/linear_algebra_defs.h"
namespace fl {
namespace dense {
template<fl::la::MemoryAlloc T>
class AllocationTrait {
  public:
    template<typename Container>
    static inline void Init(index_t length, Container *cont);

    static inline void Init(index_t length, float **cont);

    static inline void Init(index_t length, double **cont);

    static inline void Init(index_t length, long double **cont);

    template<typename Container>
    static inline void Init(index_t dim1, index_t dim2, Container *cont);
};

template<>
class AllocationTrait<fl::la::Init> {
  public:
    template<typename Container>
    static inline void Init(index_t length, Container *cont) {
      cont->Init(length);
    }

    static inline void Init(index_t length, float **cont) {
      *cont =  new float[length];
    }

    static inline void Init(index_t length, double **cont) {
      *cont =  new double[length];
    }

    static inline void Init(index_t length, long double **cont) {
      *cont =  new long double[length];
    }

    template<typename Container>
    static inline void Init(index_t dim1, index_t dim2, Container *cont) {
      cont->Init(dim1, dim2);
    }
};

template<>
class AllocationTrait<fl::la::Overwrite> {
  public:
    template<typename Container>
    static inline void Init(index_t length, Container *cont) {
      BOOST_ASSERT(length == cont->length());
    }

    static inline void Init(index_t length, float **cont) {
      DEBUG_ASSERT_MSG(*cont != NULL, "Uninitialized pointer");
    }

    static inline void Init(index_t length, double  **cont) {
      DEBUG_ASSERT_MSG(*cont != NULL, "Uninitialized pointer");
    }

    static inline void Init(index_t length, long double **cont) {
      DEBUG_ASSERT_MSG(*cont != NULL, "Uninitialized pointer");
    }

    template<typename Container>
    static inline void Init(index_t dim1, index_t dim2, Container *cont) {
      DEBUG_SAME_SIZE(dim1, cont->n_rows());
      DEBUG_SAME_SIZE(dim2, cont->n_cols());
    }
};
} //dense namespace
} //fl namespace

#endif
