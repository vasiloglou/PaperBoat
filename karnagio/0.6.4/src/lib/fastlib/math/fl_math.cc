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
#include "fastlib/math/fl_math.h"

namespace fl {
namespace math {

/**
 * Creates an identity permutation where the element i equals i.
 *
 * Low-level pointer version -- preferably use the @c ArrayList
 * version instead.
 *
 * For instance, result[0] == 0, result[1] == 1, result[2] == 2, etc.
 *
 * @param size the number of elements in the permutation
 * @param array a place to store the permutation
 */
void MakeIdentityPermutation(index_t size, index_t *array) {
  for (index_t i = 0; i < size; i++) {
    array[i] = i;
  }
}

/**
 * Inverts or transposes an existing permutation.
 */
void MakeInversePermutation(index_t size,
                            const index_t *original, index_t *reverse) {
  for (index_t i = 0; i < size; i++) {
    reverse[original[i]] = i;
  }
}

/**
 * Creates a random permutation and stores it in an existing C array
 * (power user version).
 *
 * The random permutation is over the integers 0 through size - 1.
 *
 * @param size the number of elements
 * @param array the array to store a permutation in
 */
void MakeRandomPermutation(index_t size, index_t *array) {
  // Regular permutation algorithm.
  // This is cache inefficient for large sizes; large caches might
  // warrant a more sophisticated blocked algorithm.

  if (size == 0) {
    return;
  }

  array[0] = 0;

  for (index_t i = 1; i < size; i++) {
    index_t victim = rand() % i;
    array[i] = array[victim];
    array[victim] = i;
  }
}
};
};
