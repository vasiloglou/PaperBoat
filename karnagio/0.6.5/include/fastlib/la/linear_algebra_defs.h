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
#ifndef FL_LITE_FASTLIB_LA_LINEAR_ALGEBRA_DEFS_H_
#define FL_LITE_FASTLIB_LA_LINEAR_ALGEBRA_DEFS_H_
namespace fl {
namespace la {
/**
 * @brief Initialization of a container
 *        If you want the function to reuse an already initialized
 *        container, use la::Overwrite. Otherwise if you want the
 *        function to initialize the result, use la::Init
 * @code
 * enum MemoryAlloc {Init=0, Overwrite};
 * @endcode
 */
enum MemoryAlloc {Init = 0, Overwrite};
/**
 * @brief Sometimes in linear algebra we need the transpose
 *        of a matrix. It is in general a bad idea to transpose the
 *        matrix. BLAS has as an option Transpose/Non Transpose mode
 *        This enum lets you choose if you want the matrix as it is
 *        or in the transpose mode.
 * @code
 *  enum TransMode {NoTrans=0, Trans};
 * @endcode
 */
enum TransMode {NoTrans = 0, Trans};

}
}
#endif
