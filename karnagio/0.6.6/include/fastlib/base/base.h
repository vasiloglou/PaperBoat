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
/**
 * @file base.h
 *
 *
 * @see common.h
 * @see debug.h
 * @see compiler.h
 * @see cc.h
 * @see otrav.h
 */

#ifndef FL_LITE_FASTLIB_BASE_H
#define FL_LITE_FASTLIB_BASE_H

//#include "common.h"
//#include "debug.h"
#include "boost/assert.hpp"
namespace boost
{
   void assertion_failed(char const * expr, 
       char const * function, 
       char const * file, 
       long line);
}

#define DEBUG_ASSERT BOOST_ASSERT

#ifdef DEBUG
#define DEBUG_ASSERT_MSG(expr, message) \
{ if (!(expr)) ::boost::assertion_failed(#expr, message, __FILE__, __LINE__); } 
#else
#define DEBUG_ASSERT_MSG(expr, message)
#endif

#ifdef DEBUG
#define DEBUG_SAME_SIZE(x, y) \
  DEBUG_ASSERT_MSG((x) == (y), #x " is not equal to "#y)
#define DEBUG_BOUNDS(x, bounds) \
 DEBUG_ASSERT_MSG((x<bounds) && (x>=0), #x " is out of bounds")
#else
#define DEBUG_SAME_SIZE(x, y) 
#define DEBUG_BOUNDS(x, bounds) 
#endif

typedef enum {
  /** Upper-bound value indicating failed operation. */
  SUCCESS_FAIL = 31,
  /** A generic warning value. */
  SUCCESS_WARN = 48,
  /** Lower-bound value indicating successful operation. */
  SUCCESS_PASS = 96
} success_t;

#define PASSED(x) (x >= SUCCESS_PASS)

#include "logger.h"
#include "randomizer.h"
#include "basic_types.h"
#define  index_t int64
#include "constant_strings.h"

#endif /* BASE_H */
