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
// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * @file fortran.h
 *
 * Basic types for FORTRAN compatability.
 */

#ifndef FL_LITE_FASTLIB_BASE_FORTRAN_H
#define FL_LITE_FASTLIB_BASE_FORTRAN_H

/* These typedefs should work for all machines I'm aware of. */

/** FORTRAN integer type. */
typedef int f77_integer;
/** FORTRAN Boolean type with values F77_TRUE and F77_FALSE. */
typedef unsigned int f77_logical;
/** FORTRAN single-precision type (e.g. float). */
typedef float f77_real;
/** FORTRAN double-precision type (e.g. double). */
typedef double f77_double;

/**
 * FORTRAN void return value.
 *
 * FORTRAN subroutines will still be prototyped to return an int, but this
 * integer must be ignored.
 */
typedef int f77_ret_void;
/** FORTRAN integer return value. */
typedef f77_integer f77_ret_integer;
/** FORTRAN Boolean return value. */
typedef f77_logical f77_ret_logical;
/**
 * FORTRAN single-precision return value.
 *
 * Note that FORTRAN seems to return doubles even for single-precision
 * functions.
 */
typedef f77_real f77_ret_real;
/** FORTRAN double-precision return value. */
typedef f77_double f77_ret_double;

/** FORTRAN single-precision complex number. */
typedef struct {
  f77_real re;
  f77_real im;
} f77_complex;
/** FORTRAN double-precision complex number. */
typedef struct {
  f77_double re;
  f77_double im;
} f77_doublecomplex;

/** Length of a FORTRAN string. */
typedef long f77_str_len;

/** False value for f77_logical type. */
#define F77_FALSE ((f77_logical)0)
/** True value for f77_logical type. */
#define F77_TRUE (~F77_FALSE)

/**
 * Does name-mangling for FORTRAN functions.
 *
 * Example:
 * @code
 *   F77_FUNC(fname)(a, b, c, d);
 * @endcode
 * translates to:
 * @code
 *   fname_(a, b, c, d);
 * @endcode
 */
#define F77_FUNC(fname) fname ## _

#ifdef __cplusplus
#define F77_UNKNOWN_ARGS ...
#else
#define F77_UNKNOWN_ARGS
#endif

/** FORTRAN function-pointer for integers. */
typedef f77_ret_integer(*f77_integer_func)(F77_UNKNOWN_ARGS);
/** FORTRAN function-pointer for Booleans. */
typedef f77_ret_logical(*f77_logical_func)(F77_UNKNOWN_ARGS);
/** FORTRAN function-pointer for single-precision. */
typedef f77_ret_real(*f77_real_func)(F77_UNKNOWN_ARGS);
/** FORTRAN function-pointer for double-precision. */
typedef f77_ret_double(*f77_double_func)(F77_UNKNOWN_ARGS);

#endif /* BASE_FORTRAN_H */
