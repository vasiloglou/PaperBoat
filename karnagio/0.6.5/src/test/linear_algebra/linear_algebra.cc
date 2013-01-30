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
 * @file uselapack_test.h
 *
 * Tests for LAPACK integration.
 */
#include<stdio.h>
#define BOOST_TEST_MAIN
#include "linear_algebra_test.h"

BOOST_AUTO_TEST_CASE(linear_algebra_test) {

  const bool print_debug_info = true;
  NOTIFY("Testing float LAPACK\n");
  TestVectorDot<float>();
  TestVectorSimpleMath<float>();
  TestMatrixSimpleMath<float>();
  TestMultiply<float>();
  // TestInverse<float>();
  //TestDeterminant<float>();
  //TestQR<float>();
  //TestEigen<float>();
  //TestSchur<float>();
  //TestSVD<float>();
  //TestCholesky<float>();
  //TestSolve<float>();
  //TestLeastSquareFit<float>();
  TestEntrywiseOperations<float>(print_debug_info);
  TestApplyFunc<float>(print_debug_info);

  NOTIFY("Testing double LAPACK\n");
  TestVectorDot<double>();
  TestVectorSimpleMath<double>();
  TestMatrixSimpleMath<double>();
  TestMultiply<double>();
  // TestInverse<double>();
  //TestDeterminant<double>();
  //TestQR<double>();
  //TestEigen<double>();
  //TestSchur<double>();
  //TestSVD<double>();
  //TestCholesky<double>();
  //TestSolve<double>();
  //TestLeastSquareFit<double>();
  TestEntrywiseOperations<double>(print_debug_info);
  TestApplyFunc<double>(print_debug_info);

  NOTIFY("ALL TESTS PASSED!!!!\n");

}
