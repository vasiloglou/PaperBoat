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
#define BOOST_TEST_MAIN
#include "boost/test/unit_test.hpp"
#include "fastlib/math/fl_math.h"

BOOST_AUTO_TEST_SUITE(math)

BOOST_AUTO_TEST_CASE(TestPermutation) {
  std::vector<index_t> p_rand;
  std::vector<index_t> p_idnt;
  std::vector<int> visited;
  int n = 3111;

  fl::math::MakeRandomPermutation(n, &p_rand);
  fl::math::MakeIdentityPermutation(n, &p_idnt);

  for (int i = 0; i < n; i++) {
    BOOST_CHECK(p_idnt[i] == i);
  }

  visited.resize(n);

  for (int i = 0; i < n; i++) {
    visited[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    visited[p_rand[i]]++;
  }

  for (int i = 0; i < n; i++) {
    BOOST_CHECK(visited[i] == 1);
  }
}

BOOST_AUTO_TEST_CASE(TestFactorial) {
  BOOST_CHECK(fl::math::Factorial<double>(0) == 1.0);
  BOOST_CHECK(fl::math::Factorial<double>(1) == 1.0);
  BOOST_CHECK(fl::math::Factorial<double>(2) == 2.0);
  BOOST_CHECK(fl::math::Factorial<double>(3) == 6.0);
  BOOST_CHECK(fl::math::Factorial<double>(4) == 24.0);
  BOOST_CHECK(fl::math::Factorial<double>(5) == 120.0);
  BOOST_CHECK(fl::math::Factorial<double>(6) == 720.0);
}

BOOST_AUTO_TEST_CASE(TestSphereVol) {
  BOOST_CHECK(fabs(fl::math::SphereVolume(2.0, 1) - 2.0 * 2.0) < 1.0e-7);
  BOOST_CHECK(fabs(fl::math::SphereVolume(2.0, 2) - fl::math::Const<double>::PI * 4.0) < 1.0e-7);
  BOOST_CHECK(fabs(fl::math::SphereVolume(2.0, 3) - 4.0 / 3.0 * fl::math::Const<double>::PI * 8.0) < 1.0e-7);
  BOOST_CHECK(fabs(fl::math::SphereVolume(3.0, 3) - 4.0 / 3.0 * fl::math::Const<double>::PI * 27.0) < 1.0e-7);
}

BOOST_AUTO_TEST_CASE(TestKernel) {
  fl::math::GaussianKernel<double> k;

  k.Init(2.0);

  BOOST_CHECK(k.EvalUnnormOnSq(0) == 1.0);
  BOOST_CHECK(k.EvalUnnorm(0) == 1.0);

  BOOST_CHECK(fabs(k.EvalUnnormOnSq(1) - exp(-0.125)) < 1.0e-7);

  BOOST_CHECK((k.CalcNormConstant(1) - sqrt(4 * 2 * fl::math::Const<double>::PI)) < 1.0e-7);

  fl::math::EpanKernel<double> k2;
  k2.Init(2);
  BOOST_CHECK(k2.EvalUnnormOnSq(0) == 1.0);
  BOOST_CHECK(fabs(k2.EvalUnnormOnSq(1) - 0.75) < 1.0e-7);
  BOOST_CHECK(fabs(k2.CalcNormConstant(1) - 4.0 * 2.0 / 3.0) < 1.0e-7);
  // TODO: Test the constant factor
}

BOOST_AUTO_TEST_CASE(TestMisc) {
  BOOST_CHECK(fl::math::Sqr(3.0) == 9.0);
  BOOST_CHECK(fl::math::Sqr(-3.0) == 9.0);
  BOOST_CHECK(fl::math::Sqr(0.0) == 0.0);
  BOOST_CHECK(fl::math::Sqr(0.0) == 0.0);
  BOOST_CHECK(fl::math::ClampNonNegative(-1.0) == 0.0);
  BOOST_CHECK(fl::math::ClampNonNegative(0.0) == 0.0);
  BOOST_CHECK(fl::math::ClampNonNegative(2.25) == 2.25);
  BOOST_CHECK(fl::math::ClampNonPositive(-1.25) == -1.25);
  BOOST_CHECK(fl::math::ClampNonPositive(0.0) == 0.0);
  BOOST_CHECK(fl::math::ClampNonPositive(2.25) == 0.0);
}

BOOST_AUTO_TEST_CASE(TestPow) {
  BOOST_CHECK_EQUAL((fl::math::Pow<double, 2, 2>(3.0)), 3.0);
  BOOST_CHECK_EQUAL((fl::math::Pow<double, 1, 1>(3.0)), 3.0);
  BOOST_CHECK_EQUAL((fl::math::Pow<double, 2, 1>(3.0)), 9.0);
  BOOST_CHECK_EQUAL((fl::math::Pow<double, 1, 2>(9.0)), 3.0);
  BOOST_CHECK_EQUAL((fl::math::Pow<double, 3, 3>(9.0)), 9.0);
  BOOST_CHECK_CLOSE((fl::math::Pow<double, 1, 3>(8.0)), 2.0, 1.0e-6);
  BOOST_CHECK_CLOSE((fl::math::PowAbs<double, 1, 3>(-8.0)), 2.0, 1.0e-6);
  BOOST_CHECK_EQUAL((fl::math::PowAbs<double, 1, 1>(-8.0)), 8.0);
  BOOST_CHECK_EQUAL((fl::math::PowAbs<double, 2, 1>(-8.0)), 64.0);
}

BOOST_AUTO_TEST_SUITE_END()

