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

// for BOOST testing
#define BOOST_TEST_MAIN

#include "boost/test/unit_test.hpp"
#include "fastlib/optimization/lbfgs/lbfgs_dev.h"

namespace fl {
namespace ml {
namespace lbfgs_test {

class ExtendedRosenbrockFunction {

  private:

    int num_dimensions_;

  public:
    double Evaluate(const fl::data::MonolithicPoint<double> &x) {
      double fval = 0;
      for (int i = 0; i < num_dimensions() - 1; i++) {
        fval = fval + 100 * fl::math::Sqr(x[i] * x[i] - x[i + 1]) +
               fl::math::Sqr(x[i] - 1);
      }
      return fval;
    }

    void Gradient(const fl::data::MonolithicPoint<double> &x,
                  fl::data::MonolithicPoint<double> *gradient) {

      gradient->SetZero();
      for (int k = 0; k < num_dimensions() - 1; k++) {
        (*gradient)[k] = 400 * x[k] * (x[k] * x[k] - x[k+1]) + 2 * (x[k] - 1);
        if (k > 0) {
          (*gradient)[k] = (*gradient)[k] + 200 * (x[k] - x[k - 1] * x[k - 1]);
        }
      }
      (*gradient)[num_dimensions() - 1] =
        200 * (x[num_dimensions() - 1] -
               fl::math::Sqr(x[num_dimensions() - 2]));
    }

    int num_dimensions() const {
      return num_dimensions_;
    }

    void InitStartingIterate(fl::data::MonolithicPoint<double> *iterate) {
      num_dimensions_ = 4 * fl::math::Random(2, 200);
      iterate->Init(num_dimensions_);
      for (int i = 0; i < num_dimensions_; i++) {
        if (i % 2 == 0) {
          (*iterate)[i] = -1.2;
        }
        else {
          (*iterate)[i] = 1.0;
        }
      }
    }

};

class WoodFunction {

  public:
    double Evaluate(const fl::data::MonolithicPoint<double> &x) {
      return 100 * fl::math::Sqr(x[0]* x[0] - x[1]) +
             fl::math::Sqr(1 - x[0]) +
             90*fl::math::Sqr(x[2] * x[2] - x[3]) + fl::math::Sqr(1 - x[2]) +
             10.1*(fl::math::Sqr(1 - x[1]) + fl::math::Sqr(1 - x[3])) +
             19.8*(1 - x[1])*(1 - x[3]);
    }

    void Gradient(const fl::data::MonolithicPoint<double> &x,
                  fl::data::MonolithicPoint<double> *gradient) {
      (*gradient)[0] = 400 * x[0] * (x[0] * x[0] - x[1]) + 2 * (x[0] - 1);
      (*gradient)[1] = 200 * (x[1] - x[0] * x[0]) + 20.2 * (x[1] - 1) +
                       19.8 * (x[3] - 1);
      (*gradient)[2] = 360 * x[2] * (x[2] * x[2] - x[3]) + 2 * (x[2] - 1);
      (*gradient)[3] = 180 * (x[3] - x[2] * x[2]) + 20.2 * (x[3] - 1) +
                       19.8 * (x[1] - 1);
    }

    int num_dimensions() const {
      return 4;
    }

    void InitStartingIterate(fl::data::MonolithicPoint<double> *iterate) {

      iterate->Init(num_dimensions());
      (*iterate)[0] = (*iterate)[2] = -3;
      (*iterate)[1] = (*iterate)[3] = -1;
    }

};

class LbfgsTest {
  public:

    void TestExtendedRosenbrockFunction() {

      fl::logger->Message() << "Testing extended Rosenbrock function: " <<
      "optimal value: 0";
      for (int i = 0; i < 10; i++) {
        fl::ml::lbfgs_test::ExtendedRosenbrockFunction
        extended_rosenbrock_function;
        fl::ml::Lbfgs < fl::ml::lbfgs_test::ExtendedRosenbrockFunction >
        extended_rosenbrock_function_lbfgs;
        fl::data::MonolithicPoint<double>
        extended_rosenbrock_function_optimized;
        extended_rosenbrock_function.InitStartingIterate(
          &extended_rosenbrock_function_optimized);
        extended_rosenbrock_function_lbfgs.Init(
          extended_rosenbrock_function,
          std::min(extended_rosenbrock_function.num_dimensions() / 2, 20));
        extended_rosenbrock_function_lbfgs.Optimize(
          -1, &extended_rosenbrock_function_optimized);

        // Test whether the evaluation is close to the zero.
        double function_value = extended_rosenbrock_function.Evaluate(
                                  extended_rosenbrock_function_optimized);
        fl::logger->Message() << extended_rosenbrock_function.num_dimensions()
        << " dimensional extended Rosenbrock function: " <<
        "optimized to the function value of " << function_value;
        if (function_value > 0.5 || function_value < -0.5) {
          throw std::runtime_error("Aborted in extended Rosenbrock test");
        }

        // It should converge to something close to all 1's.
        for (int i = 0; i < extended_rosenbrock_function_optimized.length();
             i++) {
          if (extended_rosenbrock_function_optimized[i] > 1.5 ||
              extended_rosenbrock_function_optimized[i] < 0.5) {
            throw std::runtime_error("Invalid optimal point");
          }
        }
      }
    }

    void TestWoodFunction() {
      fl::logger->Message() << "Testing wood function: optimal value: 0";
      fl::ml::lbfgs_test::WoodFunction wood_function;
      fl::data::MonolithicPoint<double> wood_function_optimized;
      fl::ml::Lbfgs< fl::ml::lbfgs_test::WoodFunction > wood_function_lbfgs;
      wood_function.InitStartingIterate(&wood_function_optimized);
      wood_function_lbfgs.Init(wood_function, 2);
      wood_function_lbfgs.Optimize(-1, &wood_function_optimized);

      // It should converge to something close to (1, 1, 1, 1)^T
      for (int i = 0; i < wood_function_optimized.length(); i++) {
        if (wood_function_optimized[i] < 0.5 ||
            wood_function_optimized[i] > 1.5) {
          throw std::runtime_error("Failed in wood function");
        }
      }
    }
};

};
};
};

BOOST_AUTO_TEST_SUITE(TestSuiteLbfgs)
BOOST_AUTO_TEST_CASE(TestCaseLbfgs) {
  fl::Logger::SetLogger(std::string("verbose"));
  fl::ml::lbfgs_test::LbfgsTest test;
  test.TestExtendedRosenbrockFunction();
  test.TestWoodFunction();
  fl::logger->Message() << "All tests passed!";
}
BOOST_AUTO_TEST_SUITE_END()
