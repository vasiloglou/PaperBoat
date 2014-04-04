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
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/test/unit_test.hpp"
#include "fastlib/data/point.h"
#include "fastlib/la/linear_algebra.h"

void TestSparsePoint() {
  fl::data::SparsePoint<float> point1;
  point1.Init(100);
  point1.set(0, 13.4f);
  point1.set(99, 33.2f);
  point1.set(42, 21.2f);
  BOOST_ASSERT(point1.get(0) == 13.4f);
  BOOST_ASSERT(point1[1] == 0.0f);
  BOOST_ASSERT(point1.get(99) == 33.2f);
  BOOST_ASSERT(point1.get(42) == 21.2f);
  // test serialization
  {
    std::ofstream ofs("filename");
    boost::archive::text_oarchive oa(ofs);
    oa << point1;
  }
  { 
    fl::data::SparsePoint<float> new_point1;
    std::ifstream ifs("filename");
    boost::archive::text_iarchive ia(ifs);
    ia >> new_point1;
    BOOST_ASSERT(new_point1.get(0) == 13.4f);
    BOOST_ASSERT(new_point1[1] == 0.0f);
    BOOST_ASSERT(new_point1.get(99) == 33.2f);
    BOOST_ASSERT(new_point1.get(42) == 21.2f);
  }
   
  fl::data::SparsePoint<float> point2;
  point2.Alias(point1);
  BOOST_ASSERT(point1.size() == point2.size());
  for (index_t i = 0; i < point2.size(); i++) {
    BOOST_ASSERT(point1.get(i) == point2[i]);
  }
  // test serialization
  {
    std::ofstream ofs("filename");
    boost::archive::text_oarchive oa(ofs);
    oa << point2;
  }
  { 
    fl::data::SparsePoint<float> new_point2;
    std::ifstream ifs("filename");
    boost::archive::text_iarchive ia(ifs);
    ia >> new_point2;
    BOOST_ASSERT(new_point2.get(0) == 13.4f);
    BOOST_ASSERT(new_point2[1] == 0.0f);
    BOOST_ASSERT(new_point2.get(99) == 33.2f);
    BOOST_ASSERT(new_point2.get(42) == 21.2f);
  }

  fl::data::SparsePoint<float> point3(100);
  point3.set(0, 23.4f);
  point3.set(99, 43.2f);
  point3.set(42, 31.2f);
  point3.SwapValues(&point1);
  BOOST_ASSERT(point3.get(0) == 13.4f);
  BOOST_ASSERT(point3[1] == 0.0f);
  BOOST_ASSERT(point3.get(99) == 33.2f);
  BOOST_ASSERT(point3.get(42) == 21.2f);

  BOOST_ASSERT(point1.get(0) == 23.4f);
  BOOST_ASSERT(point1[1] == 0.0f);
  BOOST_ASSERT(point1.get(99) == 43.2f);
  BOOST_ASSERT(point1.get(42) == 31.2f);
  // Some linear algebra tests
  fl::data::SparsePoint<float> point4;
  point4.Init(100);
  point4.set(0, 13.4f);
  point4.set(99, 33.2f);
  point4.set(42, 21.2f);
  point4.Print(std::cout, ",");
  std::cout << "\n";

  fl::data::SparsePoint<float> point5;
  point5.Init(100);
  point5.set(0, -10.4f);
  point5.set(99, 23.2f);
  point5.set(43, -21.2f);
  point5.Print(std::cout, ",");
  std::cout << "\n";

  BOOST_ASSERT((fl::data::SparsePoint<float>::LengthEuclidean(point4) ==
                fl::math::Pow<float, 1, 2>(13.4*13.4 + 33.2*33.2 + 21.2*21.2)));
  BOOST_ASSERT((fabs(fl::data::SparsePoint<float>::Dot(point4, point4) -
                     (13.4*13.4 + 33.2*33.2 + 21.2*21.2) < 1e-3)));
  BOOST_ASSERT((fabs(fl::data::SparsePoint<float>::Dot(point4, point5) -
                     (-13.4*10.4 + 33.2*23.2)) < 1e-2));
  BOOST_ASSERT((fabs(fl::data::SparsePoint<float>::Dot(point5, point4) ==
                     (-13.4*10.4 + 33.2*23.2)) < 1e-2));
  float distancesq;
  fl::data::SparsePoint<float>::RawLMetric<2>(point4, point5, &distancesq);
  BOOST_ASSERT(fabs(distancesq - ((13.4 + 10.4)*(13.4 + 10.4) + (33.2 - 23.2)*(33.2 - 23.2) +
                                  21.2*21.2 + 21.2*21.2)) < 1e-3);

}

struct MyMixedPointArgs : public fl::data::MixedPointArgs {
  typedef boost::mpl::vector1<double> DenseTypes;
  typedef boost::mpl::vector1<float>  SparseTypes;
  typedef double CalcPrecisionType;
  typedef index_t MetaDataType;
};
typedef fl::data::MixedPoint<MyMixedPointArgs> MyMixedPoint;

/*
  typedef boost::mpl::vector1<double>::type DenseTypes;
  typedef boost::mpl::vector1<float>::type  SparseTypes;
  typedef boost::mpl::map4<
    boost::mpl::pair<fl::data::MixedPointArgs::DenseTypes, DenseTypes>,
    boost::mpl::pair<fl::data::MixedPointArgs::SparseTypes, SparseTypes>,
    boost::mpl::pair<fl::data::MixedPointArgs::CalcPrecisionType, double>,
    boost::mpl::pair<fl::data::MixedPointArgs::MetaDataType, index_t> > Args;
  typedef fl::data::MixedPoint<Args> MyMixedPoint;
*/
void TestMixedPoint() {
  std::vector<index_t> sizes(2);
  sizes[0] = 2;
  sizes[1] = 100;
  MyMixedPoint point1, point2, point4;
  point1.Init(sizes);
  BOOST_ASSERT(point1.size() == 102);
  point1.set_meta_data(1000);
  point1.set(0, 1.1f);
  point1.set(1, 2.2f);
  point1.set(44, 4.4f);
  point1.set(77, 7.7f);
  BOOST_ASSERT(point1[0] == 1.1f);
  BOOST_ASSERT(point1[1] == 2.2f);
  BOOST_ASSERT(fabs(point1[44] - 4.4f) < 1e-4);
  BOOST_ASSERT(fabs(point1[77] - 7.7f) < 1e-4);
  // test serialization
  {
    std::ofstream ofs("filename");
    boost::archive::text_oarchive oa(ofs);
    oa << point1;
  }
  { 
    MyMixedPoint new_point1;
    std::ifstream ifs("filename");
    boost::archive::text_iarchive ia(ifs);
    ia >> new_point1;
    BOOST_ASSERT(new_point1[0] == 1.1f);
    BOOST_ASSERT(new_point1[1] == 2.2f);
    BOOST_ASSERT(new_point1[44] == 4.4f);
    BOOST_ASSERT(new_point1[77] == 7.7f);
  }

  point4.Init(sizes);
  point4.set_meta_data(1001);
  point4.set(0, 1.1f);
  point4.set(1, 2.2f);
  point4.set(44, 4.4f);
  point4.set(76, 7.7f);

  point1.Print(std::cout, ",");
  std::cout << "\n";
  fl::data::MonolithicPoint<double> point3;
  fl::dense::Matrix<double, true> point5;
  point3.Init(102);
  for (size_t i = 0; i < point3.size(); i++) {
    point3[i] = i + 1;
  }
  double result = fl::la::Dot(point3, point1);
  BOOST_ASSERT(fabs(result - (1.1*1 + 2.2*2 + 4.4*45 + 7.7*78)) < 1e-1);
  result = fl::la::Dot(point1, point1);
  BOOST_ASSERT(fabs(result - (1.1*1.1 + 2.2*2.2 + 4.4*4.4 + 7.7*7.7)) < 1e-1);

  fl::la::AddExpert(double(2), point1, &point3);

  BOOST_ASSERT(fabs(point3[0] - (1 + 2*1.1)) < 1e-1);
  BOOST_ASSERT(fabs(point3[1] - (2 + 2*2.2)) < 1e-1);
  BOOST_ASSERT(fabs(point3[44] - (45 + 2*4.4)) < 1e-1);
  BOOST_ASSERT(fabs(point3[77] - (78 + 2*7.7)) < 1e-1);

  result = fl::la::Dot(point1, point1);
  BOOST_ASSERT(fabs(result - (1.1*1.1 + 2.2*2.2 + 4.4*4.4 + 7.7*7.7)) < 1e-1);

  fl::la::RawLMetric<2>(point1.sparse_point<float>(), point4.sparse_point<float>(), &result);
  fl::la::RawLMetric<2>(point4.sparse_point<float>(), point1.sparse_point<float>(), &result);
  fl::la::RawLMetric<2>(point4, point1, &result);
  BOOST_ASSERT(fabs(result - (2*7.7*7.7)) < 1e-1);
  BOOST_ASSERT(fl::la::DistanceSqEuclidean(point1, point1) == 0);




  BOOST_ASSERT(point1[0] == 1.1f);
  BOOST_ASSERT(point1[1] == 2.2f);
  BOOST_ASSERT(point1[44] == 4.4f);
  BOOST_ASSERT(point1[77] == 7.7f);
  BOOST_ASSERT(point1.meta_data() == 1000);

  point2.Alias(point1);
  BOOST_ASSERT(point1[0] == point2[0]);
  BOOST_ASSERT(point1[1] == point2[1]);
  BOOST_ASSERT(point1[44] == point2[44]);
  BOOST_ASSERT(point1[77] == point2[77]);
  BOOST_ASSERT(point1.meta_data() == point2.meta_data());
  MyMixedPoint point6;
  point6.Copy(point1);
  BOOST_ASSERT(point1[0] == point6[0]);
  BOOST_ASSERT(point1[1] == point6[1]);
  BOOST_ASSERT(point1[44] == point6[44]);
  BOOST_ASSERT(point1[77] == point6[77]);
  BOOST_ASSERT(point1.meta_data()
               == point6.meta_data());
  point6.SetAll(0.0);
};

BOOST_AUTO_TEST_SUITE(TestPoint)
BOOST_AUTO_TEST_CASE(TesSparsePoint) {
  TestSparsePoint();
  BOOST_MESSAGE("SparsePoint tests passed!!!");
}
BOOST_AUTO_TEST_CASE(TesMixedPoint) {
  TestMixedPoint();
  BOOST_MESSAGE("MixedPoint tests passed!!!");
}
BOOST_AUTO_TEST_SUITE_END()

