/*
Copyright © 2010, Ismion Inc.
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
LLC, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE ISMION INC "AS IS" AND ANY
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

#include "fastlib/table/matrix_table.h"
#include "fastlib/table/default_table.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/table/linear_algebra.h"
#include "fastlib/util/timer.h"

double TableMatrixMul(index_t N, index_t D) {
  // create a dense table
  fl::table::DefaultTable table1;
  table1.Init("table1", 
              std::vector<index_t>(1, D),
              std::vector<index_t>(),
              N);

  fl::table::DefaultTable::Point_t point;
  for(index_t i=0; i<table1.n_entries(); ++i) {
    table1.get(i, &point);
    point.SetRandom(0.0, 1.0);  
  }
  fl::table::MatrixTable result_table;
  fl::util::Timer timer;
  timer.Start();
  fl::table::Mul<fl::la::NoTrans, fl::la::Trans>(table1,
      table1, &result_table);
  timer.End();
  return timer.GetTotalElapsedTime();
}

double TableMatrixMulTrans(index_t N, index_t D) {
  // create a dense table
  fl::table::DefaultTable table1;
  table1.Init("table1", 
              std::vector<index_t>(1, D),
              std::vector<index_t>(),
              N);

  fl::table::DefaultTable::Point_t point;
  for(index_t i=0; i<table1.n_entries(); ++i) {
    table1.get(i, &point);
    point.SetRandom(0.0, 1.0);  
  }
  fl::table::MatrixTable result_table;
  fl::util::Timer timer;
  timer.Start();
  fl::table::Mul<fl::la::Trans, fl::la::NoTrans>(table1,
      table1, &result_table);
  timer.End();
  return timer.GetTotalElapsedTime();
}


double NativeMatrixMul(index_t N, index_t D) { 
  double *native_matrix;
  native_matrix = new double[N*D];
  for(index_t i=0; i<N*D; ++i) {
    native_matrix[i]=fl::math::Random(0.0, 1.0);
  }
  double *native_result=new double[N*N];
  fl::util::Timer timer;
  timer.Start();
  for(index_t i=0; i<N; ++i) {
    for(index_t j=0; j<N; ++j) {
      native_result[i+j*N]=0;
      for(index_t k=0; k<D; ++k) {
        native_result[i+j*N]+=native_matrix[i+k*N]*native_matrix[j+k*N];     
      }
    }
  }
  timer.End();
  delete []native_matrix;
  delete []native_result;
  return timer.GetTotalElapsedTime();
}

double NativeMatrixMulTrans(index_t N, index_t D) { 
  double *native_matrix;
  native_matrix = new double[N*D];
  for(index_t i=0; i<N*D; ++i) {
    native_matrix[i]=fl::math::Random(0.0, 1.0);
  }
  double *native_result=new double[D*D];
  fl::util::Timer timer;
  timer.Start();
  for(index_t i=0; i<D; ++i) {
    for(index_t j=0; j<D; ++j) {
      native_result[i+j*D]=0;
      for(index_t k=0; k<N; ++k) {
        native_result[i+j*D]+=native_matrix[k+i*N]*native_matrix[k+j*N];     
      }
    }
  }
  timer.End();
  delete []native_matrix;
  delete []native_result;
  return timer.GetTotalElapsedTime();
}

double NativeMatrixMulTrans1(index_t N, index_t D) { 
  double *native_matrix;
  native_matrix = new double[N*D];
  for(index_t i=0; i<N*D; ++i) {
    native_matrix[i]=fl::math::Random(0.0, 1.0);
  }
  double *native_result=new double[D*D];
  memset(native_result, 0, D*D*sizeof(double));
  fl::util::Timer timer;
  timer.Start();
  for(index_t i=0; i<N; ++i) {
    for(index_t j=0; j<D; ++j) {
      for(index_t k=0; k<D; ++k) {
        native_result[j+k*D]+=native_matrix[i+k*N]*native_matrix[i+j*N];     
      }
    }
  }
  timer.End();
  delete []native_matrix;
  delete []native_result;
  return timer.GetTotalElapsedTime();
}


int main(int argc, char *argv[]) {
  int iterations=10;
  double time=0;
  double time1=0;
  double time2=0;
  for(int i=0; i<iterations; ++i) {
    time+=TableMatrixMul(1000, 10);
  }
  
  for(int i=0; i<iterations; ++i) {
    time1+=NativeMatrixMul(1000, 10);
  }
  std::cout<<"1000x10 ** "<<time/iterations<<"--"<<time1/iterations<<std::endl;

  time=0;
  time1=0;
  for(int i=0; i<iterations; ++i) {
    time+=TableMatrixMul(1000, 100);
  }
  for(int i=0; i<iterations; ++i) {
    time1+=NativeMatrixMul(1000, 100);
  }
  std::cout<<"1000x100 ** "<<time/iterations<<"--"<<time1/iterations<<std::endl;

  time=0;
  time1=0;
  for(int i=0; i<iterations; ++i) {
    time+=TableMatrixMulTrans(10000, 10);
  }
  for(int i=0; i<iterations; ++i) {
    time1+=NativeMatrixMulTrans(10000, 10);
  }
  time2=0;
  for(int i=0; i<iterations; ++i) {
    time2+=NativeMatrixMulTrans1(10000, 10);
  }

  std::cout<<"10x10000 ** "
    <<time/iterations<<"--"<<time1/iterations
    <<"--"<<time2/iterations
    <<std::endl;

  time=0;
  time1=0;
  for(int i=0; i<iterations; ++i) {
    time+=TableMatrixMulTrans(100000, 10);
  }
  for(int i=0; i<iterations; ++i) {
    time1+=NativeMatrixMulTrans(100000, 10);
  }
  time2=0;
  for(int i=0; i<iterations; ++i) {
    time2+=NativeMatrixMulTrans1(100000, 10);
  }

  std::cout<<"10x100000 ** "
    <<time/iterations<<"--"
    <<time1/iterations
    <<"--"<<time2/iterations
    <<std::endl;

  time=0;
  time1=0;
  for(int i=0; i<iterations; ++i) {
    time+=TableMatrixMulTrans(1000000, 10);
  }
  for(int i=0; i<iterations; ++i) {
    time1+=NativeMatrixMulTrans(1000000, 10);
  }
  time2=0;
  for(int i=0; i<iterations; ++i) {
    time2+=NativeMatrixMulTrans1(1000000, 10);
  }
  std::cout<<"10x1000000 ** "
    <<time/iterations
    <<"--"<<time1/iterations
    <<"--"<<time2/iterations
    <<std::endl;


  time=0;
  time1=0;
  for(int i=0; i<iterations; ++i) {
    time+=TableMatrixMulTrans(100000, 100);
  }
  for(int i=0; i<iterations; ++i) {
    time1+=NativeMatrixMulTrans(100000, 100);
  }
  time2=0;
  for(int i=0; i<iterations; ++i) {
    time2+=NativeMatrixMulTrans1(100000, 100);
  }
  std::cout<<"100x100000 ** "
    <<time/iterations
    <<"--"<<time1/iterations
    <<"--"<<time2/iterations
    <<std::endl;

  time=0;
  time1=0;
  iterations=2;
  for(int i=0; i<iterations; ++i) {
    time+=TableMatrixMulTrans(1000000, 200);
  }
  for(int i=0; i<iterations; ++i) {
    time1+=NativeMatrixMulTrans(1000000, 200);
  }
  std::cout<<"200x1000000 ** "<<time/iterations<<"--"<<time1/iterations<<std::endl;

  time=0;
  time1=0;
  iterations=2;
  for(int i=0; i<iterations; ++i) {
    time+=TableMatrixMulTrans(1000000, 300);
  }
  for(int i=0; i<iterations; ++i) {
    time1+=NativeMatrixMulTrans(1000000, 300);
  }
  std::cout<<"300x1000000 ** "<<time/iterations<<"--"<<time1/iterations<<std::endl;

  time=0;
  time1=0;
  iterations=2;
  for(int i=0; i<iterations; ++i) {
    time+=TableMatrixMulTrans(1000000, 400);
  }
  for(int i=0; i<iterations; ++i) {
    time1+=NativeMatrixMulTrans(1000000, 400);
  }
  std::cout<<"400x1000000 ** "<<time/iterations<<"--"<<time1/iterations<<std::endl;



  time=0;
  time1=0;
  iterations=2;
  for(int i=0; i<iterations; ++i) {
    time+=TableMatrixMulTrans(1000000, 500);
  }
  for(int i=0; i<iterations; ++i) {
    time1+=NativeMatrixMulTrans(1000000, 500);
  }
  std::cout<<"500x1000000 ** "<<time/iterations<<"--"<<time1/iterations<<std::endl;


}
