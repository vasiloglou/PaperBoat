/*
Copyright © 2010, Ismion Inc
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
#ifndef PAPER_BOAT_FASTLIB_TABLE_MATRIX_TABLE_H_
#define PAPER_BOAT_FASTLIB_TABLE_MATRIX_TABLE_H_
#include "default/dense/labeled/kdtree/table.h"

namespace fl { namespace table{
typedef fl::table::dense::labeled::kdtree::Table  MatrixTable;
/*
  class MatrixTable : public fl::table::dense::labeled::kdtree::Table {
  public:
    typedef fl::table::dense::labeled::kdtree::Table::Point_t Point_t;
    fl::dense::Matrix<double> &get() {
      return this->get_point_collection().dense->get<double>();
    }
    
    const fl::dense::Matrix<double> &get() const {
      return this->get_point_collection().dense->get<double>();
    }

    double get(index_t i, index_t j=0) const {
      return this->get_point_collection().dense->get<double>().get(j,i);
    }
    
    void get(index_t i, Point_t *p) const {
      fl::table::dense::labeled::kdtree::Table::get(i, p);
    }

    void set(index_t i, index_t j, double value) {
      this->get_point_collection().dense->get<double>().set(j, i, value);
    }
    void SetAll(double value) {
      this->get_point_collection().dense->get<double>().SetAll(value);
    }
    void UpdatePlus(index_t i, index_t j, double value) {
      double old_value=this->get(i, j);
      this->set(i, j, old_value+value); 
    }
    void UpdateMul(index_t i, index_t j, double value) {
      double old_value=this->get(i, j);
      this->set(i, j, old_value*value);   
    }
};
*/
}}
#endif
