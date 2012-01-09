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

#ifndef PAPERBOAT_INCLUDE_FASTLIB_TABLE_LINEAR_ALGEBRA_H_
#define PAPERBOAT_INCLUDE_FASTLIB_TABLE_LINEAR_ALGEBRA_H_
#include "boost/mpl/if.hpp"
#include "fastlib/la/linear_algebra_defs.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/dense/linear_algebra.h"
#include "boost/type_traits/is_same.hpp"
#include "boost/static_assert.hpp"
#include "fastlib/table/matrix_table.h"

namespace fl {namespace table {
  template<fl::la::TransMode IsTransA=fl::la::NoTrans, 
    fl::la::TransMode IsTransB=fl::la::NoTrans>
  class Mul {
    public:
      struct SelectOrder1 {
        template<typename PointType1, typename PointType2>
        static double Do(PointType1 &p1, PointType2 &p2) {
          return fl::la::Dot(p1, p2);
        } 
      };
      struct SelectOrder2 {
        template<typename PointType1, typename PointType2>
        static double Do(PointType1 &p1, PointType2 &p2) {
          return fl::la::Dot(p2, p1);
        } 
      };

      template<typename TableA, typename TableB, typename TableC>
      Mul(TableA &a_table, TableB &b_table, TableC *c_table) {
        BOOST_STATIC_ASSERT((IsTransB==fl::la::NoTrans &&
            boost::is_same<TableC, MatrixTable>::value) ||
            IsTransB==fl::la::Trans);  
        typename TableA::Point_t point_a;
        typename TableB::Point_t point_b;
        typename TableC::Point_t point_c;
        if (IsTransA==fl::la::NoTrans && IsTransB==fl::la::Trans) {
          if (a_table.n_attributes()!=b_table.n_attributes()) {
            fl::logger->Die()<<"Dimension of input matrices are not "
              "correct";
          }
          if (c_table->n_entries()==0) {
            c_table->Init("",
                std::vector<index_t>(1, b_table.n_entries()),
                std::vector<index_t>(),
                a_table.n_entries());
          } else {
            if (c_table->n_entries()!= a_table.n_entries() ||
                c_table->n_attributes()!= b_table.n_entries()) {
              fl::logger->Die()<<"Result table has been initialized "
                "with the wrong dimensions";
            }
          }
          c_table->SetAll(0.0);
          for(index_t i=0; i<a_table.n_entries(); ++i) {
            a_table.get(i, &point_a);
            c_table->get(i, &point_c);
            for(index_t j=0; j<b_table.n_entries(); ++j) {
              b_table.get(j, &point_b);
              // most likely the b is the dense point
              // so put it first
              double result=boost::mpl::if_<
                boost::is_same<
                  TableA, MatrixTable
                >,
                SelectOrder1,
                SelectOrder2
              >::type::Do(point_a, point_b);
              point_c.set(j, result);
            }
          }
        } 
        if (IsTransA==fl::la::Trans && IsTransB==fl::la::Trans) {
          fl::logger->Die()<<"You are probably computing "
              "a product that doesn't make sense"; 
        }
        if (IsTransA==fl::la::Trans && IsTransB==fl::la::NoTrans) {
          if (a_table.n_entries()!=b_table.n_entries()) {  
            fl::logger->Die()<<"Dimension of input matrices are not "
                "correct";
          }
          if (c_table->n_entries()==0) {
            c_table->Init("",
                std::vector<index_t>(1, b_table.n_attributes()),
                std::vector<index_t>(),
                a_table.n_attributes()
                );
          } else {
            if (c_table->n_entries()!= a_table.n_attributes() ||
                c_table->n_attributes()!= b_table.n_attributes()) {
              fl::logger->Die()<<"Result table has been initialized "
                "with the wrong dimensions";
            }
          }
          c_table->SetAll(0.0);
          for(index_t i=0; i<a_table.n_entries(); ++i) {
            a_table.get(i, &point_a);
            b_table.get(i, &point_b);
            // this is suboptimal
            for(typename TableA::Point_t::iterator ita=point_a.begin();
                ita!=point_a.end(); ++ita) {
              for(typename TableB::Point_t::iterator itb=point_b.begin();
                  itb!=point_b.end(); ++itb) {
               c_table->UpdatePlus(ita.attribute(), itb.attribute(),
                   ita.value()*itb.value());
              }
            } 
          }
        }
        if (IsTransA==fl::la::NoTrans && IsTransB==fl::la::NoTrans) {
          if (a_table.n_attributes()!=b_table.n_entries()) {  
            fl::logger->Die()<<"Dimension of input matrices are not "
                "correct";
          }
          if (c_table->n_entries()==0) {
            c_table->Init("",
                std::vector<index_t>(1, b_table.n_attributes()),
                std::vector<index_t>(),
                a_table.n_entries()); 
          } else {
            if (c_table->n_entries()!=a_table.n_entries() ||
                c_table->n_attributes()!=b_table.n_attributes()) {
              fl::logger->Die()<<"Result table has been initialized "
                "with the wrong dimensions";            
            }
          }
          c_table->SetAll(0.0);
          for(index_t i=0; i<a_table.n_entries(); ++i) {
            a_table.get(i, &point_a);
            for(typename TableA::Point_t::iterator ita=point_a.begin();
                ita!=point_a.end(); ++ita) {
              b_table.get(ita.attribute(), &point_b);
              for(typename TableB::Point_t::iterator itb=point_b.begin();
                  itb!=point_b.end(); ++itb) {
                c_table->UpdatePlus(i, itb.attribute(),  
                    ita.value() * itb.value());
              } 
            }
          }      
        }
      }
  };
  
  template<typename TableA, typename TableB>
  void Scale(TableA &a_table, TableB &b_table, TableA *c_table) {
    if (c_table->n_entries()==0) {
      c_table->Init("",
          a_table.dense_sizes(),
          a_table.sparse_sizes(), 
          a_table.n_entries());
    } else {
      if (c_table->n_entries()!=a_table.n_entries()
          !=c_table->n_attributes()!=a_table.n_entries()) {
        fl::logger->Die()<<"Result table is not initializes properly"
          <<std::endl;
      }
    }  
    if (b_table.size()!=a_table.n_attributes()) {
      fl::logger->Die()<<"Input arguments have incorrect dimensions"
        <<std::endl;
    }
    typename TableA::Point_t a_point;
    typename TableB::Point_t b_point;
    typename TableA::Point_t c_point;
    for(index_t i=0; i<a_table.n_enries(); ++i) {
      a_table.get(i, &a_point);
      c_table.get(i, &c_point); 
      c_point.SetAll(0.0);
      for(typename TableA::Point_t::iterator it=a_point.begin();
          it!=a_point.end(); ++it) {
        c_point.set(it.attribute(), 
            it.value()*b_point[it.attribute()]);
      }
    }
  }

  template<typename TableA, typename TableB>
  void SelfScale(TableB &b_table, TableA *a_table) {
    if (b_table.size()!=a_table.n_attributes()) {
      fl::logger->Die()<<"Input arguments have incorrect dimensions"
        <<std::endl;
    }
    typename TableA::Point_t a_point;
    typename TableB::Point_t b_point;
    typename TableA::Point_t c_point;
    for(index_t i=0; i<a_table.n_enries(); ++i) {
      a_table.get(i, &a_point);
      for(typename TableA::Point_t::iterator it=a_point.begin();
          it!=a_point.end(); ++it) {
        a_point.set(it.attribute(), 
            it.value()*b_point[it.attribute()]);
      }
    }
  }

  /**
   * QR decomposition for matrices disguised under tables
   *    no need to initialize the results
   *    It will automatically detect if they are initialized and
   *    it will initialize them properly
   */ 
  inline void QR(const fl::table::MatrixTable &table_in, fl::table::MatrixTable *q_table,
          fl::table::MatrixTable *r_table) {
    if(table_in.n_entries()<table_in.n_attributes()) {
      fl::logger->Die()<<"QR error, number of table entries must be "
        "greater than table attributes";
    }
    if (q_table->n_entries()==0) {
      q_table->Init("", 
          std::vector<index_t>(1, table_in.n_attributes()),
          std::vector<index_t>(), 
          table_in.n_entries());
    } else {
      if (q_table->n_entries()!=table_in.n_entries() ||
          q_table->n_attributes()!=table_in.n_attributes()) {
        fl::logger->Die()<<"QR error, the q_table is not initialized "
          "properly";
      }
    }
    if (r_table->n_entries()==0) {
      r_table->Init("", 
          std::vector<index_t>(1, table_in.n_attributes()),
          std::vector<index_t>(), 
          table_in.n_attributes());
    } else {
      if (r_table->n_entries()!=table_in.n_attributes()||
          r_table->n_attributes()!=table_in.n_attributes()) {
        fl::logger->Die()<<"QR error, the q_table is not initialized "
          "properly";
      }
    }
    success_t success_flag;
    fl::dense::ops::QR<fl::la::Overwrite, fl::la::Trans>(table_in.get(), 
        &(q_table->get()), &(r_table->get()), &success_flag); 
    if (success_flag!=SUCCESS_PASS) {
      fl::logger->Warning()<<"There was an error in LAPACK QR computation, "
        "problem unstable"<<std::endl;
    }
  }
  
  /**
   * SVD decomposition for matrices disguised under tables
   *    no need to initialize the results
   *    It will automatically detect if they are initialized and
   *    it will initialize them properly
   */ 
  inline void SVD(const fl::table::MatrixTable &table_in,
           fl::table::MatrixTable *sv,
           fl::table::MatrixTable *left,
           fl::table::MatrixTable *right_trans) {

    index_t svd_rank=std::min(table_in.n_attributes(),
        table_in.n_entries());
    if (sv->n_entries()==0) {
      sv->Init("",
          std::vector<index_t>(1, 1),
          std::vector<index_t>(), 
          svd_rank); 
    } else {
      if (sv->n_attributes()!=1 || 
          sv->n_entries() != svd_rank) {
        fl::logger->Die()<<"Svd error, improper initialization of "
          " sv";
      } else {
        svd_rank=sv->n_entries();
      }
    }
    if (left->n_entries()==0) {
      left->Init("",
         std::vector<index_t>(1, svd_rank),
         std::vector<index_t>(),
         table_in.n_entries());    
    } else {
      if (left->n_attributes()!=svd_rank ||
          left->n_entries()!=table_in.n_entries()) {
        fl::logger->Die()<<"Svd error, improper initialization of "
          " left";
      }
    }
    if (right_trans->n_entries()==0) {
      right_trans->Init("",
         std::vector<index_t>(1, svd_rank),
         std::vector<index_t>(),
         table_in.n_attributes());   
    } else {
      if (right_trans->n_attributes()!=svd_rank ||
          right_trans->n_entries()!=table_in.n_attributes()) {
        fl::logger->Die()<<"Svd error, improper initialization of "
          " right";
      }
    }
    success_t success_flag;
    if (table_in.n_entries()<table_in.n_attributes()) {
      fl::dense::Matrix<double> temp_right;
      temp_right.Init(table_in.n_attributes(), table_in.n_entries());
      fl::dense::ops::SVD<fl::la::Overwrite, fl::la::NoTrans>(table_in.get(), &(sv->get()), 
         &temp_right, &(left->get()), &success_flag);
       fl::dense::ops::Transpose<fl::la::Overwrite>(
           temp_right, &(right_trans->get()));      
    } else {
       fl::dense::ops::SVD<fl::la::Overwrite, fl::la::Trans>(table_in.get(), &(sv->get()), 
          &(left->get()), &(right_trans->get()), &success_flag);
    }
    if (success_flag!=SUCCESS_PASS) {
      fl::logger->Warning()<<"There was an error in LAPACK SVD computation, "
        "problem unstable"<<std::endl;
    }

  }

}}
#endif
