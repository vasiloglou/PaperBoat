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
  /**
   *  @brief Given a Table  it computes a table A * A^T and stores it 
   *  to a dense table. If A is too long the memory will explode
   */ 
  template<typename InTableType, typename WorkSpaceType>
  void SelfOuterProduct(InTableType &in_table, 
      typename WorkSpaceType::DefaultTable_t *out_table) {
    if (out_table->n_entries()!=0) {
      if (out_table->n_entries()!=out_table->n_attributes() 
          || out_table->n_attributes()!=in_table.n_entries()) {
        fl::logger->Die()<<"Table dimensions do not agree "
            "in table "<<in_table.n_entries() <<" x "
            << in_table.n_attributes()
            <<" , while out table is initialized as "
            << out_table->n_entries() <<" x "
            << out_table->n_attributes();
      } 
    } else {
      out_table->Init("", 
          std::vector<index_t>(1, in_table.n_entries()),
          std::vector<index_t>(),
          in_table.n_entries());
      
    }
    // load the table in a vector of pairs
    // and then do the outer product
    typename InTableType::Point_t point;
    typename InTableType::Point_t::iterator it;
    std::vector<std::vector<std::pair<index_t, double> > >
      cont(in_table.n_attributes());
    for(index_t i=0; i<in_table.n_entries(); ++i) {
      in_table.get(i, &point);
      for(it=point.begin(); it!=point.end(); ++it) {
         cont[it.attribute()].push_back(
             std::make_pair(i, it.value()));   
      }
    }
    // now sort them
    for(size_t i=0; i<cont.size(); ++i) {
      std::sort(cont[i].begin(), cont[i].end());
    }
    out_table->SetAll(0.0);
    for(size_t i=0; i<cont.size(); ++i) {
      for(std::vector<std::pair<index_t, double> >::iterator it1=cont[i].begin();
          it1!=cont[i].end(); ++it1) {
        for(std::vector<std::pair<index_t, double> >::iterator it2=it1;
            it2!=cont[i].end(); ++it2) {
          out_table->UpdatePlus(
              it1->first, 
              it2->first, 
              it1->second *it2->second);
          out_table->set(it2->first,
              it2->first,
              out_table->get(it1->first,
                it2->first));
        }
      }
    }
  }

  /**
   *  @brief Given a table A it compues A * A^T and stores it in 
   *         a sparse matrix. If A must be sparse, actually very sparse
   *         so that A * A^T is still sparse.
   */
  template<typename InTableType, typename WorkSpaceType>
  void SelfOuterProduct(InTableType &in_table,
     double clip_value,
     typename WorkSpaceType::DefaultSparseDoubleTable_t *out_table) {
    if (out_table->n_entries()!=0) {
      if (out_table->n_entries()!=out_table->n_attributes() 
          || out_table->n_attributes()!=in_table.n_entries()) {
        fl::logger->Die()<<"Table dimensions do not agree "
            "in table "<<in_table.n_entries() <<" x "
            << in_table.n_attributes()
            <<" , while out table is initialized as "
            << out_table->n_entries() <<" x "
            << out_table->n_attributes();
      } 
    } else {
      out_table->Init("", 
          std::vector<index_t>(),
          std::vector<index_t>(1, in_table.n_entries()),
          in_table.n_entries());
      
    }
    // load the table in a vector of pairs
    // and then do the outer product
    typename InTableType::Point_t point;
    typename InTableType::Point_t::iterator it;
    std::vector<std::vector<std::pair<index_t, double> > >
      cont(in_table.n_attributes());
    for(index_t i=0; i<in_table.n_entries(); ++i) {
      in_table.get(i, &point);
      for(it=point.begin(); it!=point.end(); ++it) {
         cont[it.attribute()].push_back(
             std::make_pair(i, it.value()));   
      }
    }
    // now sort them
    for(size_t i=0; i<cont.size(); ++i) {
      std::sort(cont[i].begin(), cont[i].end());
    }
    out_table->SetAll(0.0);
    std::vector<std::map<index_t, double> > result(in_table.n_entries());
    for(size_t i=0; i<cont.size(); ++i) {
      for(std::vector<std::pair<index_t, double> >::iterator it1=cont[i].begin();
          it1!=cont[i].end(); ++it1) {
        for(std::vector<std::pair<index_t, double> >::iterator it2=it1;
            it2!=cont[i].end(); ++it2) {
          double value = it1->second *it2->second;
          result[it1->first][it2->first] += value;          
        }
      }
    }

    std::vector<std::map<index_t,double> > to_be_included; 
    bool clip = (clip_value!=-std::numeric_limits<double>::max());
    if (clip==true) {
      to_be_included.resize(result.size());
    }
    for(size_t i=0; i<result.size(); ++i) {
      for(std::map<index_t, double>::iterator 
          it=result[i].begin();
          it!=result[i].end(); ++it) {
        if (clip==true) {
          if (it->second>clip_value) {
            to_be_included[i][it->first]=it->second;
            to_be_included[it->first][i]=it->second;
          } 
        } else {
          result[it->first][i]=it->second; 
        }
      }
    }
    
    typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t point1;
    for(index_t i=0; i<out_table->n_entries(); ++i) {
      out_table->get(i, &point1);
      std::vector<std::pair<index_t, double> > filtered;
      if (clip==false) {    
        point1.template sparse_point<double>().Load(
            result[i].begin(), result[i].end());
      } else {
        point1.template sparse_point<double>().Load(
            to_be_included[i].begin(), to_be_included[i].end());

      }
    }
  }

  /**
   *  @brief Given a Table  it computes a table A^T * A and stores it 
   *  to a dense table. If A is too long the memory will explode
   */ 
  template<typename InTableType, typename WorkSpaceType>
  void SelfInnerProduct(InTableType &in_table, typename WorkSpaceType::DefaultTable_t *out_table) {
    if (out_table->n_entries()!=0) {
      if (out_table->n_entries()!=out_table->n_attributes() 
          || out_table->n_attributes()!=in_table.n_attributes()) {
        fl::logger->Die()<<"Table dimensions do not agree "
            "in_table is "<<in_table.n_entries() <<" x "
            << in_table.n_attributes()
            <<" , while out_table is initialized as "
            << out_table->n_entries() <<" x "
            << out_table->n_attributes();
      } 
    } else {
      out_table->Init("", 
      std::vector<index_t>(1, in_table.n_attributes()),
            std::vector<index_t>(),
            in_table.n_attributes());
    }
   
    // load the table in a vector of pairs
    // and then do the inner product
    typename InTableType::Point_t point;
    typename InTableType::Point_t::iterator it1, it2;
    out_table->SetAll(0.0);
    for(index_t i=0; i<in_table.n_entries(); ++i) {
      in_table.get(i, &point);
      for(it1=point.begin(); it1!=point.end(); ++it1) {
        for(it2=it1; it2!=point.end(); ++it2) {
          double value=it1.value()*it2.value();
          out_table->UpdatePlus(it1.attribute(), 
              it2.attribute(), value);
        }   
      }
    }
    for(index_t i=0; i<out_table->n_entries(); ++i) {
      for(index_t j=i+1; j<out_table->n_attributes(); ++j) {
        out_table->set(j, i, out_table->get(i, j));
      }
    }
  }

  /**
   *  @brief Given a table A it compues A^T * A and stores it in 
   *         a sparse matrix. If A must be sparse, actually very sparse
   *         so that A^T * A is still sparse.
   */
  template<typename InTableType, typename WorkSpaceType>
  void SelfInnerProduct(InTableType &in_table, 
      double clip_value,
     typename WorkSpaceType::DefaultSparseDoubleTable_t *out_table) {
    if (out_table->n_entries()!=0) {
      if (out_table->n_entries()!=out_table->n_attributes() 
          || out_table->n_attributes()!=in_table.n_attributes()) {
        fl::logger->Die()<<"Table dimensions do not agree "
            "in table "<<in_table.n_entries() <<" x "
            << in_table.n_attributes()
            <<" , while out table is initialized as "
            << out_table->n_entries() <<" x "
            << out_table->n_attributes();
      } 
    } else {
      out_table->Init("", 
            std::vector<index_t>(),
            std::vector<index_t>(1, in_table.n_entries()),
            in_table.n_entries());
    }
    typename InTableType::Point_t point;
    typename InTableType::Point_t::iterator it1, it2;
    out_table->SetAll(0.0);
    std::vector<std::map<index_t, double> > result(in_table.n_attributes());
    for(index_t i=0; i<in_table.n_entries(); ++i) {
      in_table.get(i, &point);
      for(it1=point.begin(); it1!=point.end(); ++it1) {
        for(it2=it1; it2!=point.end(); ++it2) {
          double value=it1.value()*it2.value();
          result[it1.attribute()][it2.attribute()]+=value;
        }
      }
    }
    // Symmetrize
    std::vector<std::map<index_t,double> > to_be_included; 
    bool clip=(clip_value!=-std::numeric_limits<double>::max());
    if (clip==true) {
      to_be_included.resize(result.size());
    }
    for(size_t i=0; i<result.size(); ++i) {
      for(std::map<index_t, double>::iterator 
          it=result[i].begin();
          it!=result[i].end(); ++it) {
        if (clip==true) {
          if (it->second>clip_value) {
            to_be_included[i][it->first]=it->second;
            to_be_included[it->first][i]=it->second;
          } 
        } else {
          result[it->first][i]=it->second; 
        }
      }
    }

    
    typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t point1;
    for(index_t i=0; i<out_table->n_entries(); ++i) {
      out_table->get(i, &point1);
      if (clip==false) {
        point1.template sparse_point<double>().Load(
            result[i].begin(), result[i].end());
      } else {
        point1.template sparse_point<double>().Load(
            to_be_included[i].begin(), to_be_included[i].end());
      }
    }
  }

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
    FL_SCOPED_LOG(Scale);
    if (c_table->n_entries()==0) {
      c_table->Init("",
          a_table.dense_sizes(),
          a_table.sparse_sizes(), 
          a_table.n_entries());
    } else {
      if (c_table->n_entries()!=a_table.n_entries()
          || c_table->n_attributes()!=a_table.n_entries()) {
        fl::logger->Die()<<"Result table is not initialized properly"
          <<std::endl;
      }
    }  
    if (b_table.n_attributes()!=a_table.n_attributes()) {
      fl::logger->Die()<<"Input arguments have incorrect dimensions"
        <<std::endl;
    }
    typename TableA::Point_t a_point;
    typename TableB::Point_t b_point;
    typename TableA::Point_t c_point;
    b_table.get(0, &b_point);
    for(index_t i=0; i<a_table.n_entries(); ++i) {
      a_table.get(i, &a_point);
      c_table->get(i, &c_point); 
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
    fl::dense::ops::QR<fl::la::Overwrite, fl::la::Trans>(
        table_in.get_point_collection().dense->get<double>(), 
        &(q_table->get_point_collection().dense->get<double>()), 
        &(r_table->get_point_collection().dense->get<double>()), &success_flag); 
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
      fl::dense::ops::SVD<fl::la::Overwrite, fl::la::NoTrans>(
          table_in.get_point_collection().dense->get<double>(), 
          &(sv->get_point_collection().dense->get<double>()), 
         &temp_right, &(left->get_point_collection().dense->get<double>()), &success_flag);
       fl::dense::ops::Transpose<fl::la::Overwrite>(
           temp_right, &(right_trans->get_point_collection().dense->get<double>()));      
    } else {
       fl::dense::ops::SVD<fl::la::Overwrite, fl::la::Trans>(
           table_in.get_point_collection().dense->get<double>(), 
           &(sv->get_point_collection().dense->get<double>()), 
          &(left->get_point_collection().dense->get<double>()), 
          &(right_trans->get_point_collection().dense->get<double>()), &success_flag);
    }
    if (success_flag!=SUCCESS_PASS) {
      fl::logger->Warning()<<"There was an error in LAPACK SVD computation, "
        "problem unstable"<<std::endl;
    }
  }

}}
#endif
