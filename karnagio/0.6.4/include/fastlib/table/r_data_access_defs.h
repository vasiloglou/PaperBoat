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
#ifndef FL_LITE_FASTLIB_TABLE_R_DATA_ACCESS_DEFS_H_
#define FL_LITE_FASTLIB_TABLE_R_DATA_ACCESS_DEFS_H_

namespace fl {
namespace table {
template<typename TableParamsType>
void RDataAccess::Attach(const std::string &name,
                            fl::table::Table<TableParamsType> * const table) {

  double *ptr;
  index_t n_attributes=0;
  index_t n_points=0;
  Get(name, 
      &ptr, 
      &n_points,
      &n_attributes);
  std::vector<index_t> dense_sizes(1, n_attributes);
  std::vector<index_t> sparse_sizes;
  table->Init(name,
              dense_sizes,
              sparse_sizes,
              n_points);

  typename fl::table::Table<TableParamsType>::Point_t point;
  for(index_t i=0; i<n_points; ++i) {
    table->get(i, &point);
    for(index_t j=0; j<n_attributes; ++j) {
      point.set(j, *(ptr+i*n_attributes+j));
    }
  }
}

template<typename TableParamsType>
void RDataAccess::Attach(const std::string &name,
                            std::vector<index_t> dense_sizes,
                            std::vector<index_t> sparse_sizes,
                            const index_t num_of_points,
                            fl::table::Table<TableParamsType> * const table) {

  table->Init(name, dense_sizes, sparse_sizes, num_of_points);
}

template<typename TableParamsType>
void RDataAccess::Detach(fl::table::Table<TableParamsType> &table) {

}

template<typename TableParamsType>
void RDataAccess::Purge(fl::table::Table<TableParamsType> &table) {
  double *ptr=NULL;
  SEXP r_name = NEW_STRING(table.filename().size());
  if (STRING_PTR(r_name)==NULL) {
     fl::logger->Die() << "Failed to allocate space for a string ";
   }
   SET_STRING_ELT(r_name, 0, mkChar(table.filename().c_str())); 
   SEXP ans;
   PROTECT(ans=NEW_NUMERIC(table.n_entries()*table.n_attributes()));
   SEXP dims ;
   PROTECT(dims = NEW_INTEGER(2));
   INTEGER(VECTOR_DATA(dims)[0])[0]=table.n_entries();
   INTEGER(VECTOR_DATA(dims)[1])[0]=table.n_attributes();
   SET_DIM(ans, dims);
   setVar(r_name, ans, *environment_);
   ptr=NUMERIC_POINTER(ans);

  typename fl::table::Table<TableParamsType>::Point_t point;
  for(index_t i=0; i<table.n_entries(); ++i) {
    table.get(i, &point);
    for(index_t j=0; j<table.n_attributes(); ++j) {
      *(ptr+i*table.n_attributes()+j)=point[j];
    }
  }
  UNPROTECT(2);
}

template<typename TableParamsType1, typename TableParamsType2>
void RDataAccess::TieLabels(fl::table::Table<TableParamsType1> *table,
                               fl::table::Table<TableParamsType2> *labels) {
  if (labels->n_entries() != table->n_entries()) {
    fl::logger->Die() << "Labels and table must have the same number of entries()";
  }
  typename fl::table::Table<TableParamsType2>::Point_t labels_vector;
  labels->get(0, &labels_vector);
  for (index_t i = 0; i < table->n_entries(); ++i) {
    typename fl::table::Table<TableParamsType1>::Point_t p;
    table->get(i, &p);
    p.meta_data().template get<0>() = labels_vector[i];
  }
}

} // table
} // fl

#endif
