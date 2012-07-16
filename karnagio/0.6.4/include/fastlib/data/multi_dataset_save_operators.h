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
#ifndef FL_LITE_FASTLIB_DATA_MULTIDATASET_SAVE_OPERATORS_H_
#define FL_LITE_FASTLIB_DATA_MULTIDATASET_SAVE_OPERATORS_H_
#include "boost/type_traits/is_same.hpp"
/**
 * @brief the following structs are used for saving a point
 *
 */
struct DenseSavePoint {

  DenseSavePoint(DenseIterators *its,
                 std::ostream *out,
                 const std::string *delimeter) {
    its_ = its;
    out_ = out;
    delimeter_ = delimeter;
  }

  template<typename T>
  void operator()(T) {
    typename DenseIterators::template wrap<T>::Iterator_t &it =
      its_->template get<T>();
    index_t len = it->size();
    for (index_t i = 0; i < len; i++) {
      if (!boost::is_same<T, signed char>::value &&  !boost::is_same<T, unsigned char>::value) {
        *out_ << it->get(i) << *delimeter_;
      } else {
        *out_ << static_cast<int>(it->get(i)) << *delimeter_;
      }
      if (out_->fail()) {
        fl::logger->Die() << "Somethng went wrong while saving data to a file."
          <<" Maybe you are writing out of disk space.";
      }
    }
    ++it;
  }
private:
  DenseIterators *its_;
  std::ostream *out_;
  const std::string *delimeter_;
};

struct SparseSavePoint {
  struct CompressedFormatChoice {
    template<typename T>
    static void Print(std::ostream &out, index_t ind, T &token) {
      if (!boost::is_same<T, signed char>::value &&  !boost::is_same<T, unsigned char>::value) {
        out << ind << ":" << token;
      } else {
        out << ind << ":" << static_cast<int>(token);
      }
      if (out.fail()) {
        fl::logger->Die() << "Somethng went wrong while saving data to a file."
          <<"Maybe you are running out of disk space";

      }
    }
  };
  struct MixedFormatChoice {
    template<typename T>
    static void Print(std::ostream &out, index_t ind, T &token) {
      if (!boost::is_same<T, signed char>::value &&  !boost::is_same<T, unsigned char>::value) {
        out << Typename<T>::Name() << ind << ":" << token;
      } else {
        out << Typename<T>::Name() << ind << ":" << static_cast<int>(token);
      }
      if (out.fail()) {
        fl::logger->Die() << "Somethng went wrong while saving data to a file."
          << "Maybe you are running out of disk space.";
      }

    }
  };


  SparseSavePoint(SparseIterators *its,
                  std::ostream *out,
                  index_t *offset,
                  index_t *cur_size,
                  std::vector<index_t> &sizes,
                  const std::string *delimeter) :
   its_(its), out_(out), offset_(offset),cur_size_(cur_size), sizes_(sizes), delimeter_(delimeter) 
  {}

  template<typename T>
  void operator()(T) {
    typename SparseIterators::template wrap<T>::Iterator_t &it = its_->template get<T>();
    typename SparsePoint<T>::Iterator nz_it;
    typename SparsePoint<T>::Iterator nz_it_begin = it->begin();
    typename SparsePoint<T>::Iterator nz_it_end = it->end();

    for (nz_it = nz_it_begin; nz_it != nz_it_end; ++nz_it) {
// This is deprecated we use only the compressed format since we can infer types from
// offset
//      boost::mpl::if_c <
//      IsSparseOnly_t::value ||
//      (boost::mpl::size<DenseTypeList_t>::value == 1
//       && boost::mpl::size<SparseTypeList_t>::value == 1),
//      CompressedFormatChoice,
//      MixedFormatChoice >::type::Print(*out_, nz_it->first, nz_it->second);
      CompressedFormatChoice::Print(*out_, nz_it->first+*offset_, nz_it->second);
      *out_ << *delimeter_;
      if (out_->fail()) {
        fl::logger->Die() << "Somethng went wrong while saving data to a file."
          << "Maybe you are running out of disk space.";
      }
    }
    (*offset_)+=sizes_[*cur_size_];
    *cur_size_+=1;
    ++it;
  }
private:
  SparseIterators *its_;
  std::ostream *out_;
  index_t *offset_;
  index_t *cur_size_;        
  std::vector<index_t> &sizes_; 
  const std::string *delimeter_;
};

struct MetaSaver  {
  MetaSaver(MetaDataIterator &it,
            index_t num_of_metadata,
            const std::string &delimeter,
            std::ostream *out) : it_(it),
      num_of_metadata_(num_of_metadata),
      delimeter_(delimeter)  {
    out_ = out;
  }

  ~MetaSaver() {
  }
  template<typename T>
  void operator()(T) {
    static const int ind=T::value;
    typedef typename boost::mpl::at_c<
      typename MetaDataType_t::TypeList_t, 
    ind>::type Type;
    if (ind < num_of_metadata_) {
      if (!boost::is_same<Type, signed char>::value &&  !boost::is_same<Type, unsigned char>::value) {
        *out_ << it_->template get<ind>() << delimeter_;
      } else {
        *out_ << static_cast<int>(it_->template get<ind>()) << delimeter_;
      }
      if (out_->fail()) {
        fl::logger->Die() << "Somethng went wrong while saving data to a file."
          <<"Maybe you are running out of disk space.";
      }
    }
  }

private:
  MetaDataIterator &it_;
  index_t num_of_metadata_;
  const std::string &delimeter_;
  std::ostream *out_;
};

#endif
