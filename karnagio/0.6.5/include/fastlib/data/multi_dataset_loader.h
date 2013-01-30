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
#ifndef FL_LITE_FASTLIB_DATA_MULTIDATASET_LOADER_H_
#define FL_LITE_FASTLIB_DATA_MULTIDATASET_LOADER_H_

struct DensePointLoader {
  DensePointLoader(DenseIterators *its,
                   std::deque<std::string> *tokens,
                   index_t *offset) {
    its_ = its;
    tokens_ = tokens;
    offset_ = offset;
  }
  ~DensePointLoader() {
  }
  template<typename T>
  void operator()(T) {
    typename DenseIterators::template wrap<T>::Iterator_t &it = DenseIterators::template get<T>(*its_);
    index_t len = it->size();
    // this does check if there is a length violation
    BOOST_ASSERT(static_cast<size_t>(len) == it->size());
    for (index_t i = 0; i < len; i++) {
      try {
        it->set(i, static_cast<T>(boost::lexical_cast<double>(tokens_->front())));
      }
      catch(const boost::bad_lexical_cast &e) {
        fl::logger->Die() << "There is something wrong in your file. "
          << "I found the following token (" 
          << tokens_->front()
          << ")that I cannot cast to a number";
      }
      tokens_->pop_front();
    }
    ++it;
    (*offset_) += len;
  }
private:
  DenseIterators *its_;
  std::deque<std::string> *tokens_;
  index_t *offset_;
};

struct DensePointLoaderFast {
  DensePointLoaderFast(DenseIterators *its,
                   const std::string &delimeter,
                   char** tok,
                   index_t *offset) :  
     its_(its), delimeter_(delimeter), tok_(tok), offset_(offset) {
  }
  ~DensePointLoaderFast() {
  }
  template<typename T>
  void operator()(T) {
    typename DenseIterators::template wrap<T>::Iterator_t &it = DenseIterators::template get<T>(*its_);
    index_t len = it->size();
    try {
      for (index_t i = 0; i < len; i++) {
        T value=static_cast<T>(strtod(*tok_, NULL));
        it->set(i, value);
        *tok_=strtok(NULL, delimeter_.c_str());
      }
    }
    catch(const boost::bad_lexical_cast &e) {
      fl::logger->Die() << "There is something wrong in your file. "
          << "I found the following token (" 
          << tok_
          << ")that I cannot cast to a number";
    }

    ++it;
    (*offset_) += len;
  }
private:
  DenseIterators *its_;
  const std::string &delimeter_;
  char** tok_;
  index_t *offset_;
};

struct SparsePointLoader {
  SparsePointLoader(bool is_2_tokens,
                    SparseIterators *its,
                    std::deque<std::string> *tokens,
                    index_t *offset) {
    is_2_tokens_ = is_2_tokens;
    its_ = its;
    tokens_ = tokens;
    offset_ = offset;
  }
  ~SparsePointLoader() {
  }
  // if the sparse point uses a map use this
  struct ContainerNullary1 {
    struct type {
      template<typename IteratorType, typename ElementType>
      static inline void set(IteratorType &it, index_t i, index_t offset, ElementType value) {
        try {
          it->set(i - offset, value);
        }
        catch(const std::bad_alloc &e) {
          fl::logger->Die() << "Failed to allocate memory. Most likely you are using "
            "a 32bit system and it cannot allocate more than 4GB";
        }
      }
      template<typename IteratorType>
      void SetSize(IteratorType &it, index_t size) {

      }
      template<typename IteratorType>
      static void Post(IteratorType &it) {

      }
    };
  };

  // Otherwise if it uses a vector use this one
  struct ContainerNullary2 {
    struct type {
      template<typename IteratorType, typename ElementType>
      static inline void set(IteratorType &it, index_t i, index_t offset, ElementType value) {
        try {
          it->elem_->push_back(std::make_pair(i - offset, value));
        }
        catch(const std::bad_alloc &e) {
          fl::logger->Die() << "Failed to allocate memory. Most likely you are using "
            "a 32bit system and it cannot allocate more than 4GB";
        }
      }
      template<typename IteratorType>
      void SetSize(IteratorType &it, index_t size) {
        try {
          it->reserve(size);
        }
        catch(const std::bad_alloc &e) {
          fl::logger->Die() << "Failed to allocate memory. Most likely you are using "
            "a 32bit system and it cannot allocate more than 4GB";
        }
      }
      template<typename IteratorType>
      static void Post(IteratorType &it) {
        std::sort(it->elem_->begin(), it->elem_->end());
      }
    };
  };

  template<typename T>
  void operator()(T) {
    typename SparseIterators::template wrap<T>::Iterator_t &it = SparseIterators::template get<T>(*its_);
    /*    if(boost::mpl::size<SparseTypeList_t>::type::value==1) {
          boost::mpl::eval_if<
            ContainerNullary1,
            ContainerNullary2
          >::type::SetSize(it, tokens_->size());
        }
    */
    std::vector<std::string> two_or_three_tokens;
    index_t ind0;
    index_t ind1;
    if (is_2_tokens_ == true) {
      ind0 = 0;
      ind1 = 1;
    }
    else {
      ind0 = 1;
      ind1 = 2;
    }
    // This should be the case. We can infer the the precision from their index
    // there is no need to use 3 tokens anymore
    if (is_2_tokens_ == true) {
      while (!tokens_->empty()) {
        if (tokens_->front().size() == 0) {
          tokens_->pop_front();
          continue;
        }
        // the tokens are coming in this format index:value
        two_or_three_tokens.clear();
        boost::algorithm::split(two_or_three_tokens, tokens_->front(),
                                boost::algorithm::is_any_of(":"));
        index_t ind=-1;
        try {
          ind = boost::lexical_cast<index_t>(two_or_three_tokens[ind0]);
        } 
        catch(const boost::bad_lexical_cast &e) {
          fl::logger->Die() << "Something went wrong while reading the file. "
            << "The following token ("
            << two_or_three_tokens[ind0]
            << ") can not be cast as an index ";
       
        }
        T value=0;
        try {
          value = static_cast<T>(boost::template lexical_cast<double>(two_or_three_tokens[ind1]));
        } 
        catch(const boost::bad_lexical_cast &e) {
          fl::logger->Die() << "Something went wrong while reading the file. "
            << "The following token ("
            << two_or_three_tokens[ind1]
            << ") can not be cast as a number ";
        }

        if (ind >= *offset_ + it->size()) {
          break;
        }
        BOOST_ASSERT(ind >= *offset_ && ind < *offset_ + it->size());
        boost::mpl::eval_if <
        boost::is_same <
        typename SparsePoint<T>::Container_t,
        std::map<index_t, T>
        > ,
        ContainerNullary1,
        ContainerNullary2
        >::type::set(it, ind, *offset_,  value);
        tokens_->pop_front();
      }
    }
    else {
      // This is deprecated, since we can now load sparse data without knowing their precision
      // the precision is infered from their index
      boost::algorithm::split(two_or_three_tokens, tokens_->front(),
                              boost::algorithm::is_any_of(":"));
      while (!tokens_->empty() && Typename<T>::Name() == two_or_three_tokens[0]) {
        if (tokens_->front().size() == 0) {
          tokens_->pop_front();
          continue;
        }
        index_t ind=-1;
        try {
          // the tokens are coming in this format precision:index:value
          ind = boost::template lexical_cast<index_t>(two_or_three_tokens[1]);
        }
        catch(const boost::bad_lexical_cast &e) {
          fl::logger->Die() << "Something went wrong while reading the file. "
            << "The following token ("
            << two_or_three_tokens[1]
            << ") can not be cast as an index ";
        }
        T value=0;
        try {
          value = static_cast<T>(boost::template lexical_cast<double>(two_or_three_tokens[2]));
        }
        catch(const boost::bad_lexical_cast &e) {
          fl::logger->Die() << "Something went wrong while reading the file. "
            << "The following token ("
            << two_or_three_tokens[2]
            << ") can not be cast as a number ";
        }


        BOOST_ASSERT(ind >= *offset_ && ind < *offset_ + it->size());
        boost::mpl::eval_if <
        boost::is_same <
        typename SparsePoint<T>::Container_t,
        std::map<index_t, T>
        > ,
        ContainerNullary1,
        ContainerNullary2
        >::type::set(it, ind, *offset_, value);

        tokens_->pop_front();
        two_or_three_tokens.clear();
        boost::algorithm::split(two_or_three_tokens, tokens_->front(), boost::algorithm::is_any_of(":"));
      }
    }
    boost::mpl::eval_if <
    boost::is_same <
    typename SparsePoint<T>::Container_t,
    std::map<index_t, T>
    > ,
    ContainerNullary1,
    ContainerNullary2
    >::type::Post(it);
    (*offset_) += it->size();
    ++it;
  }

private:
  std::deque<std::string> *tokens_;
  bool is_2_tokens_;
  SparseIterators *its_;
  index_t *offset_;
};

struct SparsePointLoaderFast {
  SparsePointLoaderFast(SparseIterators *its,
                    const std::string &delimeter,
                    char **tok,     
                    index_t *offset) :
      its_(its), delimeter_(delimeter), tok_(tok), offset_(offset) {
  }
  ~SparsePointLoaderFast() {
  }
  // if the sparse point uses a map use this
  struct ContainerNullary1 {
    struct type {
      template<typename IteratorType, typename ElementType>
      static inline void set(IteratorType &it, index_t i, index_t offset, ElementType value) {
        it->set(i - offset, value);
      }
      template<typename IteratorType>
      void SetSize(IteratorType &it, index_t size) {

      }
      template<typename IteratorType>
      static void Post(IteratorType &it) {

      }
    };
  };

  // Otherwise if it uses a vector use this one
  struct ContainerNullary2 {
    struct type {
      template<typename IteratorType, typename ElementType>
      static inline void set(IteratorType &it, index_t i, index_t offset, ElementType value) {
        it->elem_->push_back(std::make_pair(i - offset, value));
      }
      template<typename IteratorType>
      void SetSize(IteratorType &it, index_t size) {
        it->reserve(size);
      }
      template<typename IteratorType>
      static void Post(IteratorType &it) {
        std::sort(it->elem_->begin(), it->elem_->end());
      }
    };
  };

  template<typename T>
  void operator()(T) {
    typename SparseIterators::template wrap<T>::Iterator_t &it = SparseIterators::template get<T>(*its_);
    /*    if(boost::mpl::size<SparseTypeList_t>::type::value==1) {
          boost::mpl::eval_if<
            ContainerNullary1,
            ContainerNullary2
          >::type::SetSize(it, tokens_->size());
        }
    */

    while (true) {
      if (*tok_==NULL) {
        break;        
      }
      // the tokens are coming in this format index:value
      char *rem;
      index_t ind = static_cast<index_t>(strtod(*tok_, &rem));
      T value = static_cast<T>(strtod(rem+1, NULL));
      
      if (ind >= *offset_ + it->size()) {
        break;
      }
      BOOST_ASSERT(ind >= *offset_ && ind < *offset_ + it->size());
      boost::mpl::eval_if <
      boost::is_same <
      typename SparsePoint<T>::Container_t,
      std::map<index_t, T>
      > ,
      ContainerNullary1,
      ContainerNullary2
      >::type::set(it, ind, *offset_,  value);
      *tok_=strtok(NULL, delimeter_.c_str());
    }

    boost::mpl::eval_if <
    boost::is_same <
    typename SparsePoint<T>::Container_t,
    std::map<index_t, T>
    > ,
    ContainerNullary1,
    ContainerNullary2
    >::type::Post(it);
    (*offset_) += it->size();
    ++it;
  }

  private:
    SparseIterators *its_;
    const std::string &delimeter_;
    char **tok_;     
    index_t *offset_;
};

struct MetaLoader  {
  MetaLoader(MetaDataIterator &it,
             const index_t num_of_metadata_to_load,
             std::deque<std::string> *tokens) : 
               it_(it), 
               largest_meta_(num_of_metadata_to_load), 
               tokens_(tokens) {
  }

  ~MetaLoader() {
  }
  template<typename T>
  void operator()(T) {
    static const int ind=T::value;
    try {
      if (largest_meta_>ind) {
        it_->template get<ind>() =
          static_cast <
            typename boost::mpl::at_c <
              typename MetaDataType_t::TypeList_t,
              ind
            >::type
          >(boost::template lexical_cast<double>((*tokens_)[0]));
        tokens_->pop_front();
      }
    }
    catch(const boost::bad_lexical_cast &e) {
      fl::logger->Die() << "Something went wrong while reading the file. "
          << "The following token in the meta data ("
          << (*tokens_)[0]
          << ") can not be cast as a number ";
    }
  }

  private:
    MetaDataIterator &it_;
    const index_t largest_meta_;
    std::deque<std::string> *tokens_;
};

struct MetaLoaderFast  {
  MetaLoaderFast(MetaDataIterator &it,
             const std::string &delimeter,
             const index_t num_of_metadata_to_load,
             char **tok) : 
    it_(it),
    delimeter_(delimeter),
    largest_meta_(num_of_metadata_to_load),
    tok_(tok)
 {
  }

  ~MetaLoaderFast() {
  }
  template<typename T>
  void operator()(T) {
    static const int ind=T::value;
    if (largest_meta_>ind) {
      it_->template get<ind>() =
        static_cast <
          typename boost::mpl::at_c <
            typename MetaDataType_t::TypeList_t,
            ind
           >::type
         >(strtod(*tok_, NULL));
    }
    *tok_=strtok(NULL, delimeter_.c_str());
  }

  private:
    MetaDataIterator &it_;
    const std::string &delimeter_;
    const index_t largest_meta_;
    char** tok_;
};

#endif
