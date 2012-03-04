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

#ifndef PAPERBOAT_HPCC_SRC_WORKSPACE_DEV_H_
#define PAPERBOAT_HPCC_SRC_WORKSPACE_DEV_H_
#include "workspace/workspace.h"
#include "fastlib/util/string_utils.h"

namespace fl {namespace hpcc{

  template<typename T>
  void WorkSpace::LoadDenseHPCCDataSet(const std::string &name,
      index_t n_attributes,
      index_t n_entries,
      const void *in_data) {
    typedef typename boost::mpl::at<
      DenseTables_t,
      typename T::Value_t
    >::type Table_t;
    boost::shared_ptr<Table_t> table;
    this->Attach(name,
        std::vector<index_t>(1, n_attributes),
        std::vector<index_t>(),
        n_entries,
        &table); 
    index_t counter=0;
    typename Table_t::Point_t point;
    for(index_t i=0; i<n_entries; ++i) {
      table->get(i, &point);
      point.meta_data().template get<2>()=T((char*)in_data+counter).id();
      for(index_t j=0; j<n_attributes; ++j) {
         T datum((char*)in_data+counter);
         double val=datum.value();
         point.set(j, val);
         counter+=datum.size();
      }
    }
    this->Purge(name);
    this->Detach(name);
  }

  template<typename T>
  void WorkSpace::ExportDenseHPCCDataSet(const std::string &name,
      void **in_data, uint32 *length) {
    typedef typename boost::mpl::at<
      DenseTables_t,
      typename T::Value_t
    >::type Table_t;
    boost::shared_ptr<Table_t> table;
    this->Attach(name, &table);
    *length=table->n_entries() 
            * table->n_attributes()
            * T::size();
    *in_data=rtlMalloc(*length);
    typedef typename Table_t::Point_t Point_t;
    Point_t point;
    index_t counter=0;
    int id=0;
    for(index_t i=0; i<table->n_entries(); ++i) {
      table->get(i, &point);
      if (point.meta_data().template get<2>()==0) {
        id=point.meta_data().template get<2>();
      } else {
        id++;
      }
      std::cout<<"id="<<id<<std::endl;
      for(typename Point_t::iterator it=point.begin();
          it!=point.end(); ++it) {
        T datum((char*)(*in_data)+counter);  
        datum.id()=id;
        datum.value()=it.value();
        datum.number()=it.attribute();
        counter+=datum.size();
      }
    }
  }
  
  /**
   *  @brief it requires the input dataset to be sorted 
   *  according to file_id, id, number
   *  use ECL SORT(mydataset, file_id, id, number, LOCAL);
   *  The arguments will contain input files with the _in suffix
   *  The filenames must be of the form storage:number
   *  where storage is either dense or sparse. The number is 
   *  between 0-255
   */
  template<typename T>
  void WorkSpace::LoadAllDenseHPCCDataSets(const std::string &arguments,
      const char *in_data,
      const uint64 data_len) {
    typedef typename boost::mpl::at<
      DenseTables_t,
      typename T::Value_t
    >::type Table_t;
    std::vector<std::string> tokens=fl::SplitString(arguments, " ");
    std::vector<std::pair<std::string, uint8> > infiles;
    for(size_t i=0; i<tokens.size(); ++i) {
      std::vector<std::string> toks=fl::SplitString(tokens[i], "=");
      if (fl::StringEndsWith(toks[0], "_in")) {
        std::string file_name=toks[1];
        std::vector<std::string> three_tokens;
        three_tokens=fl::SplitString(toks[1], "$");
        if (three_tokens.size()!=3) {
          fl::logger->Die()<<"Dataset name must be of the form "
            " storage:precision:number, for example: "
            " dense$double$3 or sparse$bool$5";
        }
        std::string storage=three_tokens[0];
        std::string type=three_tokens[1];
        std::string name=three_tokens[2];
        if (storage!="dense") {
          continue;
        }
        if (type!=fl::data::Typename<typename T::Value_t>::Name()) {
          continue;
        }
        infiles.push_back(std::make_pair(toks[1], 
             (uint8)boost::lexical_cast<int>(name))); 
      }
    } 
    if (infiles.empty()) {
      return;
    }
    // uint8 -> file_id
    // std::set<uint64> -> set point ids
    // index_t -> n_attributes
    std::map<uint8, std::pair<std::set<uint64>, index_t> > file_dims;
    for(size_t i=0; i<infiles.size(); ++i) {
      file_dims[infiles[i].second].second=0;
    }
    uint64 counter=0;
    while (counter<data_len) {
      T datum(const_cast<char*>(in_data)+counter);
      uint8 file_id=datum.file_id();
      if (file_dims.count(file_id)==0) {
        counter+=T::size();
        continue;
      }
      file_dims[file_id].first.insert(datum.id());
      file_dims[file_id].second=std::max(file_dims[file_id].second,
         static_cast<index_t>(datum.number())); 
      counter+=T::size();
    }
    std::map<uint8, boost::shared_ptr<Table_t> > tables;
    for(size_t i=0; i<infiles.size(); ++i) {
      tables[infiles[i].second].reset(new Table_t());
      this->Attach(infiles[i].first,
          std::vector<index_t>(1, file_dims[infiles[i].second].second+1),
          std::vector<index_t>(),
          file_dims[infiles[i].second].first.size(),
          &tables[infiles[i].second]);
    }
    counter=0;
    while (counter<data_len) {
      T datum(const_cast<char*>(in_data)+counter);
      uint8 file_id=datum.file_id();
      if (file_dims.count(file_id)==0) {
        counter+=T::size();
        continue;
      }
      boost::shared_ptr<Table_t> table=tables[file_id];
      typename Table_t::Point_t point;
      for(index_t i=0; i<table->n_entries(); ++i) {
        T datum(const_cast<char*>(in_data)+counter);
        table->get(i, &point);
        point.meta_data().template get<2>()=datum.id();
        for(index_t j=0; j<point.size(); ++j) {
          T datum(const_cast<char*>(in_data)+counter);
          point.set(j, datum.value());
          counter+=T::size();
        }
      }
    }
    for(size_t i=0; i<infiles.size(); ++i) {
      this->Purge(infiles[i].first);
      this->Detach(infiles[i].first);
    }
  }
  
  /**
   *  @brief it requires the input dataset to be sorted 
   *  according to file_id, id, number
   *  use ECL SORT(mydataset, file_id, id, number, LOCAL);
   *  The arguments will contain input files with the _in suffix
   *  The filenames must be of the form storage:number
   *  where storage is either dense or sparse. The number is 
   *  between 0-255
   */
  template<typename T>
  void WorkSpace::LoadAllSparseHPCCDataSets(const std::string &arguments,
      const char *in_data,
      const uint64 data_len) {

    typedef typename boost::mpl::at<
      SparseTables_t,
      typename T::Value_t
    >::type Table_t;
    std::vector<std::string> tokens=fl::SplitString(arguments, " ");
    std::vector<std::pair<std::string, uint8> > infiles;
    for(size_t i=0; i<tokens.size(); ++i) {
      std::vector<std::string> toks=fl::SplitString(tokens[i], "=");
      if (fl::StringEndsWith(toks[0], "_in")) {
        std::string file_name=toks[1];
        std::vector<std::string> three_tokens;
        three_tokens=fl::SplitString(toks[1], "$");
        if (three_tokens.size()!=3) {
          fl::logger->Die()<<"Dataset name must be of the form "
            " storage:precision:number, for example: "
            " dense$double$3 or sparse$bool$5";
        }
        std::string storage=three_tokens[0];
        std::string type=three_tokens[1];
        std::string name=three_tokens[2];
        if (storage!="sparse") {
          continue;
        }
        if (type!=fl::data::Typename<typename T::Value_t>::Name()) {
          continue;
        }
        infiles.push_back(std::make_pair(toks[1], 
             (uint8)boost::lexical_cast<int>(name))); 
      }
    } 
    if (infiles.empty()) {
      return;
    }
    // uint8 -> file_id
    // std::set<uint64> -> set point ids
    // index_t -> n_attributes
    std::map<uint8, std::pair<std::set<uint64>, index_t> > file_dims;
    for(size_t i=0; i<infiles.size(); ++i) {
      file_dims[infiles[i].second].second=0;
    }
    uint64 counter=0;
    while (counter<data_len) {
      T datum(const_cast<char*>(in_data)+counter);
      uint8 file_id=datum.file_id();
      if (file_dims.count(file_id)==0) {
        continue;
      }
      file_dims[file_id].first.insert(datum.id());
      file_dims[file_id].second=std::max(file_dims[file_id].second,
         static_cast<index_t>(datum.number())); 
      counter+=T::size();
    }
    std::map<uint8, boost::shared_ptr<Table_t> > tables;
    for(size_t i=0; i<infiles.size(); ++i) {
      tables[infiles[i].second].reset(new Table_t());
      this->Attach(infiles[i].first,
          std::vector<index_t>(1, file_dims[infiles[i].second].second+1),
          std::vector<index_t>(),
          file_dims[infiles[i].second].first.size(),
          &tables[infiles[i].second]);
    }
    counter=0;
    while (counter<data_len) {
      T datum(const_cast<char*>(in_data)+counter);
      uint8 file_id=datum.file_id();
      if (file_dims.count(file_id)==0) {
        counter+=T::size();
        continue;
      }
      boost::shared_ptr<Table_t> table=tables[file_id];
      typename Table_t::Point_t point;
      for(index_t i=0; i<table->n_entries(); ++i) {
        T datum(const_cast<char*>(in_data)+counter);
        table->get(i, &point);
        uint16 current_number=datum.number();
        uint16 old_number=current_number;
        std::vector<std::pair<index_t,double> > elements;
        while (current_number==old_number) {
          elements.push_back(std::make_pair(datum.number(), 
              datum.value()));
          old_number=datum.number();
          counter+=T::size();
          datum.set_ptr(const_cast<char*>(in_data)+counter);  
          current_number=datum.number();
        }            
      }
    }
    for(size_t i=0; i<infiles.size(); ++i) {
      this->Purge(infiles[i].first);
      this->Detach(infiles[i].first);
    }
  }
   
 /**
  *  @brief     
  *  The arguments will contain input files with the _out suffix
  *  The filenames must be of the form storage:number
  *  where storage is either dense or sparse. The number is 
  *  between 0-255
  */
  template<typename T>
  void WorkSpace::ExportAllDenseHPCCDataSets(const std::string &arguments,
      void **out_data,
      unsigned int *data_len) {
    char** out_data1=reinterpret_cast<char**>(out_data);
    typedef typename boost::mpl::at<
      DenseTables_t,
      typename T::Value_t
    >::type Table_t;
    std::vector<std::string> tokens=fl::SplitString(arguments, " ");
    std::vector<std::pair<std::string, uint8> > outfiles;
    for(size_t i=0; i<tokens.size(); ++i) {
      std::vector<std::string> toks=fl::SplitString(tokens[i], "=");
      if (fl::StringEndsWith(toks[0], "_out")) {
        std::string file_name=toks[1];
        std::vector<std::string> three_tokens;
        three_tokens=fl::SplitString(toks[1], "$");
        if (three_tokens.size()!=3) {
          fl::logger->Die()<<"Dataset name must be of the form "
            " storage:precision:number, for example: "
            " dense:$double$3 or sparse$bool$5";
        }
        std::string storage=three_tokens[0];
        std::string type=three_tokens[1];
        std::string name=three_tokens[2];
        if (storage!="dense") {
          continue;
        }
        if (type!=fl::data::Typename<typename T::Value_t>::Name()) {
          continue;
        }
        outfiles.push_back(std::make_pair(file_name, 
              (uint8)boost::lexical_cast<int>(name))); 
      }
    } 
    if (outfiles.empty()) {
      return;
    }
    // Find the total elements from all the matrices
    uint64 total_elements=0;
    for(size_t i=0; i<outfiles.size(); ++i) {
      boost::shared_ptr<DefaultTable_t> table;
      this->Attach(outfiles[i].first, &table);
      total_elements+=table->n_entries()*table->n_attributes();
    }
    *data_len=total_elements*T::size();
    *out_data1=(char*)rtlMalloc(*data_len);
    index_t counter=0;
    T datum(*out_data1+counter);
    for(size_t i=0; i<outfiles.size(); ++i) {
      boost::shared_ptr<Table_t> table;
      this->Attach(outfiles[i].first, &table);
      uint8 file_id=outfiles[i].second;
      typename Table_t::Point_t point;
      index_t id=-1;
      for(index_t j=0; j<table->n_entries(); ++j) {
        table->get(j, &point);
        if (point.meta_data(). template get<2>()==0) {
          id++;
        } else {
          id=point.meta_data(). template get<2>();
        }
        for(index_t k=0; k<point.size(); ++k) {
          datum.set_ptr(*out_data1+counter);
          datum.file_id()=file_id;
          datum.id()=id;
          datum.value()=point[k];
          datum.number()=k;
          counter+=T::size();
        }
      }
    }
  }
  
  /**
  *  @brief     
  *  The arguments will contain input files with the _out suffix
  *  The filenames must be of the form storage:number
  *  where storage is either dense or sparse. The number is 
  *  between 0-255
  */
  template<typename T>
  void WorkSpace::ExportAllSparseHPCCDataSets(const std::string &arguments,
      void **out_data,
      uint64 *data_len) {
    char** out_data1=static_cast<char**>(out_data);
    typedef typename boost::mpl::at<
      SparseTables_t,
      typename T::Value_t
    >::type Table_t;
    std::vector<std::string> tokens=fl::SplitString(arguments, " ");
    std::vector<std::pair<std::string, uint8> > outfiles;
    for(size_t i=0; i<tokens.size(); ++i) {
      std::vector<std::string> toks=fl::SplitString(tokens[i], "=");
      if (fl::StringEndsWith(toks[0], "_out")) {
        std::string file_name=toks[1];
        std::vector<std::string> three_tokens;
        three_tokens=fl::SplitString(toks[1], "$");
        if (three_tokens.size()!=3) {
          fl::logger->Die()<<"Dataset name must be of the form "
            " storage:precision:number, for example: "
            " dense:double:3 or sparse:bool:5";
        }
        std::string storage=three_tokens[0];
        std::string type=three_tokens[1];
        std::string name=three_tokens[2];
        if (storage!="sparse") {
          continue;
        }
        if (type!=fl::data::Typename<typename T::Value_t>::Name()) {
          continue;
        }
        outfiles.push_back(std::make_pair(file_name, 
              (uint8)boost::lexical_cast<int>(name)));       }
    } 
    if (outfiles.empty()) {
      return;
    }
    uint64 total_elements;
    for(size_t i=0; i<outfiles.size(); ++i) {
      boost::shared_ptr<Table_t> table;
      this->Attach(outfiles[i].first, &table);
      for(index_t j=0; j<table->n_entries(); ++j) {
        typename Table_t::Point_t point;
        table->get(j, &point);
        for(typename Table_t::Point_t::iterator 
            it=point.begin(); it!=point.end(); ++it) {
          total_elements++;
        }   
      }
    }
    *data_len=total_elements*T::size();
    *out_data1=rtlMalloc(*data_len);
    index_t counter=0;
    T datum(*out_data1+counter);
    for(size_t i=0; i<outfiles.size(); ++i) {
      boost::shared_ptr<Table_t> table;
      this->Attach(outfiles[i].first, &table);
      uint8 file_id=outfiles[i].second;
      typename Table_t::Point_t point;
      int id=-1;
      for(index_t j=0; j<table->n_entries(); ++j) {
        table->get(j, &point);
        if (point.meta_data(). template get<2>()==0) {
          id++;
        } else {
          id=point.meta_data(). template get<2>();
        }
        for(typename Table_t::Point_t::iterator 
            it=point.begin(); it!=point.end(); ++it) {
          datum.set_ptr(*out_data1+counter);
          datum.set_id()=file_id;
          datum.id()=id;
          datum.value()=it->value();
          datum.number()=it->attribute();
          counter+=T::size();
        }
      }
    }
  }

}}

#endif
