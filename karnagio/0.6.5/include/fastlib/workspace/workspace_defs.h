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

#ifndef  INCLUDE_FASTLIB_WORKSPACE_WORKSPACE_DEFS_H_
#define  INCLUDE_FASTLIB_WORKSPACE_WORKSPACE_DEFS_H_
#include "workspace.h"
#include "boost/algorithm/string/predicate.hpp"
#include "boost/lexical_cast.hpp"

namespace fl { namespace ws {

  template<typename TableType>
  void WorkSpace::LoadTable(const std::string &name, boost::shared_ptr<TableType> table) {
    boost::mutex* mutex;
    global_mutex_.lock();
    if (mutex_map_.count(name)==0) {
      mutex=new boost::mutex();
      mutex_map_[name].reset(mutex);
    } else {
      mutex=mutex_map_[name].get();
    }
    mutex->try_lock();
    var_map_[name]=table;
    mutex->unlock();
    global_mutex_.unlock(); 
  }

  template<typename TableSetType>
  void WorkSpace::LoadFromFile(const std::string &name,
      const std::string &filename) {
    FL_SCOPED_LOG(LoadFromFile);
    boost::mutex* mutex;
    global_mutex_.lock();
    if (mutex_map_.count(name)==0) {
      mutex=new boost::mutex();
      mutex_map_[name].reset(mutex);
    } else {
      mutex=mutex_map_[name].get();
    }
    mutex->try_lock();
    global_mutex_.unlock();
    bool success=false;
    boost::mpl::for_each<TableSetType>(LoadMeta(name, filename,  
          mutex, global_mutex_, &var_map_, &success));
    mutex->unlock();
    if (success==false) {
      fl::logger->Die()<< "Failed to load "<< filename<<" unsupported type";
    }
  }

  template<typename TableType>
  void WorkSpace::LoadMeta::operator()(TableType&) {
    if (*success_==true) {
      return;
    }
    try {
      boost::shared_ptr<TableType> table(
        new TableType());
      table->data()->TryToInit(filename_);
      table->Init(filename_, "r");
      global_mutex_.lock();
      var_map_->operator[](name_)=table;
      global_mutex_.unlock();
      *success_=true;
      fl::logger->Debug()<<"File: "<<filename_<<" was loaded in the workspace"
       <<std::endl; 
    }
    catch(const fl::TypeException &e) {
      
    }
  }


  template<typename TableType>
  void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<TableType> *table) {

    boost::mutex* mutex;
    global_mutex_.lock();
    if (mutex_map_.count(name)==0) {
      mutex=new boost::mutex();
      mutex_map_[name].reset(mutex);
      mutex->lock();
    } else {
      mutex=mutex_map_[name].get();      
    }
    global_mutex_.unlock();
    mutex->lock();
    try {
      global_mutex_.lock();
      if (var_map_.count(name)==0) {
        fl::logger->Die()<<"Detected unsafe deletion of table ("
          <<name<<")";
      }
      *table=boost::any_cast<boost::shared_ptr<TableType> >(var_map_[name]);
      global_mutex_.unlock();
    }
    catch(const boost::bad_any_cast &e) {
      mutex->unlock();
      global_mutex_.unlock();
      fl::logger->Die()<<"Failed to attach ("<<name<<"), There seems to be "
        "a bad type conversion";
    }
    mutex->unlock();
  }

  template<typename TableType>
  void WorkSpace::TryToAttach(const std::string &name) {
    boost::mutex* mutex;
    global_mutex_.lock();
    bool is_name_in_map=mutex_map_.count(name)==0;
    global_mutex_.unlock();
    if (is_name_in_map) {
      mutex=new boost::mutex();
      global_mutex_.lock();
      mutex_map_[name].reset(mutex);
      global_mutex_.unlock();
      mutex->lock();
    } else {
      global_mutex_.lock();
      mutex=mutex_map_[name].get();
      global_mutex_.unlock();      
    }
    mutex->lock();
    try {
      boost::shared_ptr<TableType> table;
      global_mutex_.lock();
      table=boost::any_cast<boost::shared_ptr<TableType> >(var_map_[name]);
      global_mutex_.unlock();
    }
    catch(const boost::bad_any_cast &e) {
      mutex->unlock();
      global_mutex_.unlock();
      throw fl::TypeException();  
    }
    mutex->unlock();
  }

  template<typename TableType>
  void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<TableType> *table) {
    boost::mutex* mutex;
    global_mutex_.lock();
    bool  has_name=var_map_.count(name)!=0;
    global_mutex_.unlock();
    if (has_name) {
      fl::logger->Die()<<"Workspace ("+workspace_name_+") already contains "
        "variable: "+name;
    }
    global_mutex_.lock();
    if (mutex_map_.count(name)!=0) {
      mutex=mutex_map_[name].get();
    } else {
      mutex=new boost::mutex();
      mutex_map_[name].reset(mutex);
    } 
    mutex->try_lock();
    var_map_[name]=boost::shared_ptr<TableType>(new TableType());
    boost::any_cast<boost::shared_ptr<TableType> >(var_map_[name])->Init(name, 
          dense_sizes, sparse_sizes,
          num_of_points);
    *table=boost::any_cast<boost::shared_ptr<TableType> >(var_map_[name]);
    global_mutex_.unlock();
  }
 
  template<int Index,typename TableType1, typename TableType2, typename TableType3>
  void WorkSpace::TieLabels(boost::shared_ptr<TableType1> table,
        boost::shared_ptr<TableType2> labels, 
        const std::string &new_name,
        boost::shared_ptr<TableType3> *new_table) {

    this->Attach(new_name,
        table->dense_sizes(),
        table->sparse_sizes(),
        table->n_entries(),
        new_table);
    
    typename TableType1::Point_t point1;
    typename TableType2::Point_t point2;
    typename TableType3::Point_t point3;
    for(index_t i=0; i<table->n_entries(); ++i) {
      table->get(i, &point1);
      labels->get(i, &point2);
      (*new_table)->get(i, &point3);
      point3.CopyValues(point1);
      point3.meta_data(). template get<Index>()=point2[0];
    }
  }
     
  template<typename TableType1, typename TableType2>
  void WorkSpace::CopyAndDestruct(boost::shared_ptr<TableType1> table1,
        boost::shared_ptr<TableType2> *table2) {

    
    var_map_[table1->filename()]=boost::shared_ptr<TableType2>(new TableType2());
    boost::any_cast<boost::shared_ptr<
      TableType2> >(var_map_[table1->filename()])->Init(table1->filename(), 
          table1->dense_sizes(), table1->sparse_sizes(),
          table1->n_entries());
    *table2=boost::any_cast<
      boost::shared_ptr<TableType2> >(var_map_[table1->filename()]);   
    typename TableType1::Point_t point1;
    typename TableType2::Point_t point2;
    for(index_t i=0; i<table1->n_entries(); ++i) {
      table1->get(i, &point1);
      (*table2)->get(i, &point2);
      point2.CopyValues(point1);
    }
  }

}}

#endif

