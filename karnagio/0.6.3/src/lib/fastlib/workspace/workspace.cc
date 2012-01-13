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

#include "fastlib/table/table_dev.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/workspace/workspace_defs.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "boost/algorithm/string/predicate.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/bind.hpp"

namespace fl { namespace ws {
  WorkSpace::WorkSpace() {
    temp_var_counter_=0;
    temp_var_prefix_="ismion_temp_vars_";
    schedule_mode_=0;
    //
  }
  WorkSpace::~WorkSpace() {
    if (schedule_mode_==0) {
      pool_->wait();
    } else {
      if (schedule_mode_==1) {
        for(size_t i=0; i<vector_pool_.size(); ++i) {
          vector_pool_[i]->join();
        }     
      }
    }
    std::map<std::string, boost::shared_ptr<boost::mutex> >::iterator it;
    for(it=mutex_map_.begin(); it!=mutex_map_.end(); ++it) {
      // we need to make sure that all the mutexes are unlocked
      it->second->try_lock();
      it->second->unlock();
      //delete it->second;
    }
    if (fl::global_exception) {
        boost::rethrow_exception(fl::global_exception);
    }
  }

  void WorkSpace::LoadAllTables(const std::vector<std::string> &args) {
    for(unsigned int i=0; i<args.size(); ++i) {
      if (boost::algorithm::starts_with(args[i], "--") &&
          boost::algorithm::contains(args[i], "_in=")) {
        std::vector<std::string> tokens;
        boost::algorithm::split(tokens, args[i],
            boost::algorithm::is_any_of("="));
        // check if it contains more than one filenames
        if (boost::algorithm::contains(tokens[1], ":")==false
            && boost::algorithm::contains(tokens[1], ",")==false) {
          std::string filename=tokens[1];
          std::string variable=filename;
          if (boost::algorithm::contains(tokens[0], "references")||
             boost::algorithm::contains(tokens[0], "queries")) {
            LoadFromFile<DataTables_t>(variable, filename);
          } else {
            LoadFromFile<ParameterTables_t>(variable, filename);
          }
        } else {
          std::vector<std::string> filenames;
          boost::algorithm::split(filenames, args[i], 
             boost::algorithm::is_any_of(":,"));
          for(unsigned int i=0; i<filenames.size(); ++i) {
            std::string variable(filenames[i]);
            if (boost::algorithm::contains(tokens[0], "references")||
              boost::algorithm::contains(tokens[0], "queries")) {
              LoadFromFile<DataTables_t>(variable, filenames[i]);
            } else {
              LoadFromFile<ParameterTables_t>(variable, filenames[i]);
            }
          }
        }
      }
    }
  }

  void WorkSpace::IndexAllReferencesQueries(
      const std::vector<std::string> &args) {
    std::string metric("l2");
    std::string metric_args;
    int leaf_size=20;
    for(unsigned int i=0; i<args.size(); ++i) {
      if (boost::algorithm::contains(args[i],"metric=")) {
        std::vector<std::string> tokens;
        boost::algorithm::split(tokens, args[i],
            boost::algorithm::is_any_of("="));
        metric=tokens[1];
        if (metric=="l2") {
          break;
        }
      } else {
        if (boost::algorithm::contains(args[i],"metric_weights_in=")) {
          std::vector<std::string> tokens;
          boost::algorithm::split(tokens, args[i],
              boost::algorithm::is_any_of("="));
          metric_args=tokens[1];
        } else {
          if (boost::algorithm::contains(args[i],"leaf_size=")) {
            std::vector<std::string> tokens;
            boost::algorithm::split(tokens, args[i],
              boost::algorithm::is_any_of("="));
            leaf_size=boost::lexical_cast<int>(tokens[1]);         
          }
        }
      }  
    }
    for(unsigned int i=0; i<args.size(); ++i) {
      if (boost::algorithm::contains(args[i], "references_in=") ||
         boost::algorithm::contains(args[i], "queries_in=")) {
        std::vector<std::string> tokens;
        boost::algorithm::split(tokens, args[i],
            boost::algorithm::is_any_of("="));
        // check if it contains more than one filenames
        if (boost::algorithm::contains(tokens[1], ":")==false
            && boost::algorithm::contains(tokens[1], ",")==false) {
          std::string filename=tokens[1];
          std::string variable=filename;
          IndexTable(variable, metric, metric_args, leaf_size);
        } else {
          std::vector<std::string> filenames;
          boost::algorithm::split(filenames, args[i], 
             boost::algorithm::is_any_of(":,"));
          for(unsigned int i=0; i<filenames.size(); ++i) {
            std::string variable(filenames[i]);
            IndexTable(variable, metric, metric_args, leaf_size);
          }
        }
      }
    }
  }

  void WorkSpace::ExportAllTablesTask(
      const std::vector<std::string> &args) {
    for(unsigned int i=0; i<args.size(); ++i) {
      if (boost::algorithm::contains(args[i], "_out=")) {
        std::vector<std::string> tokens;
        boost::algorithm::split(tokens, args[i],
            boost::algorithm::is_any_of("="));
        // check if it contains more than one filenames
        if (boost::algorithm::contains(tokens[1], ":")==false
            && boost::algorithm::contains(tokens[1], ",")==false) {
          std::string filename=tokens[1];
          std::string variable=filename;
          ExportToFile(variable, filename);
        } else {
          std::vector<std::string> filenames;
          boost::algorithm::split(filenames, args[i], 
             boost::algorithm::is_any_of(":,"));
          for(unsigned int i=0; i<filenames.size(); ++i) {
            std::string variable(filenames[i]);
            ExportToFile(variable, filenames[i]);
          }
        }
      }
    }
  }

  void WorkSpace::ExportAllTables(
      const std::vector<std::string> &args) {
    this->schedule(boost::bind(&WorkSpace::ExportAllTablesTask, this, args));
  }

  void WorkSpace::IndexTable(const std::string &variable, 
        const std::string &metric,
        const std::string &metric_args,
        const int leaf_size) {
    boost::mutex* mutex;
    global_mutex_.lock();
    bool is_name_in_map=mutex_map_.count(variable)==0;
    global_mutex_.unlock();
    if (is_name_in_map) {
      mutex=new boost::mutex();
      global_mutex_.lock();
      mutex_map_[variable].reset(mutex);
      global_mutex_.unlock();
      mutex->lock();
    } else {
      global_mutex_.lock();
      mutex=mutex_map_[variable].get();
      global_mutex_.unlock();
    }
    
    mutex->lock();
    bool success=false;
    boost::mpl::for_each<DataTables_t>(IndexMeta(this,
            &var_map_, 
            variable,
            metric,
            metric_args,
            leaf_size,
            &success));
    mutex->unlock();
  }

  template<typename TableType>
  void WorkSpace::IndexMeta::operator()(TableType&) {
    if (*success_==true) {
      return;
    }
    boost::shared_ptr<TableType> table;
    try {
      table=boost::any_cast<boost::shared_ptr<TableType> >(
          var_map_->operator[](variable_));
      *success_=true;
    }
    catch(const boost::bad_any_cast &e) {
      return; 
    }
    if (metric_=="l2") {
      typename TableType::template IndexArgs<fl::math::LMetric<2> > index_args;
      index_args.leaf_size = leaf_size_;
      if (table->is_indexed()==false) {
        table->IndexData(index_args);
      }
    } else {
      if (metric_=="weighted_l2") {
        typename TableType::template IndexArgs<fl::math::WeightedLMetric < 2,
        fl::data::MonolithicPoint<double> > > w_index_args;
        typename DefaultTable_t::Point_t weight_point;
        boost::shared_ptr<DefaultTable_t> metric_weights;
        ws_->Attach(metric_args_, &metric_weights);
        metric_weights->get(0, &weight_point);
        w_index_args.metric.set_weights(weight_point.template dense_point<double>());
        w_index_args.leaf_size = leaf_size_;
        if (table->is_indexed()==false) {
          table->IndexData(w_index_args);
        }
      } else {
        fl::logger->Die() << "Unknown metric ("<<metric_<<")";     
      }
    }
  }

  void WorkSpace::ExportToFile(const std::string &name, const std::string &filename) {
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
    bool success=false;
    boost::mpl::for_each<ParameterTables_t>(SaveMeta(&var_map_, 
          name, filename, &success));
    boost::mpl::for_each<DataTables_t>(SaveMeta(&var_map_, 
          name, filename, &success));
    if (success==false) {
      fl::logger->Die()<<"Cannot save table ("<<name<<"), the table "
        "type is not supported"; 
    }
    global_mutex_.lock();
    mutex->unlock();
    global_mutex_.unlock();
  }

  bool WorkSpace::IsTableAvailable(const std::string &name) {
    global_mutex_.lock();
    bool is_name_in_map=mutex_map_.count(name)==0;
    if (is_name_in_map) {
      global_mutex_.unlock();
      return false;
    } else {
      boost::mutex* mutex;
      mutex=mutex_map_[name].get();
      global_mutex_.unlock();
      bool lock_success=mutex->try_lock();
      if (lock_success==true) {
        mutex->unlock();
      }
      return lock_success;
    }
  }

  template<typename TableType>
  void WorkSpace::SaveMeta::operator()(TableType&) {
    if (*success_==true) {
      return;
    }
    try {
      boost::any_cast<boost::shared_ptr<TableType> >((*var_map_)[name_])->filename()=filename_;
      boost::any_cast<boost::shared_ptr<TableType> >((*var_map_)[name_])->Save();
      *success_=true;
    }
    catch(const boost::bad_any_cast &e) {
    
    }
  }

  void WorkSpace::LoadDataTableFromFile(const std::string &name,
      const std::string &filename) {
    LoadFromFile<DataTables_t>(name, filename);

  }

  void WorkSpace::LoadParameterTableFromFile(const std::string &name,
      const std::string &filename) {
    LoadFromFile<ParameterTables_t>(name, filename);
  }

  void WorkSpace::Detach(const std::string  &table_name) {
  }

  void WorkSpace::Purge(const std::string &table_name) {
    global_mutex_.lock();
    if (mutex_map_.count(table_name)==0) {
      fl::logger->Die() << "Variable ("<<table_name<<")"
        << "does not exist, detachment failed";
    }
    mutex_map_[table_name]->unlock();
    global_mutex_.unlock();
  }

  void WorkSpace::set_schedule_mode(int schedule_mode) {
    schedule_mode_=schedule_mode;  
    if(schedule_mode == 0) {
       pool_.reset(new boost::threadpool::pool(2));
    }
  }

  void WorkSpace::set_pool(int n_threads) {
    pool_.reset(new boost::threadpool::pool(n_threads));
  }

  void WorkSpace::DummyThreadCancel(
      boost::shared_ptr<boost::thread> thread) {
    thread->join();
  }

  void WorkSpace::DummyThreadLaunch(boost::threadpool::task_func const & task) {
    boost::thread t(task);
    t.join();
  }

  void WorkSpace::CancelAllTasks() {
    if (schedule_mode_==0) {
      pool_->clear();
    } else {
      if (schedule_mode_==1) {
        for(size_t i=0; i<vector_pool_.size(); ++i) {
          //boost::thread t(boost::bind(&WorkSpace::DummyThreadCancel,
          //        this, vector_pool_[i]));
          vector_pool_[i]->interrupt();
        }
      }
    }
  }

  void WorkSpace::WaitAllTasks() {
    if (schedule_mode_==0) {
      pool_->wait();
    } else {
      if (schedule_mode_==1) {
        for(size_t i=0; i<vector_pool_.size(); ++i) {
          vector_pool_[i]->join();
        }
      }
    }
  }

  const std::string WorkSpace::GiveTempVarName() {
    boost::mutex::scoped_lock lock(temp_var_mutex_);
    std::string name(temp_var_prefix_);
    name.append(boost::lexical_cast<std::string>(temp_var_counter_));
    temp_var_counter_++;
    return name;
  }

  void WorkSpace::MakeACopy(WorkSpace *ws) {
    global_mutex_.lock();
    ws->temp_var_prefix_=this->temp_var_prefix_;
    ws->temp_var_counter_=this->temp_var_counter_;
    ws->workspace_name_=this->workspace_name_;
    ws->mutex_map_=this->mutex_map_;
    std::map<std::string, boost::shared_ptr<boost::mutex> >::iterator it;
    ws->var_map_=this->var_map_;
    global_mutex_.unlock();
  } 

  void WorkSpace::schedule(boost::threadpool::task_func const & task) {
    boost::mutex::scoped_lock lock(schedule_mutex_);
    if (schedule_mode_==0) {
      pool_->schedule(task);
    } else {
        try { 
        boost::shared_ptr<boost::thread> ptr(
            new boost::thread(&WorkSpace::DummyThreadLaunch, this, task));
        vector_pool_.push_back(ptr);
        } catch (const boost::thread_resource_error& exception)
        {
            fl::logger->Message() << "boost::thread_resource_erorr has been raised " << vector_pool_.size() << " " << std::endl;
        }
    }
  }

}}


