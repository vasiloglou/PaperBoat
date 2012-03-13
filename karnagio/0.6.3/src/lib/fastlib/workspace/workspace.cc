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
#include "boost/thread/thread_time.hpp"
#include "fastlib/util/string_utils.h"

namespace fl { namespace ws {
  WorkSpace::WorkSpace() {
    temp_var_counter_=0;
    temp_var_prefix_="ismion_temp_vars_";
    schedule_mode_=0;
    //
  }
  WorkSpace::~WorkSpace() {
    WaitAllTasks();
    std::map<std::string, boost::any>::iterator it1;
    for(it1=var_map_.begin();it1!=var_map_.end(); ++it1) {
      it1->second=std::string();
    }
    std::map<std::string, boost::shared_ptr<boost::mutex> >::iterator it;
    global_mutex_.try_lock();
    for(it=mutex_map_.begin(); it!=mutex_map_.end(); ++it) {
      // we need to make sure that all the mutexes are unlocked
      it->second->try_lock();
      it->second->unlock();
      it->second.reset(new boost::mutex());
    }
    global_mutex_.unlock();
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
      const std::vector<std::string> args) {
    try {
      for(unsigned int i=0; i<args.size(); ++i) {
        if (boost::algorithm::contains(args[i], "_out=")) {
          std::vector<std::string> tokens(2);
          size_t pos=args[i].find('=');
          tokens[1]=args[i].substr(pos+1); 
          //tokens=fl::SplitString(args[i], "=");
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
    catch(const boost::thread_interrupted &e) {
      std::cout<< "thread interruption"<<std::endl;
    }
    catch(...) {
      boost::mutex::scoped_lock lock(*global_exception_mutex);
      fl::global_exception=boost::current_exception();
    }
  }

  void WorkSpace::ExportAllTables(
      const std::vector<std::string> args) {
    this->schedule(boost::bind(&WorkSpace::ExportAllTablesTask, this, args));
  }

  void WorkSpace::LoadFileSequence(
      const boost::program_options::variables_map &vm, 
      const std::string &flag_argument, 
      const std::string &variable_name, 
      std::vector<std::string> *references_filenames) {
 
    if (vm.count(flag_argument+"_in")>0) {
      *references_filenames=
        fl::SplitString(vm[flag_argument+"_in"].as<std::string>(), ",");    
    } else {
      if (vm.count(flag_argument+"_prefix_in")>0) {
        if (vm.count(flag_argument+"_num_in")==0) {
          fl::logger->Die()<<"Since you defined flag "
            <<"--"<<flag_argument+"_prefix_in"
            <<", you should also define "
            <<"--"<<flag_argument+"_num_in";
        }
        std::string prefix=vm[flag_argument+"_prefix_in"].as<std::string>();
        int32 num=vm[flag_argument+"_num_in"].as<int32>();
        for(int32 i=0; i<num; ++i) {
          references_filenames->push_back(
              fl::StitchStrings(prefix, i));
        }
      }
    }
 
    this->schedule(boost::bind(
      &WorkSpace::LoadFileSequenceTask,
      this,
      variable_name,
      *references_filenames));
  }
  
  void WorkSpace::LoadFileSequenceTask(
      const std::string variable_name,
      std::vector<std::string> references_filenames) {
    
    for(int32 i=0; i<references_filenames.size(); ++i) {
      LoadDataTableFromFile(fl::StitchStrings(variable_name, i),
          references_filenames[i]);
    }
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
    bool is_name_in_map;
    {
      boost::mutex::scoped_lock lock(global_mutex_);
      is_name_in_map=mutex_map_.count(name)==0;
    }
    if (is_name_in_map) {
      mutex=new boost::mutex();
      boost::mutex::scoped_lock lock(global_mutex_);
      mutex_map_[name].reset(mutex);
      mutex->lock();
    } else {
      boost::mutex::scoped_lock lock(global_mutex_);
      mutex=mutex_map_[name].get();
    }
    boost::mutex::scoped_lock lock(*mutex);
    bool success=false;
    boost::mpl::for_each<ParameterTables_t>(SaveMeta(&var_map_, 
          name, filename, &success));
    boost::mpl::for_each<DataTables_t>(SaveMeta(&var_map_, 
          name, filename, &success));
    if (success==false) {
      fl::logger->Die()<<"Cannot save table ("<<name<<"), the table "
        "type is not supported"; 
    }
  }

  void WorkSpace::ExportToFileSequence(
        const boost::program_options::variables_map &vm, 
        const std::string &var_name_prefix, 
        const std::string &flag_argument, 
        std::vector<std::string> *file_sequence) {
 
    if (vm.count(flag_argument+"_out")>0) {
      *file_sequence=
        fl::SplitString(vm[flag_argument+"_out"].as<std::string>(), ",");    
    } else {
      if (vm.count(flag_argument+"_prefix_out")>0) {
        if (vm.count(flag_argument+"_num_out")==0) {
          fl::logger->Die()<<"Since you defined flag "
            <<"--"<<flag_argument+"_prefix_out"
            <<", you should also define "
            <<"--"<<flag_argument+"_num_out";
        }
        std::string prefix=vm[flag_argument+"_prefix_out"].as<std::string>();
        int32 num=vm[flag_argument+"_num_out"].as<int32>();
        for(int32 i=0; i<num; ++i) {
          file_sequence->push_back(
              fl::StitchStrings(prefix, i));
        }
      }
    }
    for(int32 i=0; i<file_sequence->size(); ++i) {
      std::string name=fl::StitchStrings(var_name_prefix, i);
      ExportToFile(name, (*file_sequence)[i]);
    } 
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

  void WorkSpace::LoadDataTableFromFileTask(const std::string name,
      const std::string filename) {
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

  namespace WorkSpace_GetTableInfo {
    class Do {
      public:
        Do(WorkSpace *ws,
            const std::string &table_name,
            index_t *n_entries,
            index_t *n_attributes,
            std::vector<index_t> *dense_sizes,
            std::vector<index_t> *sparse_sizes,
            bool *success) :
          ws_(ws), table_name_(table_name), n_entries_(n_entries),
          n_attributes_(n_attributes), dense_sizes_(dense_sizes),
          sparse_sizes_(sparse_sizes), success_(success) {
        }
        template<typename TableType>
        void operator()(TableType&) {
          if (*success_==true) {
            return;
          }
          boost::shared_ptr<TableType> table;
          try {
            ws_->TryToAttach<TableType>(table_name_);
            ws_->Attach(table_name_, &table);
            *success_=true;
            if (n_entries_!=NULL) {
              *n_entries_=table->n_entries();
            }
            if (n_attributes_!=NULL) {
              *n_attributes_=table->n_attributes();
            }
            if (dense_sizes_!=NULL) {
              *dense_sizes_=table->dense_sizes();
            }
            if (sparse_sizes_!=NULL) {
              *sparse_sizes_=table->sparse_sizes();
            }
          }
          catch(const fl::TypeException &e) {
            return; 
          }
        }
      private:
        WorkSpace *ws_;
        const std::string &table_name_;
        index_t *n_entries_;
        index_t *n_attributes_;
        std::vector<index_t> *dense_sizes_;
        std::vector<index_t> *sparse_sizes_;  
        bool *success_;
    };
  }
  void WorkSpace::GetTableInfo(const std::string &table_name,
        index_t *n_entries, 
        index_t *n_attributes,
        std::vector<index_t> *dense_sizes,
        std::vector<index_t> *sparse_sizes) {
    bool success=false;
    boost::mutex::scoped_lock lock(global_mutex_);
    boost::mpl::for_each<DataTables_t>(
        WorkSpace_GetTableInfo::Do(this,
          table_name,
          n_entries,
          n_attributes,
          dense_sizes,
          sparse_sizes,
          &success));
    boost::mpl::for_each<ParameterTables_t>(
        WorkSpace_GetTableInfo::Do(this,
          table_name,
          n_entries,
          n_attributes,
          dense_sizes,
          sparse_sizes,
          &success)
        ); 
    if (success==false) {
      fl::logger->Die()<<"Table ("<<table_name<<")" 
        "is not the type you requested, It wasn't possible to "
        "attach it, Type Error"; 
    }
  }

  void WorkSpace::set_schedule_mode(int schedule_mode) {
    schedule_mode_=schedule_mode;  
    if(schedule_mode == 0) {
       pool_.reset(new boost::threadpool::pool(2));
    }
  }

  void WorkSpace::set_pool(int n_threads) {
    if (schedule_mode_==0) {
      pool_.reset(new boost::threadpool::pool(n_threads));
    }
  }

  void WorkSpace::DummyThreadCancel(
      boost::shared_ptr<boost::thread> thread) {
    thread->join();
  }

  void WorkSpace::DummyThreadLaunch(boost::threadpool::task_func const & task) {
    boost::thread t(task);
    try {
      t.join();
    }
    catch(const boost::thread_interrupted &e) {
      std::cout<< "dummy interrupted"<<std::endl;
      t.sleep(boost::posix_time::ptime(boost::posix_time::pos_infin));
    }
  }

  void WorkSpace::CancelAllTasks() {
    boost::mutex::scoped_lock lock(schedule_mutex_);
    if (schedule_mode_==0) {
      pool_->clear();
    } else {
      if (schedule_mode_==1) {
        for(std::list<boost::thread*>::iterator it=vector_pool_.begin();
              it!=vector_pool_.end(); ++it) {
           (*it)->interrupt();
           (*it)->join();
          it=vector_pool_.erase(it);
        }
      }
    }
  }

  void WorkSpace::WaitAllTasks() {
    if (schedule_mode_==0) {
      while(true) {
        //boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
        boost::mutex::scoped_lock lock(*global_exception_mutex);
        if (fl::global_exception) {
          CancelAllTasks();
        }
        if (pool_->empty()) {
          break;
        }
      }
    } else {
      if (schedule_mode_==1) {
        size_t pool_size=1;
        while (pool_size>0) {
          {
            std::list<boost::thread*>::iterator list_end;
            std::list<boost::thread*>::iterator list_begin;
            {
              boost::mutex::scoped_lock lock1(schedule_mutex_);
              list_begin=vector_pool_.begin();
              list_end=vector_pool_.end();
            }
            for(std::list<boost::thread*>::iterator it=list_begin;
                it!=list_end; ++it) {
              if ((*it)->get_id()==boost::this_thread::get_id()) {
                std::cout<<(*it)->get_id()<<std::endl;
                thread_group_.remove_thread(*it);
                delete *it;
                { 
                  boost::mutex::scoped_lock lock1(schedule_mutex_);
                  it=vector_pool_.erase(it);
                }
                continue;
              }
              if ((*it)->joinable()==false) {
                thread_group_.remove_thread(*it);
                delete *it;
                { 
                  boost::mutex::scoped_lock lock1(schedule_mutex_);
                  it=vector_pool_.erase(it);
                }
                continue;
              }
              if ((*it)->timed_join(boost::posix_time::milliseconds(100))==true) {
                thread_group_.remove_thread(*it);
                delete *it;
                { 
                  boost::mutex::scoped_lock lock1(schedule_mutex_);
                  it=vector_pool_.erase(it);
                }
                continue;
              }
            }
          }
          boost::this_thread::yield();
          // boost::this_thread::sleep(boost::posix_time::milliseconds(100));
          {
            boost::mutex::scoped_lock lock2(*global_exception_mutex);
            if (fl::global_exception) {
              CancelAllTasks();
            }
          }
          {
            boost::mutex::scoped_lock lock1(schedule_mutex_);
            pool_size=vector_pool_.size();
          }
        }     
        thread_group_.join_all();
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
    boost::mutex::scoped_lock lock1(schedule_mutex_);
    if (fl::global_exception) {
      return;
    }
    if (schedule_mode_==0) {
      boost::mutex::scoped_lock lock2(*global_exception_mutex);
      pool_->schedule(task);
    } else {
      if (schedule_mode_==1) {
        boost::mutex::scoped_lock lock2(*global_exception_mutex);
        try { 
          vector_pool_.push_back(thread_group_.create_thread(
               task));
        } 
        catch (const boost::thread_resource_error& exception)
        {
            fl::logger->Message() << "boost::thread_resource_erorr has been raised, "
              "threads already allocated: " << vector_pool_.size() << " " << std::endl;
        }
      } else {
        if (schedule_mode_==2) {
          task();
        }
      }
    }
  }

}}


