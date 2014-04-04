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
#include "fastlib/table/table_serialization.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/workspace/workspace_defs.h"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "boost/algorithm/string/predicate.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/bind.hpp"
#include "boost/thread/thread_time.hpp"
#include "fastlib/util/string_utils.h"
#include "boost/algorithm/string/trim.hpp"
#include "boost/program_options.hpp"
#include "boost/program_options/parsers.hpp"
#include "boost/filesystem.hpp"
#include "boost/iostreams/filtering_streambuf.hpp"
#include "boost/iostreams/copy.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "fastlib/workspace/arguments.h"

namespace fl { namespace ws {
  WorkSpace::WorkSpace() {
    Init("ismion"+boost::lexical_cast<std::string>(time(NULL)));
  }

  WorkSpace::WorkSpace(const std::string &workspace_dir) {
    Init(workspace_dir);
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
    timer_.End();
    fl::logger->Debug()<<"Workspace terminated at time: "<<timer_.GetTotalElapsedTime()
      <<"sec "<<std::endl;
  }

  void WorkSpace::Init(const std::string &workspace_dir) {
    timer_.Start();
    temp_var_counter_=0;
    temp_var_prefix_="ismion_temp_vars_"+boost::lexical_cast<std::string>(time(NULL))+"_";
    schedule_mode_=0;
    use_paging_=true;
    boost::system::error_code error_code;
    temp_directory_=boost::filesystem::temp_directory_path(error_code);
    if (error_code.value()!=0) {
      fl::logger->Warning()<<"Failed to find a temp directory error: "
        <<error_code.message();
    } 
    temp_directory_ /= boost::filesystem::path(workspace_dir);
    try {
      boost::filesystem::create_directory(temp_directory_);
    }
    catch(...) {
      fl::logger->Die()<<"Failed to create temp directory";
    }

  }

  void WorkSpace::LoadAllTables(const std::vector<std::string> &args) {
    // this variable is used to handle cases where we have
    // file sequences with a prefix and an increasing numbering
    std::map<std::string, std::pair<std::string, int32> > file_sequences;
    for(unsigned int i=0; i<args.size(); ++i) {
      if (boost::algorithm::starts_with(args[i], "--") &&
          boost::algorithm::contains(args[i], "_in=")) {
        std::vector<std::string> tokens;
        boost::algorithm::split(tokens, args[i],
            boost::algorithm::is_any_of("="));
        // check if is is an increasing file sequences
        if (fl::StringEndsWith(tokens[0], "_prefix_in")) {
          std::string table_name=tokens[0].substr(0,
                tokens[0].size()-10);
          if (file_sequences.count(table_name)>0) {
            file_sequences[table_name].first=tokens[1];
          } else {
            file_sequences[table_name]=std::make_pair(tokens[1], 0);
          }
          continue;
        }
        if (fl::StringEndsWith(tokens[0], "_num_in")) {
          try {
            std::string table_name=tokens[0].substr(0, 
                  tokens[0].size()-7);
            if (file_sequences.count(table_name)>0) {
              file_sequences[table_name].second=
                boost::lexical_cast<int32>(tokens[1]);
            } else {
              file_sequences[table_name]=std::make_pair("",
                      boost::lexical_cast<int32>(tokens[1]));
            }
          }
          catch(...) {
            fl::logger->Die()<<"There is something wrong in this argument "
              <<args[i];
          }
          continue;
        }        // check if it contains more than one filenames
        if (boost::algorithm::contains(tokens[1], ":")==false
            && boost::algorithm::contains(tokens[1], ",")==false
            && fl::StringEndsWith(tokens[0], "_prefix_in")==false) {
          std::string filename=tokens[1];
          std::string variable=filename;
          if (boost::algorithm::contains(tokens[0], "references_in")||
             boost::algorithm::contains(tokens[0], "queries_in")) {
            LoadFromFile<DataTables_t>(variable, filename);
            Purge(variable);
          } else {
            LoadFromFile<ParameterTables_t>(variable, filename);
            Purge(variable);
          }
        } else {
          std::vector<std::string> filenames;
          boost::algorithm::trim_right_if(tokens[1], boost::algorithm::is_any_of(":,"));
          boost::algorithm::split(filenames, tokens[1], 
             boost::algorithm::is_any_of(":,"));
          for(unsigned int j=0; j<filenames.size(); ++j) {
            std::string variable(filenames[j]);
            if (boost::algorithm::contains(tokens[0], "references")||
              boost::algorithm::contains(tokens[0], "queries")) {
              LoadFromFile<DataTables_t>(variable, filenames[j]);
              Purge(variable);
            } else {
              LoadFromFile<ParameterTables_t>(variable, filenames[j]);
              Purge(variable);
            }
          }
        }
      }
    }
    // Now load the filesequences with a prefix and an increasing order
    for(std::map<std::string, std::pair<std::string, int32> >::iterator 
        it=file_sequences.begin(); it!=file_sequences.end(); ++it) {
      std::vector<std::string> references_filenames;
      this->LoadFileSequence( 
          it->second.first, 
          it->second.second,
          it->second.first,
          &references_filenames 
      );
    }
    int id=timer_.CheckPoint();
    fl::logger->Debug()<<"Finished loading tables at time: "
      <<timer_.GetElapsedTime(id)<<"sec"<<std::endl;
  }

  void WorkSpace::IndexAllReferencesQueries(
      const std::vector<std::string> &args) {
    if (is_mode_sequential()==false) {
      fl::logger->Die()<<"Cannot IndexAllReferencesQueries in asynchronous mode";
    }  
    IndexAllReferencesQueries(const_cast<std::vector<std::string> *>(&args));
    int id=timer_.CheckPoint();
    fl::logger->Debug()<<"Finished indexing tables at time: "
      <<timer_.GetElapsedTime(id)<<"sec"<<std::endl;
  }

  void WorkSpace::IndexAllReferencesQueries(
      std::vector<std::string> *args) {
    bool is_sequential=this->is_mode_sequential();
    std::string metric("l2");
    std::string metric_args;
    int leaf_size=20;
    std::map<std::string, std::string> argmap;
    for(unsigned int i=0; i<args->size(); ++i) {
      std::vector<std::string> tokens;
      boost::algorithm::split(tokens, (*args)[i],
          boost::algorithm::is_any_of("="));
      if (tokens.size()!=2) {
        continue;
      }
      boost::algorithm::trim_left_if(tokens[0], boost::algorithm::is_any_of("-"));
      argmap[tokens[0]]=tokens[1];
    }
    if (argmap.count("metric")>0) {
      metric=argmap["metric"];
    }
    if (argmap.count("metric_weights_in")>0) {
      metric_args=argmap["metric_weights_in"];
    }
    if (argmap.count("leaf_size")>0) {
      leaf_size=boost::lexical_cast<int>(argmap["leaf_size"]);         
    }
    std::vector<std::string> references_names;
    if (argmap.count("refernces_in")>0 || argmap.count("references_prefix_in")>0) {
      references_names=fl::ws::GetFileSequence("references", argmap);
    }
    std::vector<std::string> queries_names;
    if (argmap.count("queries_in")>0 || argmap.count("queries_prefix_in")>0) {
      queries_names=fl::ws::GetFileSequence("queries", argmap);
    }
    
    for(size_t i=0; i<references_names.size(); ++i) {
      if (is_sequential) {
        IndexTable(references_names[i], metric, metric_args, leaf_size);
      } else {
        std::string new_variable=GiveTempVarName();
        MakeTableCopy(references_names[i], new_variable);
        IndexTable(new_variable, metric, metric_args, leaf_size);
      }
    }
    for(size_t i=0; i<queries_names.size(); ++i) {
      if (is_sequential) {
        IndexTable(queries_names[i], metric, metric_args, leaf_size);
      } else {
        std::string new_variable=GiveTempVarName();
        MakeTableCopy(queries_names[i], new_variable);
        IndexTable(new_variable, metric, metric_args, leaf_size);
      }
    }
  }

  void WorkSpace::MakeTableCopy(const std::string &source_table,
      const std::string &dest_table) {    
    bool success;
    boost::mpl::for_each<DataTables_t>(CopyMeta(this, &success,
            source_table, dest_table));
  }
  

  void WorkSpace::ExportAllTablesTask(
      const std::vector<std::string> args) {
    try {
      std::map<std::string, std::pair<std::string, int32> > file_sequences;
      for(unsigned int i=0; i<args.size(); ++i) {
        if (boost::algorithm::starts_with(args[i], "--") &&
          boost::algorithm::contains(args[i], "_out=")) {
          std::vector<std::string> tokens;
          boost::algorithm::split(tokens, args[i],
              boost::algorithm::is_any_of("="));
          // check if is is an increasing file sequences
          if (fl::StringEndsWith(tokens[0], "_prefix_out")) {
            std::string table_name=tokens[0].substr(0,
                    tokens[0].size()-10);
            if (file_sequences.count(table_name)>0) {
              file_sequences[table_name].first=tokens[1];
            } else {
              file_sequences[table_name]=std::make_pair(tokens[1], 0);
            }
            continue;
          }
          if (fl::StringEndsWith(tokens[0], "_num_out")) {
            try {
              std::string table_name=tokens[0].substr(0, 
                  tokens[0].size()-7);
              if (file_sequences.count(table_name)>0) {
                  file_sequences[table_name].second=
                    boost::lexical_cast<int32>(tokens[1]);
              } else {
                file_sequences[table_name]=std::make_pair("",
                          boost::lexical_cast<int32>(tokens[1]));
              }
            }
            catch(...) {
              fl::logger->Die()<<"There is something wrong in this argument "
                  <<args[i];
            }
            continue;
          }        // check if it contains more than one filenames
          if (boost::algorithm::contains(tokens[1], ":")==false
              && boost::algorithm::contains(tokens[1], ",")==false
              && fl::StringEndsWith(tokens[0], "_prefix_out")==false) {
            std::string filename=tokens[1];
            std::string variable=filename;
            ExportToFile(variable, filename);
          } else {
            std::vector<std::string> filenames;
            boost::algorithm::split(filenames, tokens[1], 
                 boost::algorithm::is_any_of(":,"));
            for(unsigned int i=0; i<filenames.size(); ++i) {
              std::string variable(filenames[i]);
              ExportToFile(variable, filenames[i]);
            }
          }
        }
      }
      // Now export the filesequences with a prefix and an increasing order
      for(std::map<std::string, std::pair<std::string, int32> >::iterator 
        it=file_sequences.begin(); it!=file_sequences.end(); ++it) {
        std::vector<std::string> references_filenames;
        this->ExportFileSequence( 
              it->second.first, 
              it->second.second,
              it->second.first,
              &references_filenames 
        );
      }
    }
    catch(const boost::thread_interrupted &e) {
      std::cout<< "thread interruption"<<std::endl;
    }
    catch(...) {
      boost::mutex::scoped_lock lock(*global_exception_mutex);
      fl::global_exception=boost::current_exception();
    }
    int id=timer_.CheckPoint();
    fl::logger->Debug()<<"Finished exporting tables at time: "
      <<timer_.GetElapsedTime(id)<<"sec"<<std::endl;
  }

  void WorkSpace::ExportAllTables(
      const std::vector<std::string> args) {
    this->schedule(boost::bind(&WorkSpace::ExportAllTablesTask, this, args));
  }

  void WorkSpace::RemoveTable(const std::string &table_name) {
    boost::mutex* mutex;
    global_mutex_.lock();
    if (mutex_map_.count(table_name)==0) {
      fl::logger->Warning()<<"Attempt to remove non-existent table ("
       <<table_name<<") from workspace (" 
       <<workspace_name_
       <<")"
       <<std::endl;
      return;
    } else {
      mutex=mutex_map_[table_name].get();
    }
    mutex->try_lock();
    var_map_.erase(table_name);
    mutex->unlock();
    mutex_map_.erase(table_name);
    global_mutex_.unlock(); 
  }

  void WorkSpace::ExportFileSequence(
      const boost::program_options::variables_map &vm, 
      const std::string &flag_argument, 
      const std::string &variable_name, 
      std::vector<std::string> *filenames) {
 
    if (vm.count(flag_argument+"_out")>0) {
      *filenames=
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
          filenames->push_back(
              fl::StitchStrings(prefix, i));
        }
      }
    }
 
    this->schedule(boost::bind(
      &WorkSpace::ExportFileSequenceTask,
      this,
      variable_name,
      *filenames));
  }
 
  void WorkSpace::ExportFileSequence( 
      const std::string &prefix, 
      const int32 num_of_files,
      const std::string &variable_name, 
      std::vector<std::string> *filenames) {
 
    for(int32 i=0; i<num_of_files; ++i) {
      filenames->push_back(
        fl::StitchStrings(prefix, i));
    }
 
    this->schedule(boost::bind(
      &WorkSpace::ExportFileSequenceTask,
      this,
      variable_name,
      *filenames));
  }
 
  void WorkSpace::ExportFileSequenceTask(
      const std::string variable_name,
      std::vector<std::string> filenames) {
    
    for(int32 i=0; i<filenames.size(); ++i) {
      ExportToFile(fl::StitchStrings(variable_name, i),
          filenames[i]);
    }
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
        if (num<=0) {
          fl::logger->Die()<<"--"+flag_argument+"_num_in="
            <<num<<" is wrong, it should be greater than zero";
        }
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
 
  void WorkSpace::LoadFileSequence( 
      const std::string &prefix, 
      const int32 num_of_files,
      const std::string &variable_name, 
      std::vector<std::string> *references_filenames) {
 
    for(int32 i=0; i<num_of_files; ++i) {
      references_filenames->push_back(
        fl::StitchStrings(prefix, i));
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
     try {
        LoadDataTableFromFile(fl::StitchStrings(variable_name, i),
            references_filenames[i]);
        Purge(fl::StitchStrings(variable_name, i));
      } 
      catch(...) {
        fl::logger->Warning()<<"Oups failed to load it as a data table trying "
        "to load it as a parameter table";
        LoadParameterTableFromFile(fl::StitchStrings(variable_name, i),
            references_filenames[i]);
        Purge(fl::StitchStrings(variable_name, i));
      }
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
    if (use_paging_) {
      SerializeToDisk(variable); 
    }
    success=false;
    boost::mpl::for_each<DataTables_t>(
        ClearTableNameMeta(variable, var_map_, &success)); 
    if (success==false) {
      boost::mpl::for_each<ParameterTables_t>(
        ClearTableNameMeta(variable, var_map_, &success));
    }

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
      if (ws_->use_paging()) {
        ws_->SerializeFromDisk(variable_, &table); 
          (*var_map_)[variable_]=table;
      }

      *success_=true;
    }
    catch(const boost::bad_any_cast &e) {
      return; 
    }
    if (table->is_indexed()==true) {
      fl::logger->Warning()<<"Table ("<<variable_
        <<") is already indexed, skipping indexing";
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
    fl::logger->Message()<<"Indexed table: "<<variable_<<std::endl;
    table->LogTreeStats();
  }

  WorkSpace::CopyMeta::CopyMeta(WorkSpace * const ws,
     bool *success,
     const std::string &source, const std::string &dest) :
       ws_(ws), success_(success), source_(source), dest_(dest) {}

  template<typename TableType>
  void WorkSpace::CopyMeta::operator()(TableType&) {
    if (*success_==true) {
      return;
    }
    boost::shared_ptr<TableType> source_table;
    boost::shared_ptr<TableType> dest_table(new TableType());
    try {
      ws_->template TryToAttach<TableType>(source_);
      ws_->Attach(source_, &source_table);
      source_table->CloneDataOnly(dest_table.get());      
      ws_->LoadTable(dest_, dest_table);
      *success_=true;
    }
    catch(...) {
      return; 
    }

  }

  void WorkSpace::ExportToFile(const std::string &name, const std::string &filename) {
    boost::mutex* mutex;
    bool is_name_in_map;
    {
      boost::mutex::scoped_lock lock(global_mutex_);
      is_name_in_map=(mutex_map_.count(name)!=0);
    }
    if (schedule_mode_==2 && is_name_in_map==false) {
      fl::logger->Warning()<<"table ("
        <<name
        <<") was not generated at all by your program "
        <<"skiping export to filename ("
        <<filename
        <<")"<<std::endl;
      return;
    }
    if (is_name_in_map==false) {
      mutex=new boost::mutex();
      boost::mutex::scoped_lock lock(global_mutex_);
      mutex_map_[name].reset(mutex);
      mutex->lock();
    } else {
      boost::mutex::scoped_lock lock(global_mutex_);
      mutex=mutex_map_[name].get();
    }
    if (schedule_mode_==2) {
      if (mutex->try_lock()==false) {
        fl::logger->Warning()<<"something happened and table ("
          <<name
          <<") was not generated properly by your program "
          <<"skiping export to filename ("
          <<filename
          <<")"<<std::endl;
       return;
      } else {
        mutex->unlock();
      }
    }
    boost::mutex::scoped_lock lock(*mutex);
    bool success=false;
    boost::mpl::for_each<ParameterTables_t>(SaveMeta(this, &var_map_, 
          name, filename, &success));
    boost::mpl::for_each<DataTables_t>(SaveMeta(this, &var_map_, 
          name, filename, &success));
    if (success==false) {
      fl::logger->Die()<<"Cannot save table ("<<name<<"), the table "
        "type is not supported"; 
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
      if (boost::any_cast<boost::shared_ptr<TableType> >((*var_map_)[name_]).get()==NULL) {
        boost::shared_ptr<TableType> table;
        if (ws_->use_paging()) {
          ws_->SerializeFromDisk(name_, &table); 
          (*var_map_)[name_]=table;
        } else {
          fl::logger->Die() << "You are trying to attach a table ("<<name_
              << ") that somehow has been deleted, please report this bug";

        }
      }
      boost::any_cast<boost::shared_ptr<TableType> >((*var_map_)[name_])->filename()=filename_;
      boost::any_cast<boost::shared_ptr<TableType> >((*var_map_)[name_])->Save();
      (*var_map_)[name_]=boost::shared_ptr<TableType>();
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

  template<typename TableType>
  void WorkSpace::ClearTableNameMeta::operator()(TableType&) {
    if (*success_==true) {
      return;
    }
    try {
      boost::shared_ptr<TableType> dummy;
      dummy=boost::any_cast<boost::shared_ptr<TableType> >(var_map_[table_]);
      boost::shared_ptr<TableType> dummy1;
      var_map_[table_]=dummy1;
      *success_=true;
    }
    catch(const boost::bad_any_cast &e) {
    
    }
  }

  void WorkSpace::Purge(const std::string &table_name) {
    Purge(table_name, false);
  }

  void WorkSpace::Purge(const std::string &table_name,
      bool force_save) {
    FL_SCOPED_LOG(Workspace);
    FL_SCOPED_LOG(Purge);
    global_mutex_.lock();
    if (mutex_map_.count(table_name)==0) {
      global_mutex_.unlock();
      fl::logger->Die() << "Variable ("<<table_name<<")"
        << "does not exist, detachment failed";
    }
    // if the table is in read mode then it is already unlocked 
    // we do a try_lock 
    mutex_map_[table_name]->try_lock();
    // if it was in write mode then it is locked. If it was in read mode
    // it will be locked after the try_lock
    GetTableInfo(table_name,
        &(table_info_[table_name].n_entries), 
        &(table_info_[table_name].n_attributes),
        &(table_info_[table_name].dense_sizes),
        &(table_info_[table_name].sparse_sizes)); 
    
    if (IsInTempDir(table_name)==false || force_save==true) {
      if (use_paging_) {
        SerializeToDisk(table_name); 
      }
    }

    bool success=false;
    boost::mpl::for_each<DataTables_t>(
        ClearTableNameMeta(table_name, var_map_, &success)); 
    if (success==false) {
      boost::mpl::for_each<ParameterTables_t>(
        ClearTableNameMeta(table_name, var_map_, &success));
    }
    if (success==false) {
      fl::logger->Die()<<"Failed to Purge ("<<table_name<<")";
    }
    mutex_map_[table_name]->unlock();
    global_mutex_.unlock();
    fl::logger->Debug()<<"Table ("<<table_name<<") purged";
  }

  namespace WorkSpace_GetTableInfo {
    class Do {
      public:
        Do(WorkSpace *ws,
            std::map<std::string, boost::any> &var_map,
            const std::string &table_name,
            index_t *n_entries,
            index_t *n_attributes,
            std::vector<index_t> *dense_sizes,
            std::vector<index_t> *sparse_sizes,
            bool *success) :
          ws_(ws), var_map_(var_map),table_name_(table_name), n_entries_(n_entries),
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
            table=boost::any_cast<boost::shared_ptr<TableType> >(var_map_[table_name_]);
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
            fl::logger->Die()<<"Terminating...";
          }
          catch(...) {
            return; 
          }
        }
      private:
        WorkSpace *ws_;
        std::map<std::string, boost::any> &var_map_;
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
    FL_SCOPED_LOG(GetTableInfo);
    if (table_info_[table_name].n_entries>0) {
      if (n_entries!=NULL) {
          *n_entries=table_info_[table_name].n_entries;
      }
      if (n_attributes!=NULL) {
        *n_attributes=table_info_[table_name].n_attributes;
      }
      if (dense_sizes!=NULL) {
        *dense_sizes=table_info_[table_name].dense_sizes;
      }
      if (sparse_sizes!=NULL) {
         *sparse_sizes=table_info_[table_name].sparse_sizes;
      }       
      return;
    }
    bool success=false;
    boost::mpl::for_each<DataTables_t>(
        WorkSpace_GetTableInfo::Do(this,
          var_map_,
          table_name,
          n_entries,
          n_attributes,
          dense_sizes,
          sparse_sizes,
          &success));
    boost::mpl::for_each<ParameterTables_t>(
        WorkSpace_GetTableInfo::Do(this,
          var_map_,
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

  void WorkSpace::set_paging_mode(int paging_mode) {
    if (paging_mode!=0 && paging_mode!=1) {
      use_paging_=paging_mode;
    }
  }


  bool WorkSpace::is_mode_sequential() const {
    if (schedule_mode_==2) {
      return true;
    } 
    return false;
  }

  bool WorkSpace::use_paging() const {
    return use_paging_;
  }

  void WorkSpace::set_pool(int n_threads) {
    if (schedule_mode_==0) {
      pool_.reset(new boost::threadpool::pool(n_threads));
    }
  }

  void WorkSpace::set_temp_directory(const std::string &directory) {
    try {
      temp_directory_=boost::filesystem::path(directory); 
    }
    catch(...) {
      fl::logger->Die()<<"Cannot set as workspace directory path ("
        <<directory<<")";
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
        if (pool_==NULL ||  pool_->empty()) {
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
  
  std::string WorkSpace::GiveFilenameFromSequence(
        const std::string &prefix, 
        int32 index) {
    return prefix+boost::lexical_cast<std::string>(index);
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
    if (schedule_mode_!=2) {
      schedule_mutex_.lock();
    }
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
          if (fl::global_exception) {
            fl::logger->Die()<<"Program terminated";
          }
        }
      }
    }
    if (schedule_mode_!=2) {
      schedule_mutex_.unlock();
    }
  }
  
  bool WorkSpace::IsInTempDir(const std::string &name) {
    boost::filesystem::path to_be_searched=FromTableNameToPath(name);
    return boost::filesystem::exists(to_be_searched);
  }

  boost::filesystem::path WorkSpace::FromTableNameToPath(
        const std::string &table_name) {
    boost::filesystem::path slash("/");

    std::string name_with_no_slashes=boost::algorithm::replace_all_copy(
        table_name, 
        slash.make_preferred().native(), "_");
    boost::filesystem::path transformed=temp_directory_;
    transformed /= boost::filesystem::path(name_with_no_slashes);
    return transformed;
 
  }

  template<typename TableType>
  void WorkSpace::SerializeToDiskMeta::operator()(TableType&) {
    if (*success_==true) {
      return;
    }
    try {
      boost::shared_ptr<TableType> table;
      table=boost::any_cast<boost::shared_ptr<TableType> >(var_map_[name_]);
      oa_<<*table;
      *success_=true;
    }
    catch(...) {
      
    }
  }


  void WorkSpace::SerializeToDisk(const std::string &table_name) {
    FL_SCOPED_LOG(Workspace);
    boost::filesystem::path to_be_saved=FromTableNameToPath(table_name);
    std::ofstream ofs(to_be_saved.string().c_str(),
        std::ios::out|std::ios::binary);
    if (ofs.good()==false) {
      fl::logger->Die()<<"Cannot open file ("
        <<to_be_saved.string()
        <<") for saving table"; 
    }    
    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::gzip_compressor());
    out.push(ofs);
    boost::archive::binary_oarchive oa(out);
    bool success=false;
    boost::mpl::for_each<DataTables_t>(
        SerializeToDiskMeta(
          this,
          table_name,  
          var_map_,
          oa,  &success));
    if (success==false) {
      boost::mpl::for_each<ParameterTables_t>(
          SerializeToDiskMeta(
          this,
          table_name,  
          var_map_,
          oa,  &success));
    }
    if (success==false) {
      fl::logger->Warning()<<"Unable to serialize table "<<table_name<<" to disk";
    }
    //oa << *table;

  }

}}


