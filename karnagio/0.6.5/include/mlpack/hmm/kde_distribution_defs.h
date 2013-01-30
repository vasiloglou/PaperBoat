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

#ifndef PAPERBOAT_INCLUDE_MLPACK_KDE_DISTRIBUTION_DEFS_H_
#define PAPERBOAT_INCLUDE_MLPACK_KDE_DISTRIBUTION_DEFS_H_

#include "kde_distribution.h"
#include "mlpack/kde/kde.h"

namespace fl { namespace ml {
  
  template<typename TableType>
  void KdeDistribution<TableType>::Init(
      const std::vector<std::string> &args,
      int32 id,
      const std::vector<index_t> &dense_sizes, 
      const std::vector<index_t> &sparse_sizes) {
    references_.reset(new Table_t());
    references_->Init("references",
        dense_sizes,
        sparse_sizes,
        0);
    references_ptr_=references_.get(); 
    args_=args; 
    args_.push_back("--references_in=references");
    id_=id;
  } 
 
  template<typename TableType>
  template<typename WorkSpaceType>
  void KdeDistribution<TableType>::Import(
      const std::vector<std::string> &import_args,
      const std::vector<std::string> &exec_args,
      WorkSpaceType *ws,
      int32 id) {
 
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "references_prefix_in",
      boost::program_options::value<std::string>(),
      "this is the filename for storing the histogram"
    )(
      "bandwidths_prefix_in",
      boost::program_options::value<std::string>(),
      "this is the filename for storing the bandwidths"  
    );
    boost::program_options::variables_map vm;
    boost::program_options::command_line_parser clp(import_args);
    clp.style(boost::program_options::command_line_style::default_style
       ^boost::program_options::command_line_style::allow_guessing);
    try {
      boost::program_options::store(clp.options(desc).run(), vm);
    }
    catch(const boost::program_options::invalid_option_value &e) {
  	  fl::logger->Die() << "Invalid Argument: " << e.what();
    }
    catch(const boost::program_options::invalid_command_line_syntax &e) {
  	  fl::logger->Die() << "Invalid command line syntax: " << e.what(); 
    }
    catch (const boost::program_options::unknown_option &e) {
       fl::logger->Die() << e.what()
        <<" . This option will be ignored";
    }
    catch ( const boost::program_options::error &e) {
      fl::logger->Die() << e.what();
    } 
    boost::program_options::notify(vm);
    boost::shared_ptr<TableType> references_table;
    std::string name1=vm["references_prefix_in"].as<std::string>()
        +boost::lexical_cast<std::string>(id_);
    ws->Attach(name1, 
        &references_table);   
    boost::shared_ptr<fl::ws::WorkSpace::DefaultTable_t> bandwidth_table;
    std::string name2=vm["bandwidths_prefix_in"].as<std::string>()
        +boost::lexical_cast<std::string>(id_);

    typename fl::ws::WorkSpace::DefaultTable_t::Point_t point;
    ws->Attach(name2,
               &bandwidth_table);
    bandwidth_table->get(0, &point);
    point.set(0, bandwidth_);
    ws->Purge(name2);
    ws->Detach(name2); 
    args_=exec_args;
  }
 
  template<typename TableType>
  void KdeDistribution<TableType>::ResetData() {
    const std::vector<index_t> &dense_sizes=references_->dense_sizes();
    const std::vector<index_t> &sparse_sizes=references_->sparse_sizes();
    references_.reset(new Table_t());
    references_->Init("references",
        dense_sizes,
        sparse_sizes,
        0);
    references_ptr_=references_.get(); 
  }
  
  template<typename TableType>
  void KdeDistribution<TableType>::AddPoint(Point_t &point) {
    references_ptr_->push_back(point);
  }


  template<typename TableType>
  void KdeDistribution<TableType>::Train() {
    if (references_->n_entries()==0) {
      fl::logger->Warning()<<"Empty Kde Distribution "
        <<"skipping training"<<std::endl;
      return;
    }
    ws_.reset(new fl::ws::WorkSpace());
    ws_->set_schedule_mode(2);
    ws_->LoadTable("references", references_);
    ws_->IndexTable("references", 
                    "l2",
                    "",
                    20);
    boost::shared_ptr<Table_t> query_table;
    std::vector<std::string> args=args_;
    args.push_back("--bandwidth_out=bandwidth");
    args.push_back("--bandwidth_selection=plugin");
    args.push_back("--kernel=gaussian");
    fl::ml::Kde<boost::mpl::void_>::Run(ws_.get(), args);
    boost::shared_ptr<typename fl::ws::WorkSpace::DefaultTable_t> bandwidth_table;
    ws_->Attach("bandwidth", &bandwidth_table);
    bandwidth_=bandwidth_table->get(index_t(0), index_t(0));
    // we need to also derrive the bandwidth
    // Let's call KDE to get the plugin bandwidth
  }

  template<typename TableType>
  double KdeDistribution<TableType>::Eval(const Point_t &point) {
    if (references_->n_entries()==0) {
      fl::logger->Warning()<<"Attempt to evaluate an empty Kde distribution "
        "returning default";
      return 0;
    }
    boost::shared_ptr<Table_t> query_table;
    ws_->Attach("query",
        references_->dense_sizes(),
        references_->sparse_sizes(),
        0,
        &query_table);   
    query_table->push_back(const_cast<Point_t&>(point));
    ws_->Purge("query");
    ws_->Detach("query");
    ws_->IndexTable("query", 
                    "l2",
                    "",
                    2);
    std::vector<std::string> args=args_;
    args.push_back("--queries_in=query");
    args.push_back("--kernel=gaussian");
    args.push_back("--densities_out=density");
    args.push_back("--bandwidth="
        +boost::lexical_cast<std::string>(bandwidth_));
    fl::ml::Kde<boost::mpl::void_>::Run(ws_.get(), args);
    boost::shared_ptr<fl::ws::WorkSpace::DefaultTable_t> result_table;
    ws_->Attach("density", &result_table);
    double result= result_table->get(index_t(0), index_t(0));
    if (result==0) {
      std::cout<<"bandwidth="<<bandwidth_<<" density="<<result
        <<"reference_count="<<references_->n_entries()<<std::endl;
      for(size_t i=0; i<args.size(); ++i) {
        std::cout<<args[i]<<" ";
      }
      std::cout<<std::endl;
      point.Print(std::cout, ",");
      std::cout<<std::endl;
      references_->Save();
      query_table->Save();
      result_table->filename()="density";
      result_table->Save();
      exit(0);
    }

    ws_->RemoveTable("density");
    ws_->RemoveTable("query");
    return result;
  }

  template<typename TableType>
  double KdeDistribution<TableType>::LogDensity(const Point_t &point) {
    return log(this->Eval(point));
  }

  template<typename TableType>
  template<typename WorkSpaceType>
  void KdeDistribution<TableType>::Export(
      const std::vector<std::string> &args, WorkSpaceType *ws) {
    
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "references_prefix_out",
      boost::program_options::value<std::string>(),
      "this is the filename for storing the histogram"
    )(
      "bandwidths_prefix_out",
      boost::program_options::value<std::string>(),
      "this is the filename for storing the bandwidths"  
    );
    boost::program_options::variables_map vm;
    boost::program_options::command_line_parser clp(args);
    clp.style(boost::program_options::command_line_style::default_style
       ^boost::program_options::command_line_style::allow_guessing);
    try {
      boost::program_options::store(clp.options(desc).run(), vm);
    }
    catch(const boost::program_options::invalid_option_value &e) {
  	  fl::logger->Die() << "Invalid Argument: " << e.what();
    }
    catch(const boost::program_options::invalid_command_line_syntax &e) {
  	  fl::logger->Die() << "Invalid command line syntax: " << e.what(); 
    }
    catch (const boost::program_options::unknown_option &e) {
       fl::logger->Die() << e.what()
        <<" . This option will be ignored";
    }
    catch ( const boost::program_options::error &e) {
      fl::logger->Die() << e.what();
    } 
    boost::program_options::notify(vm);
    boost::shared_ptr<TableType> references_table;
    std::string name1=vm["references_prefix_out"].as<std::string>()
        +boost::lexical_cast<std::string>(id_);
    ws->LoadTable(name1, references_table);   
    boost::shared_ptr<fl::ws::WorkSpace::DefaultTable_t> bandwidth_table;
    std::string name2=vm["bandwidths_prefix_out"].as<std::string>()
        +boost::lexical_cast<std::string>(id_);

    typename fl::ws::WorkSpace::DefaultTable_t::Point_t point;
    ws->Attach(name2,
        std::vector<index_t>(1, 1),
        std::vector<index_t>(),
        1,
        &bandwidth_table);
    bandwidth_table->get(0, &point);
    point.set(0, bandwidth_);
    ws->Purge(name2);
    ws->Detach(name2);
  }

  template<typename TableType>
  index_t KdeDistribution<TableType>::count() const {
    return references_->n_entries();
  }

}}
#endif
