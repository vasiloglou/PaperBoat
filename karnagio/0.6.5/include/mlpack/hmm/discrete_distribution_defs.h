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

#ifndef PAPERBOAT_INCLUDE_MLPACK_DISCRETE_DISTRIBUTION_DEFS_H_
#define PAPERBOAT_INCLUDE_MLPACK_DISCRETE_DISTRIBUTION_DEFS_H_

#include "discrete_distribution.h"
namespace fl { namespace ml {
  template<typename TableType>
  void DiscreteDistribution<TableType>::Init(
      const std::vector<std::string> &args,
      int32 id,
      const std::vector<index_t> &dense_sizes, 
      const std::vector<index_t> &sparse_sizes) {
    references_.reset(new Table_t());
    references_->Init("",
        dense_sizes,
        sparse_sizes,
        0);
    references_ptr_=references_.get(); 
    id_=id; 
  } 
 
  template<typename TableType>
  template<typename WorkSpaceType>
  void DiscreteDistribution<TableType>::Import(
      const std::vector<std::string> &import_args,
      const std::vector<std::string> &exec_args,
      WorkSpaceType *ws,
      int32 id) {
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "histogram_prefix_in",
      boost::program_options::value<std::string>(),
      "this is the filename for storing the histogram"
    )(
      "labels_prefix_in",
      boost::program_options::value<std::string>(),
      "this is the filename for storing the labels"  
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
    boost::shared_ptr<fl::table::DefaultTable> labels_table;
    boost::shared_ptr<fl::table::DefaultTable> hist_table;
    // We will output the labels that are integers ranging possibly
    // from -inf to +inf 
    std::string hist_name=vm["histogram_prefix_in"].as<std::string>()
        +boost::lexical_cast<std::string>(id_);
    std::string labels_name=vm["labels_prefix_in"].as<std::string>()
        +boost::lexical_cast<std::string>(id_);
    ws->Attach(
        labels_name,
        &labels_table);    
    ws->Attach(
        hist_name,
        &hist_table);  
    typename fl::table::DefaultTable::Point_t point1, point2;
    labels_table->get(0, &point1);
    hist_table->get(0, &point2);
    if (point1.size()!=point2.size()) {
      fl::logger->Die()<<"The labels table and histogram table for id="
        <<id_<<" do not have the same dimensions";
    }
    for(index_t i=0; i<point1.size(); ++i) {
      distribution_[point1[i]]=point2[i];   
    }
  }
 
  template<typename TableType>
  void DiscreteDistribution<TableType>::ResetData() {
    const std::vector<index_t> &dense_sizes=references_->dense_sizes();
    const std::vector<index_t> &sparse_sizes=references_->sparse_sizes();
    references_.reset(new Table_t());
    references_->Init("",
        dense_sizes,
        sparse_sizes,
        0);
    references_ptr_=references_.get(); 
    total_sum_=0;
  }
  
  template<typename TableType>
  void DiscreteDistribution<TableType>::AddPoint(Point_t &point) {
    references_ptr_->push_back(point);
    distribution_[point[0]]+=1;
    total_sum_+=1;
  }

  template<typename TableType>
  void DiscreteDistribution<TableType>::Train() {
    for(std::map<index_t, double>::iterator it=distribution_.begin();
        it!=distribution_.end(); ++it) {
      it->second/=total_sum_;
    }
  }

  template<typename TableType>
  double DiscreteDistribution<TableType>::Eval(const Point_t &point) {
    return distribution_[point[0]]; 
  }

  template<typename TableType>
  double DiscreteDistribution<TableType>::LogDensity(const Point_t &point) {
    return log(distribution_[point[0]]); 
  }
  template<typename TableType>
  template<typename WorkSpaceType>
  void DiscreteDistribution<TableType>::Export(
      const std::vector<std::string> &args, WorkSpaceType *ws) {
    
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "histogram_prefix_out",
      boost::program_options::value<std::string>(),
      "this is the filename for storing the histogram"
    )(
      "labels_prefix_out",
      boost::program_options::value<std::string>(),
      "this is the filename for storing the labels"  
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
    boost::shared_ptr<fl::table::DefaultTable> labels_table;
    boost::shared_ptr<fl::table::DefaultTable> hist_table;
    // We will output the labels that are integers ranging possibly
    // from -inf to +inf 
    std::string hist_name=vm["histogram_prefix_out"].as<std::string>()
        +boost::lexical_cast<std::string>(id_);
    std::string labels_name=vm["labels_prefix_out"].as<std::string>()
        +boost::lexical_cast<std::string>(id_);
    index_t range=distribution_.end()->first-distribution_.begin()->first;
    ws->Attach(
        labels_name,
        std::vector<index_t>(), 
        std::vector<index_t>(1, range),
        1,
        &labels_table);    
    ws->Attach(
        hist_name,
        std::vector<index_t>(), 
        std::vector<index_t>(1, range),
        1,
        &hist_table);  
    typename fl::table::DefaultTable::Point_t point1, point2;
    hist_table->get(0, &point1);
    labels_table->get(0, &point2);
    index_t count=0;
    for(std::map<index_t, double>::const_iterator it=distribution_.begin();
        it!=distribution_.end(); ++it) {
      point1.set(count, it->first);
      point2.set(count, it->second);
      count++;
    }
    ws->Purge(hist_name);
    ws->Detach(hist_name);
    ws->Purge(labels_name);
    ws->Detach(labels_name);

  }

  template<typename TableType>
  index_t DiscreteDistribution<TableType>::count() const {
    return references_->n_entries();
  }
}}

#endif
