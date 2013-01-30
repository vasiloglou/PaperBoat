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

#ifndef PAPERBOAT_INCLUDE_MLPACK_GMM_DISTRIBUTION_DEFS_H_
#define PAPERBOAT_INCLUDE_MLPACK_GMM_DISTRIBUTION_DEFS_H_

#include "gmm_distribution.h"
namespace fl { namespace ml {
  
  template<typename TableType, bool IS_DIAGONAL>
  void GmmDistribution<TableType, IS_DIAGONAL>::Init(
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
   
  } 
  
  template<typename TableType, bool IS_DIAGONAL>
  void GmmDistribution<TableType, IS_DIAGONAL>::ResetData() {
    const std::vector<index_t> &dense_sizes=references_->dense_sizes();
    const std::vector<index_t> &sparse_sizes=references_->sparse_sizes();
    references_.reset(new Table_t());
    references_->Init("",
        dense_sizes,
        sparse_sizes,
        0);
    references_ptr_=references_.get(); 

  }
  

  template<typename TableType, bool IS_DIAGONAL>
  void GmmDistribution<TableType, IS_DIAGONAL>::AddPoint(Point_t &point) {
    references_ptr_->push_back(point);
  }

  template<typename TableType, bool IS_DIAGONAL>
  void GmmDistribution<TableType, IS_DIAGONAL>::Train() {
    mutrans_w_mu_.clear();
    for(size_t i=0; i<n_gaussians_; ++i) {
      mutrans_w_mu_.push_back(
          fl::la::Dot(
            (*means_)[i], 
            (*covariances_)[i],
            (*means_)[i] 
          ));
      fl::la::DotMul<fl::la::Init>(
          (*covariances_)[i], 
          (*means_)[i], 
          &sigma_times_mu_[i]);    
    }   
  }

  template<typename TableType, bool IS_DIAGONAL>
  double GmmDistribution<TableType, IS_DIAGONAL>::Eval(const Point_t &point) {
    double result=0;
    for(size_t i=0; i<n_gaussians_; ++i) {
      double q_form=fl::la::Dot(point, (*covariances_)[i], point) 
        -2*fl::la::Dot(sigma_times_mu_[i], point)
        -mutrans_w_mu_[i];
      result+= (*priors_)[i] * exp(-q_form);
    } 
    return result;
  }

  template<typename TableType, bool IS_DIAGONAL>
  double GmmDistribution<TableType, IS_DIAGONAL>::LogDensity(const Point_t &point) {
    return log(Eval(point)); 
  }

  template<typename TableType, bool IS_DIAGONAL>
  template<typename WorkSpaceType>
  void GmmDistribution<TableType, IS_DIAGONAL>::Export(
      const std::vector<std::string> &args, WorkSpaceType *ws) {
 
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "covariance_prefix_out",
      boost::program_options::value<std::string>(),
      "this is the filename for storing the covariance"
    )(
      "mean_prefix_out",
      boost::program_options::value<std::string>(),
      "this is the filename for storing the bandwidths"  
    )(
      "prior_prefix_out",
      boost::program_options::value<std::string>(),
      "this is the filename for storing the prior"  
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
    boost::shared_ptr<Covariance_t> covariance_table;
    std::string name1=vm["covariance_prefix_out"].as<std::string>()
        +boost::lexical_cast<std::string>(id_);
       
    boost::shared_ptr<fl::table::DefaultTable> mean_table;
    std::string name2=vm["mean_prefix_out"].as<std::string>()
        +boost::lexical_cast<std::string>(id_);
 
    boost::shared_ptr<fl::table::DefaultTable> prior_table;
    std::string name3=vm["prior_prefix_out"].as<std::string>()
        +boost::lexical_cast<std::string>(id_);

  }

  template<typename TableType, bool IS_DIAGONAL>
  index_t GmmDistribution<TableType, IS_DIAGONAL>::count() const {
    return references_->n_entries(); 
  }
}}

#endif
