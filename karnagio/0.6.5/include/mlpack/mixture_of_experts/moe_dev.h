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

#ifndef FL_LITE_SRC_MLPACK_MIXTURE_OF_EXPERTS_MOE_DEV_H_
#define FL_LITE_SRC_MLPACK_MIXTURE_OF_EXPERTS_MOE_DEV_H_
#include "moe.h"

namespace fl {namespace ml {
  template<typename ExpertType>
  Moe<ExpertType>::Moe() {
    k_clusters_=0;
    iterations_=0;
    n_restarts_=0;
    error_tolerance_=0;
    log_=false;
  }
  template<typename ExpertType>
  void Moe<ExpertType>::Compute(std::vector<index_t> *memberships,
      std::vector<double> *cluster_scores) {
    
    fl::logger->Message()<<"Building a global expert"<<std::endl;
    Expert_t global_expert;
    global_expert.set(reference_table_);
    global_expert.set_args(expert_args_);
    global_expert.set_log(log_);
    global_expert.Build();
    fl::logger->Message()<<"The global expert goal is "
      <<global_expert.score()<<std::endl;
    std::vector<boost::shared_ptr<Table_t> > best_clusters;
    double best_score=std::numeric_limits<double>::max();
    std::vector<double> best_cluster_scores(k_clusters_);
    for(int32 restart=0; restart<n_restarts_; ++restart) {
      // if the number of restarts is more than 1 then there is no point
      // on setting initial clusters which means we have to re-assign them randomly
      if (restart>0) {
        this->set_initial_clusters();
      }
      for(int32 i=0; i<k_clusters_; ++i) {
        experts_[i]->set_args(expert_args_);
        experts_[i]->set_log(log_);
      }
      fl::logger->Message()<<"restart="<<restart<<std::endl;
      std::vector<boost::shared_ptr<Table_t> > clusters(k_clusters_);
      // Build the expert models
      double old_average_score=0;
      for(int32 i=0; i<k_clusters_; ++i) {
        experts_[i]->Build();
        old_average_score+=experts_[i]->cardinality() * experts_[i]->score();
      }   
      old_average_score/=reference_table_->n_entries();
      for(int32 iteration=0; iteration<iterations_; ++iteration) {
        // Now assign points to the best model
        for(int32 i=0; i<k_clusters_; ++i) {
          clusters[i].reset(new Table_t());
          clusters[i]->Init(std::string("dummy")+boost::lexical_cast<std::string>(i),
              reference_table_->dense_sizes(),
              reference_table_->sparse_sizes(), 
              0);
          clusters[i]->labels()=reference_table_->labels();
        }
        typename Table_t::Point_t point;
        for(index_t i=0; i<reference_table_->n_entries(); ++i) {
          reference_table_->get(i, &point);
          double best_score=std::numeric_limits<double>::max();
          int32 best_expert=-1;
          double score=best_score;
          for(int32 j=0; j<k_clusters_; ++j) {
            score=experts_[j]->Evaluate(point);
            if (score<best_score) {
              best_score=score;
              best_expert=j;
            }
          }
          DEBUG_ASSERT(best_expert>=0);
          clusters[best_expert]->push_back(point);
        }
        // if there are any constraints in the memberships
        // we need to rearrange the points
        if (predefined_memberships_.size()!=0) {
          std::map<int32, std::map<int32, index_t> > votes;
          std::vector<boost::shared_ptr<Table_t> > new_clusters(k_clusters_);   
          for(int32 i=0; i<k_clusters_; ++i) {
            new_clusters[i].reset(new Table_t());
            new_clusters[i]->Init(std::string("dummy")+boost::lexical_cast<std::string>(i),
                reference_table_->dense_sizes(),
                reference_table_->sparse_sizes(), 
                0);
            new_clusters[i]->labels()=reference_table_->labels();
          }
          for(int32 i=0; i<clusters.size(); ++i) {
            for(index_t j=0; j<clusters[i]->n_entries(); ++j) {
              clusters[i]->get(j, &point);
              index_t point_id=point.meta_data(). template get<2>();
              if (predefined_memberships_.count(point_id)>0) {
                int32 bucket_id=predefined_memberships_[point_id];
                votes[bucket_id][i]+=1;
              }
            }
          }
          // maps the predefined cluster id to the real one
          std::map<int32, int32> c2c;
          for(std::map<int32, std::map<int32, index_t> >::iterator it1=votes.begin();
              it1!=votes.end(); ++it1) {
            int32 best_cluster=-1;
            int32 best_votes=0;
            for(std::map<int32, index_t>::iterator it2=it1->second.begin();
                it2!=it1->second.end(); ++it2) {
              if (it2->second>best_votes) {
                best_votes=it2->second;
                best_cluster=it2->first;
              }
            }
            c2c[it1->first]=best_cluster;
          }
          
          for(int32 i=0; i<clusters.size(); ++i) {
            for(index_t j=0; j<clusters[i]->n_entries(); ++j) {
              clusters[i]->get(j, &point);
              index_t point_id=point.meta_data(). template get<2>();
              if (predefined_memberships_.count(point_id)>0) {
                int32 bucket_id=predefined_memberships_[point_id];
                new_clusters[c2c[bucket_id]]->push_back(point);
              } else {
                new_clusters[i]->push_back(point);
              }
            }
          }
          clusters=new_clusters;
        }
        // check if there is any empty cluster
        std::vector<index_t> cluster_sizes(k_clusters_);
        std::vector<int32> empty_clusters;
        for(int32 i=0; i<k_clusters_; ++i) {
          if (clusters[i]->n_entries()==0) {
            empty_clusters.push_back(i);     
          }
          cluster_sizes[i]=clusters[i]->n_entries();
        }
        if (empty_clusters.empty()==false) {
          fl::logger->Warning()<<empty_clusters.size() <<" empty clusters found"
            <<std::endl;
          // find the biggest_cluster
          size_t max_element=-1;
          index_t max_value=0;
          for(size_t i=0; i<cluster_sizes.size(); ++i) {
            if (cluster_sizes[i]>max_value) {
              max_value=cluster_sizes[i];
              max_element=i;
            }
          }
  
          std::vector<boost::shared_ptr<Table_t> > new_clusters=clusters[max_element]->Split(
              empty_clusters.size()+1, "random_unique,0");
          clusters[max_element] = new_clusters[0];
          for(size_t i=0; i<empty_clusters.size(); ++i) {
            clusters[empty_clusters[i]]=new_clusters[i+1];
          }
        } 
        for(int32 i=0; i<k_clusters_; ++i) {
          experts_[i]->set(clusters[i]);
        }
        double average_score=0;
        for(int32 i=0; i<k_clusters_; ++i) {
          experts_[i]->Build();
          average_score+=experts_[i]->cardinality() * experts_[i]->score();
        }    
        average_score/=reference_table_->n_entries();
        fl::logger->Message()<<"iteration="<<iteration
            <<", average_expert_score="<<average_score<<std::endl;

        if (fabs(old_average_score-average_score)/old_average_score
            <error_tolerance_) {
          break;
        }
        old_average_score=average_score;
        if (old_average_score<best_score && empty_clusters.empty()) {
          best_score=old_average_score;
          best_clusters=clusters;
          for(int32 i=0; i<k_clusters_;++i) {
            best_cluster_scores[i]=experts_[i]->score();
          }
        }
      }
    }
    if (best_score==std::numeric_limits<double>::max()) {
      fl::logger->Die()<<"Failed to find a solution with non-empty clusters";
    }
    fl::logger->Message()<<"Finished task, best score: "<<best_score<<std::endl;
    // collect the results and export them  
    if (memberships!=NULL) {
      memberships->resize(reference_table_->n_entries());
      for(int32 i=0; i<k_clusters_; ++i) {
        Point_t point;    
        for(index_t j=0; j<best_clusters[i]->n_entries(); ++j) {
          best_clusters[i]->get(j, &point);
          memberships->operator[](point.meta_data().template get<2>())=i;
        }     
      }
    }
    // integrity check to make sure the predefined_arguments was respected
#ifdef DEBUG
    if (memberships!=NULL) {
      for(std::map<index_t, int32>::iterator it1=predefined_memberships_.begin(); 
          it1!=predefined_memberships_.end(); ++it1) {
        for(std::map<index_t, int32>::iterator it2=predefined_memberships_.begin(); 
            it2!=predefined_memberships_.end(); ++it2) {
          if (it1->second==it2->second) {
            int32 m1=memberships->operator[](it1->first);
            int32 m2=memberships->operator[](it2->first); 
            if (m1!=m2) {
              fl::logger->Message()<<"Integrity error "<<"("<<it1->first<<","<<m1<<") "
              <<"("<<it2->first<<","<<m2<<") ";
            }
          }
        }
      } 
    }
#endif
    if (cluster_scores!=NULL) {
      cluster_scores->resize(k_clusters_);
      for(int32 i=0; i<k_clusters_; ++i) {
        cluster_scores->operator[](i)=best_cluster_scores[i];
      }
    }
  }

  template<typename ExpertType>
  void Moe<ExpertType>::set_predefined_memberships(std::map<index_t, int32> &predefined_memberships) {
    predefined_memberships_=predefined_memberships; 
  }

  template<typename ExpertType>
  void Moe<ExpertType>::set_references(boost::shared_ptr<Table_t> references) {
    reference_table_=references;
  }

  template<typename ExpertType>
  void Moe<ExpertType>::set_expert_args(const std::vector<std::string> &args) {
    expert_args_=args;
  }

  template<typename ExpertType>
  void Moe<ExpertType>::set_expert_log(bool log) {
    log_=log;
  }

  template<typename ExpertType>
  void Moe<ExpertType>::set_k_clusters(int32 k_clusters) {
    k_clusters_=k_clusters;
  }

  template<typename ExpertType>
  void Moe<ExpertType>::set_iterations(int32 iterations) {
    iterations_=iterations;
  }

  template<typename ExpertType>
  void Moe<ExpertType>::set_n_restarts(int32 n_restarts) {
    n_restarts_=n_restarts;
  }

  template<typename ExpertType>
  void Moe<ExpertType>::set_error_tolerance(double error_tolerance) {
    error_tolerance_=error_tolerance;
  }

  template<typename ExpertType>
  void Moe<ExpertType>::set_initial_clusters(std::vector<boost::shared_ptr<Table_t> > 
       &initial_clusters) { 
    experts_.resize(k_clusters_);
    for(unsigned int i=0; i<experts_.size(); ++i) {
      experts_[i].reset(new Expert_t());
      experts_[i]->set(initial_clusters[i]);
    }
  }
  
  
  template<typename ExpertType>
  void Moe<ExpertType>::set_initial_clusters() {
    std::vector<boost::shared_ptr<Table_t> > initial_clusters=
        reference_table_->Split(k_clusters_, "random_unique,1");
    set_initial_clusters(initial_clusters); 
  }


}}
#endif

