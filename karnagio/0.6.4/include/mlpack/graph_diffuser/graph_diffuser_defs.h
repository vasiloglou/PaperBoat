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

#ifndef PAPERBOAT_INCLUDE_MLPACK_GRAPH_DIFFUSER_GRAPH_DIFFUSER_DEFS_H_
#define PAPERBOAT_INCLUDE_MLPACK_GRAPH_DIFFUSER_GRAPH_DIFFUSER_DEFS_H_

#include "graph_diffuser.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "fastlib/workspace/arguments.h"
#include "mlpack/allkn/allkn.h"
#include "mlpack/svd/svd.h"
#include "mlpack/nmf/nmf.h"
#include "fastlib/util/string_utils.h"

namespace fl { namespace ml {
  
  template<typename WorkSpaceType>
  void GraphDiffuser<WorkSpaceType>::Normalize(std::vector<std::pair<index_t, double> >::iterator begin,
      std::vector<std::pair<index_t, double> >::iterator end,
      int norm) {

    if (norm==1) {
      double sum=0;
      for(std::vector<std::pair<index_t, double> >::iterator it=begin;
        it!=end; ++it) {
        sum+=it->second;
      }
      for(std::vector<std::pair<index_t, double> >::iterator it=begin;
          it!=end; ++it) {
        it->second/=sum;
      }
    }
    if (norm==2) {
      double sum=0;
      for(std::vector<std::pair<index_t, double> >::iterator it=begin;
          it!=end; ++it) {
        sum+=it->second *it->second;
      }
      for(std::vector<std::pair<index_t, double> >::iterator it=begin;
          it!=end; ++it) {
        it->second/=sum;
      }
    }
  }

  template<typename WorkSpaceType>
  template<typename PointType>
  void GraphDiffuser<WorkSpaceType>::Normalize(
      PointType *point,
      int norm) {
   FL_SCOPED_LOG(Normalize);
   if (norm==0) {
     for(typename PointType::iterator it=point->begin();
       it!=point->end(); ++it) {
       if (it.value()>0) {
         it.value()=1;
         continue;
       } 
       if (it.value()<0) {
         it.value()=-1;
         continue;
       }
     } 
   } else {
     if (norm==1) {
       double l1=0;
       for(typename PointType::iterator it=point->begin();
           it!=point->end(); ++it) {
         l1+=fabs(it.value());
       } 
       if (l1==0) {
         fl::logger->Warning()<<"Label vector is zero"<<std::endl;
         return;
       }
       fl::la::SelfScale(1.0/l1, point); 
     } else {
       if (norm==2) {
         double l2=fl::la::LengthEuclidean(*point); 
         if (l2==0) {
           fl::logger->Warning()<<"Label vector is zero"<<std::endl;
           return;
         }
         fl::la::SelfScale(1.0/l2, point);         
       }
     }
   }
  }
  template<typename WorkSpaceType>
  template<typename EdgeFunctorType, typename GraphTableType>
  void GraphDiffuser<WorkSpaceType>::MakeGraph(
                WorkSpaceType *ws, 
                const std::string &indices_name, 
                const std::string &weights_name, 
                const EdgeFunctorType *edge_functor,
                const std::string &normalization,
                bool make_symmetric,
                const std::string &graph_name) {
    FL_SCOPED_LOG(MakeGraph);
    boost::shared_ptr<typename WorkSpaceType::UIntegerTable_t> indices_table;
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> weights_table;
    ws->Attach(indices_name, &indices_table);
    ws->Attach(weights_name, &weights_table);
    boost::shared_ptr<GraphTableType> graph;
    ws->Attach(graph_name,
            std::vector<index_t>(),
            std::vector<index_t>(1, indices_table->n_entries()),
            indices_table->n_entries(),
            &graph);
    typename WorkSpaceType::UIntegerTable_t::Point_t i_point;
    typename WorkSpaceType::DefaultTable_t::Point_t w_point; 
    typename GraphTableType::Point_t g_point;
    int norm=0;
    if (normalization=="l1") {
      norm=1;
    } else {
      if (normalization=="l2") {
        norm=2;
      } else {
        if (normalization=="none") {
          norm=3;
        } else {
          fl::logger->Die()<<"This graph normalization ("
              <<normalization <<") is not supported";
        }
      }
    }
    if (make_symmetric==false) {
       for(index_t i=0; i<indices_table->n_entries(); ++i) {
         std::vector<std::pair<index_t, double> > load_cont;
         indices_table->get(i, &i_point);
         weights_table->get(i, &w_point);
         for(int j=0; j<i_point.size(); ++j) {
           load_cont.push_back(std::make_pair(i_point[j], w_point[j]));
         }
         graph->get(i, &g_point);
         if (edge_functor!=NULL) {
           for(std::vector<std::pair<index_t, double> >::iterator 
               it=load_cont.begin(); it!=load_cont.end(); ++it) {
             (*edge_functor)(&(it->second));
           }
         }
         std::sort(load_cont.begin(), load_cont.end());
         Normalize(load_cont.begin(), load_cont.end(), norm);
         std::vector<std::pair<index_t, double> >::iterator it1,it2;
         it1=load_cont.begin();
         it2=load_cont.end();
         // This has to change in a future version
         g_point.template sparse_point<double>().Load(it1, it2);

       }
     } else {
       std::vector<std::vector<std::pair<index_t, double> > > 
         temp_graph(indices_table->n_entries());
       for(index_t i=0; i<indices_table->n_entries(); ++i) {
         indices_table->get(i, &i_point);
         weights_table->get(i, &w_point);
         for(int j=0; j<i_point.size(); ++j) {
           // in this case we have to push both neighbor pairs
           // (i, i_poin[j]) (i_point[j], i)
           temp_graph[i].push_back(std::make_pair(i_point[j], w_point[j]));
           temp_graph[i_point[j]].push_back(std::make_pair(i, w_point[j]));
         }
       } 

       // The above strategy will create redundancy pairs which we have
       // to remove before loading the points to the graph
       for(size_t i=0; i<temp_graph.size(); ++i) {
         std::vector<std::pair<index_t, double> >::iterator 
         it=std::unique(temp_graph[i].begin(), temp_graph[i].end());
         graph->get(i, &g_point);
         if (edge_functor!=NULL) {
           for(std::vector<std::pair<index_t, double> >::iterator 
               it1=temp_graph[i].begin(); it1!=it; ++it1) {
             (*edge_functor)(&(it1->second));
           }
         }
         Normalize(temp_graph[i].begin(), it, norm);
         // This has to change in a future version
         g_point.template sparse_point<double>().Load(temp_graph[i].begin(), it);
       }
     }
     ws->Purge(graph_name);
     ws->Detach(graph_name);
  }

  template<typename WorkSpaceType>
  template<typename GraphTableType>
  void GraphDiffuser<WorkSpaceType>::MakeGraph(
                WorkSpaceType *ws, 
                const std::string &indices_name, 
                const std::string &weights_name, 
                const std::string &edge_option,
                const std::string &normalization,
                bool make_symmetric,
                const std::string &graph_name) {
    if (edge_option!="none" &&
        edge_option!="dist" &&
        edge_option!="1/dist" &&
        edge_option!="exp(-dist/h)") {
      fl::logger->Die()<<"--weight_policy="<<edge_option
          <<" is not a valid option";
    }
    if (edge_option=="none") { 
      UnitEdge unit_edge;
      MakeGraph<UnitEdge, GraphTableType>(ws, 
        indices_name, 
        weights_name, 
        &unit_edge,
        normalization,
        make_symmetric,
        graph_name);
    } else {
      if (edge_option=="dist") {
        MakeGraph<UnitEdge, GraphTableType>(ws, 
          indices_name, 
          weights_name, 
          NULL,
          normalization,
          make_symmetric,
          graph_name);
      } else {
        if (edge_option=="1/dist") {
          DistInvEdge dist_inv_edge;
          MakeGraph<DistInvEdge, GraphTableType>(ws, 
            indices_name, 
            weights_name, 
            &dist_inv_edge,
            normalization,
            make_symmetric,
            graph_name);
        } else {
          if (edge_option=="exp(-dist/h)") {
            GaussEdge func;
            func.set_h(1.0);
            MakeGraph<GaussEdge, GraphTableType>(ws, 
              indices_name, 
              weights_name, 
              &func,
              normalization,
              make_symmetric,
              graph_name);
          }
        }
      }     
    }
  }
 
  template<typename WorkSpaceType>
  void GraphDiffuser<WorkSpaceType>::DotProdPolicy::Init(double val) {
    result_=val;
  }

  template<typename WorkSpaceType>
  void GraphDiffuser<WorkSpaceType>::DotProdPolicy::Reset() {
    result_=0;
  }

  template<typename WorkSpaceType>
  template<typename PointType1, typename PointType2>
  void GraphDiffuser<WorkSpaceType>::DotProdPolicy::PointEval(PointType1 &p1, PointType2 &p2) {
    // It is important that p2 which is the vector of labels goes first
    // in the linear algebra Dot. Otherwise it will not work
    result_=fl::la::Dot(p2, p1);
  }
  
  template<typename WorkSpaceType>
  template<typename T1, typename T2>
  void GraphDiffuser<WorkSpaceType>::DotProdPolicy::Update(T1 val1, T2 val2) {
    result_+=val1*val2;
  }

  template<typename WorkSpaceType>
  double GraphDiffuser<WorkSpaceType>::DotProdPolicy::Result() {
    return result_; 
  }

 
  template<typename WorkSpaceType>
  void GraphDiffuser<WorkSpaceType>::Diffuse(
      WorkSpaceType *ws, 
      const std::string &graph_name,
      const std::string &decision_policy, 
      int32 iterations,  
      boost::shared_ptr<
          typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_right_table, 
      const std::string &right_normalization,
      const std::string &right_labels_out) {
   
    std::map<std::string, int> norm2int; 
    norm2int["none"]=-1;
    norm2int["clip"]=0;
    norm2int["l1"]=1;
    norm2int["l2"]=2;

    if (decision_policy=="*vote") {
      fl::logger->Die()<<"*vote"<<NOT_SUPPORTED_MESSAGE; 
    } else {
      if (decision_policy=="+max") {
        fl::logger->Die()<<"+max"<<NOT_SUPPORTED_MESSAGE;
      } else {
        if (decision_policy=="+min") {
            fl::logger->Die()<<"+min"<<NOT_SUPPORTED_MESSAGE;
        } else {
          if (decision_policy=="*sum") {
            DotProdPolicy policy;
            DiffuseStruct<DotProdPolicy> task(ws,
                    graph_name,
                    policy,
                    iterations,
                    predefined_right_table,
                    norm2int[right_normalization],
                    right_labels_out);
            fl::ws::BasedOnTableRun(ws, graph_name, task);     
          } else {
            if (decision_policy=="*rsum") {
              fl::logger->Die()<<"*rsum"<<NOT_SUPPORTED_MESSAGE;     
            } else {
              fl::logger->Die()<<"This decision policy ("
                <<decision_policy<<") is not supported";
            }
          }
        }
      }
    }
  }
      
  template<typename WorkSpaceType>
  template<typename DecisionPolicyType>
  GraphDiffuser<WorkSpaceType>::
      DiffuseStruct<DecisionPolicyType>::DiffuseStruct(
        WorkSpaceType *ws1,
        const std::string &graph_name1,
        DecisionPolicyType &decision_policy1,
        int32 iterations1,
        boost::shared_ptr<
          typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_right_table1,
        const int right_normalization1,
        const std::string &right_labels_out1) :
            ws(ws1),
            graph_name(graph_name1),
            decision_policy(decision_policy1),
            iterations(iterations1),
            predefined_right_table(predefined_right_table1),
            right_normalization(right_normalization1),
            right_labels_out(right_labels_out1) {}


  template<typename WorkSpaceType>
  template<typename DecisionPolicyType>
  template<typename GraphTableType>
  void GraphDiffuser<WorkSpaceType>::DiffuseStruct<DecisionPolicyType>::operator()
      (GraphTableType&) {
    FL_SCOPED_LOG(Symmetric); 
    boost::shared_ptr<GraphTableType> graph;
    ws->Attach(graph_name, &graph);
    if (graph->n_entries()!=graph->n_attributes()) {
      fl::logger->Die()<<"The graph is not symmetric, you should "
        "set the --symmetric_diffusion=0";
    }
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t>
      right_labels_table;

    ws->Attach(right_labels_out,
        std::vector<index_t>(1, graph->n_attributes()),
        std::vector<index_t>(),
        1,
        &right_labels_table);
    typename WorkSpaceType::DefaultTable_t::Point_t 
      right_labels, right_labels1;

    right_labels_table->get(0, &right_labels);
    right_labels.SetAll(0.0);
    right_labels1.Copy(right_labels);
    
    typedef typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t PPoint_t; 
    PPoint_t predefined_right_point;
    predefined_right_table->get(0, &predefined_right_point);

    for(int32 iter=0; iter<iterations; ++iter) { 
      // propagate labels
      typename GraphTableType::Point_t point;
      for(index_t i=0; i<graph->n_entries(); ++i) {
        graph->get(i, &point);
        decision_policy.PointEval(point, 
              right_labels);
        right_labels1.set(i, decision_policy.Result()); 
      }
      // pin down the values of the nodes you know
      for(typename PPoint_t::iterator it=predefined_right_point.begin();
          it!=predefined_right_point.end(); ++it) {
        right_labels1.set(it.attribute(), it.value());
      }
      right_labels1.SwapValues(&right_labels);
      Normalize(&right_labels, right_normalization);
      fl::logger->Message()<<"Finished iteration="<<iter<<std::endl;
    }
    ws->Purge(right_labels_out);
    ws->Detach(right_labels_out);
  }

  template<typename WorkSpaceType>
  void GraphDiffuser<WorkSpaceType>::DiffuseBipartite(
      WorkSpaceType *ws, 
      const std::string &graph_name,
      const std::string &decision_policy, 
      int32 iterations,  
      boost::shared_ptr<
          typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_right_table, 
      const std::string &right_normalization,
      boost::shared_ptr<
          typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_left_table, 
      const std::string &left_normalization,
      const std::string &right_labels_out,
      const std::string &left_labels_out) {

    std::map<std::string, int> norm2int;
    norm2int["none"]=-1;
    norm2int["clip"]=0;
    norm2int["l1"]=1;
    norm2int["l2"]=2;

    if (decision_policy=="*vote") {
      fl::logger->Die()<<"*vote"<<NOT_SUPPORTED_MESSAGE; 
    } else {
      if (decision_policy=="+max") {
        fl::logger->Die()<<"+max"<<NOT_SUPPORTED_MESSAGE;
      } else {
        if (decision_policy=="+min") {
            fl::logger->Die()<<"+min"<<NOT_SUPPORTED_MESSAGE;
        } else {
          if (decision_policy=="*sum") {
            DotProdPolicy policy;
            DiffuseBipartiteStruct<DotProdPolicy> task(ws,
                    graph_name,
                    policy,
                    iterations,
                    predefined_right_table,
                    norm2int[right_normalization],
                    predefined_left_table,
                    norm2int[left_normalization],
                    right_labels_out,
                    left_labels_out);
            BasedOnTableRun(ws, graph_name, task);     
          } else {
            if (decision_policy=="*rsum") {
              fl::logger->Die()<<"*rsum"<<NOT_SUPPORTED_MESSAGE;     
            } else {
              fl::logger->Die()<<"This decision policy ("
                <<decision_policy<<") is not supported";
            }
          }
        }
      }
    }
  }

  template<typename WorkSpaceType>
  template<typename DecisionPolicyType>
  GraphDiffuser<WorkSpaceType>::DiffuseBipartiteStruct<DecisionPolicyType>::
      DiffuseBipartiteStruct(
             WorkSpaceType *ws1,
             const std::string &graph_name1,
             DecisionPolicyType &decision_policy1,
             int32 iterations1,
             boost::shared_ptr<
               typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_right_table1,
             const int right_normalization1,
             boost::shared_ptr<
               typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_left_table1,
             const int left_normalization1,
             const std::string &right_labels_out1,
             const std::string &left_labels_out1) :
        
      ws(ws1),
      graph_name(graph_name1),
      decision_policy(decision_policy1),
      iterations(iterations1),
      predefined_right_table(predefined_right_table1),
      right_normalization(right_normalization1),
      predefined_left_table(predefined_left_table1),
      left_normalization(left_normalization1),
      right_labels_out(right_labels_out1),
      left_labels_out(left_labels_out1) {}


  template<typename WorkSpaceType>
  template<typename DecisionPolicyType>
  template<typename GraphTableType>
  void GraphDiffuser<WorkSpaceType>::DiffuseBipartiteStruct<DecisionPolicyType>::
      operator()(GraphTableType&) {
   
    FL_SCOPED_LOG(Nonsymmetric); 
    boost::shared_ptr<GraphTableType> graph;
    ws->Attach(graph_name, &graph);

    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t>
      right_labels_table, left_labels_table;
    ws->Attach(left_labels_out,
        std::vector<index_t>(1,graph->n_entries()),
        std::vector<index_t>(),
        1,
        &left_labels_table);
    ws->Attach(right_labels_out,
        std::vector<index_t>(1,graph->n_attributes()),
        std::vector<index_t>(),
        1,
        &right_labels_table);
    typename WorkSpaceType::DefaultTable_t::Point_t left_labels,
             right_labels;
    left_labels_table->get(0, &left_labels);
    right_labels_table->get(0, &right_labels);

    left_labels.SetAll(0.0);
    right_labels.SetAll(0.0);
    typedef typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t PPoint_t; 
    PPoint_t predefined_left_point, predefined_right_point;
    if (predefined_left_table.get()!=NULL) {
      predefined_left_table->get(0, &predefined_left_point);
    }
    if (predefined_right_table.get()!=NULL) {
      predefined_right_table->get(0, &predefined_right_point);
    }
  
    // start the diffusion
    for(int32 it=0; it<iterations; ++it) { 
      // propagate labels
      typename GraphTableType::Point_t point;
      for(index_t i=0; i<graph->n_entries(); ++i) {
        graph->get(i, &point);
        decision_policy.Reset();
        decision_policy.PointEval(point, right_labels);
        left_labels.set(i, decision_policy.Result()); 
      }
      // pin down the left labels
      if (predefined_left_point.size()!=0) {
        for(typename PPoint_t::iterator it=predefined_left_point.begin();
            it!=predefined_left_point.end(); ++it) {
          left_labels.set(it.attribute(), it.value());
        }
      }
      Normalize(&left_labels, left_normalization);
      std::vector<DecisionPolicyType> aux(right_labels.size());
      for(size_t i=0; i<right_labels.size(); ++i) {
        aux[i].Init(right_labels[i]);
      }
      for(index_t i=0; i<graph->n_entries(); ++i) {
        graph->get(i, &point);
        for(typename GraphTableType::Point_t::iterator it=point.begin();
            it!=point.end(); ++it) {
          aux[it.attribute()].Update(
              it.value(),
              left_labels[i]);
          right_labels.set(it.attribute(), 
          aux[it.attribute()].Result()); 
        }
      }
      // pin down the values of the right nodes you know
      if (predefined_right_point.size()!=0) {
        for(typename PPoint_t::iterator it=predefined_right_point.begin();
            it!=predefined_right_point.end(); ++it) {
          right_labels.set(it.attribute(), it.value());
        }
      }
      Normalize(&right_labels, right_normalization);
      fl::logger->Message()<<"Finished iteration="<<it<<std::endl;
    }
    ws->Purge(right_labels_out);
    ws->Detach(right_labels_out);
    ws->Purge(left_labels_out);
    ws->Detach(left_labels_out);
  }

  template<typename WorkSpaceType>
  GraphDiffuser<boost::mpl::void_>::Core<WorkSpaceType>::Core(
     WorkSpaceType *ws, const std::vector<std::string> &args) :
   ws_(ws), args_(args)  {}


  template<typename WorkSpaceType>
  template<typename TableType>
  void GraphDiffuser<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
    FL_SCOPED_LOG(GraphDiffuser);
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_in",
      boost::program_options::value<std::string>(),
      "the reference data "
    )("summary",
      boost::program_options::value<std::string>()->default_value("none"),
      "If your data is high dimensional and you need to reduce its dimension "
      "you need to use a dimensionality reduction method to summarize your "
      "data in a lower dimensional space. Available options are: \n"
      "  none:  uses the data as they are without any summarization\n"
      "  svd :  uses svd to reduce the dimensionality\n"
      "  nmf :  uses nmf to reduce the dimensionality"
    )("connect_nodes",
      boost::program_options::value<std::string>()->default_value("none"),
      "After summarizing the data with either method, you might want to "
      "build a graph by connecting each point (node) with its nearest neighbor. "
      "Here are the available options: \n"
      "  none:  treats the summarized data as a bipartite graph points versus dimensions\n"
      "  snn :  symmetric euclidean nearest neighbors. Builds an undirected graph\n"
      "         making sure that each node is symmetrically connected to its neighbors\n"
      "  nn  :  nonsymetric euclidean nearest neighbors. Builds a directed graph by computing\n"
      "         the k-nearest neighbors for every point.\n"
    )("robust_nn",
      boost::program_options::value<bool>()->default_value(false),
      "If the summarization method is randomized then it might be a good idea "
      "to run summarization and node connection several times and aggregate "
      "the results. This will make the graph more stable"
    )("robust_nn_iterations",
      boost::program_options::value<int32>()->default_value(10),
      "If --robust_nn=1 then you need to inform the program how many times you "
      "need to run summarization and node connection"
    )("robust_nn_threshold",
      boost::program_options::value<int32>()->default_value(5),
      "if --robust_nn=1 then you need to inform the program how many times two "
      "nodes must appear as neighbors so that you should put an edge between them "
      "Notice the the --robust_nn_threshold must be less or equal to "
      "--robust_nn_iterations"
    )("run_diffusion",
      boost::program_options::value<bool>()->default_value(true),
      "if this variable is not set then the program only computes the graph, "
      "otherwise it runs diffusion with the given labels"
    )("symmetric_diffusion",
      boost::program_options::value<bool>()->default_value(true),
      "Setting this variable false makes sense only in bipartite graphs. In that case "
      "the diffusion process will generate a right_labeled and a left_labeled vector"
    )("right_labels_in",
      boost::program_options::value<std::string>(),
      "A sparse table with one sparse point only. The nonzero elements "
      "of the point have values that correspond to the labels of anchor points "
      "In case of a bipartite graph these labels correspond to the right nodes "
    )("left_labels_in",
      boost::program_options::value<std::string>(),
      "A sparse table with one sparse point only. The nonzero elements "
      "of the point have values that corresponf to the labels of anchor points "
      "In case of a bipartite graph these labels correspond to the left nodes" 
      "In case of a non-bipartite graph it is illegal to set this flag"   
    )("right_labels_out",
      boost::program_options::value<std::string>(),
      "Table name of the resulting labels. It is a table with 1 dimensional "
      "points. Each row contains the resulting label value after running the graph"
      "algorithm. These are the labels of the right nodes in case of a bipartite graph. " 
    )("right_labels_norm",
      boost::program_options::value<std::string>()->default_value("none"),
      "At every iteration you want then righ_labels vector to be normalized. "
      "Available options are: \n"
      " none:  no normalization\n"
      " clip:  it will clip positive values to +1 and negative values to -1\n"
      " l1  :  it will do an L1 normalization\n"
      " l2  :  it will do an L2 normalization"
    )("left_labels_out",
      boost::program_options::value<std::string>(),
      "Table name of the resulting labels. It is a table with 1 dimensional "
      "points. Each row contains the resulting label value after running the graph"
      "algorithm. These are the labels of the left nodes in case of a bipartite graph. " 
      "In case of a non-bipartite graph it is illegal to set this flag."
    )("left_labels_norm", 
      boost::program_options::value<std::string>()->default_value("none"),
      "At every iteration you want then left_labels vector to be normalized. "
      "Available options are: \n"
      " none:  no normalization\n"
      " clip:  it will clip positive values to +1 and negative values to -1\n"
      " l1  :  it will do an L1 normalization\n"
      " l2  :  it will do an L2 normalization"
    )("weight_policy",
      boost::program_options::value<std::string>()->default_value("none"),
      "When building the graph we need to put weights on the edges. Here are "
      "the available policy for assigning weights on the edges:\n"
      "  none        :  Edges are assigned the value 1\n"
      "  dist        :  Edges are assigned values equal to the distance between points (nodes)\n"
      "  1/dist      :  Edges are assigned values 1/distance between points (nodes)\n"
      "  exp(-dist/h):  Edges are assigned values exp(-dist/h)" 
    )("graph_out",
      boost::program_options::value<std::string>(),
      "name of the graph produced by the process"
    )("decision_policy",
      boost::program_options::value<std::string>()->default_value("*sum"),
      "At every iteration each node (point) collects info from its neighbors. "
      "It has to make a decision and form a value. Here are the available options:\n"
      "  *vote :  The node updates its value by picking the value with the most \n"
      "           votes from its neighbors\n"
      "  +max  :  The node updates its value by picking the maximum\n"
      "           value of its neighbors\n"
      "  -min  :  The node updates its value by picking the minimum\n"
      "           value of its neighbors\n"
      "  *sum  :  The node updates its value by picking the mean value of\n"
      "           its neighbors."
    )("diffusion_iterations",
      boost::program_options::value<int32>()->default_value(10),
      "number of iterations to run the diffusion"
    )("confidences_out",
      boost::program_options::value<std::string>(),
      "A table to output the confidence of every predicted label"
    )("confidence_method",
      boost::program_options::value<std::string>(),
      "Here are different methods for computing the prediction confidence:\n"
      "  pagerank   :  \n"
      "  exp(-var/h):  \n"
      "  combined   :  \n" 
    );
        
    boost::program_options::variables_map vm;
    std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args_, "");
    boost::program_options::command_line_parser clp(args1);
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
    if (vm.count("help")) {
      fl::logger->Message() << fl::DISCLAIMER << "\n";
      fl::logger->Message() << desc << "\n";
      return ;
    }

    if (vm.count("references_in")==0) {
      fl::logger->Die()<<"You must set the --references_in flag";
    }
    std::string references_in=vm["references_in"].as<std::string>();
    boost::shared_ptr<TableType> references_table;
    ws_->Attach(references_in, &references_table);
    
    std::string graph_name;
    std::string summary=vm["summary"].as<std::string>();
    bool robust_nn=vm["robust_nn"].as<bool>();
    int32 iterations=vm["robust_nn_iterations"].as<int32>();
    int32 threshold=vm["robust_nn_threshold"].as<int32>();

    if (summary=="none") {
      fl::logger->Message()<<"Skipping data summarization"<<std::endl;
      if (references_table->n_attributes()>100) {
        fl::logger->Warning()<<"Your dataset is high dimensional ("
          << references_table->n_attributes()<<") it might be a good idea "
          "to use a summarization (dimensionality reduction) method"<<std::endl;
      }
      if (vm["robust_nn"].as<bool>()==true) {
        fl::logger->Die()<<"Setting flag --robust_nn=1 when --summary=none "
          "does not make sense";
      }
    } else {
      if (vm["connect_nodes"].as<std::string>()=="none") {
        fl::logger->Die()<<"If you summarize your data then you have to pick "
          "a --connect_nodes method";
      }
      if (iterations<threshold) {
        fl::logger->Die()<<"--robust_nn_iterations="<<iterations
          <<" must be greater or equal to --robust_nn_threshold="
          <<threshold;
      }
      if (summary=="svd") {
        std::vector<std::string> svd_args=fl::ws::MakeArgsFromPrefix(args_, "svd");
        fl::logger->Message()<<"Summarizing your data with SVD"<<std::endl;
        fl::logger->Message()<<"Running SVD with the arguments filtered with the svd: prefix";
        //fl::logger->Message()<<"SVD arguments "<<svd_args<<std::endl;
        if (robust_nn==false) {
          svd_args.push_back("--references_in="+references_in);
          svd_args.push_back("--lsv_out=summarized_references");
          fl::ml::Svd<boost::mpl::void_>::Run(ws_, svd_args);
        } else {
          for(int32 i=0; i<iterations; ++i) {           
            std::vector<std::string> svd_args1=svd_args;
            svd_args1.push_back("--references_in="+references_in);
            svd_args1.push_back("--lsv_out=summarized_references"
                +boost::lexical_cast<std::string>(i));
            fl::ml::Svd<boost::mpl::void_>::Run(ws_, svd_args1);
          }
        } 
      } else {
        if (summary=="nmf") {
          std::vector<std::string> nmf_args=fl::ws::MakeArgsFromPrefix(args_, "nmf");
          fl::logger->Message()<<"Summarizing your data with NMF"<<std::endl;
          fl::logger->Message()<<"Running SVD with the arguments filtered with the nmf: prefix";
          //fl::logger->Message()<<"NMF arguments "<<nmf_args<<std::endl;
          if (robust_nn==false) {
              nmf_args.push_back("--references_in="+references_in);
            nmf_args.push_back("--w_out=summarized_references");
            fl::ml::Nmf::Run(ws_, nmf_args);
          } else {
            for(int32 i=0; i<iterations; ++i) {
              std::vector<std::string> nmf_args1=nmf_args;
              nmf_args1.push_back("--references_in="+references_in);
              nmf_args1.push_back("--w_out=summarized_references"
                  +boost::lexical_cast<std::string>(i));
              fl::ml::Nmf::Run(ws_, nmf_args1);
            } 
          }
        } else {
          fl::logger->Die()<<"This summarization method ("
            << summary<<") is not supported";
        }
      }
    }
    std::string connect_nodes=vm["connect_nodes"].as<std::string>();
    std::string weight_policy=vm["weight_policy"].as<std::string>();
    if (vm.count("graph_out")==false) {
      graph_name=ws_->GiveTempVarName();
    } else {
      graph_name=vm["graph_out"].as<std::string>();
    }
    std::vector<std::string> nn_args;
    nn_args=fl::ws::MakeArgsFromPrefix(args_, "allkn");
    if (connect_nodes!="none") {
      fl::logger->Message()<<"Connecting the nodes"<<std::endl;
      fl::logger->Message()<<"Connecting the nodes with nearest neighbors"<<std::endl;
      fl::logger->Message()<<"Running Allkn with arguments filtered with the nn: prefix"<<std::endl;
    }
    if (connect_nodes=="none") {
      if (vm.count("graph_out")) {
        fl::logger->Die()<<"When --connect_node=none then --graph_out must not be set";
      }
    }
    //fl::logger->Message()<<"ALLKN arguments "<<nn_args<<std::endl;
    std::string allkn_references;
    if (summary=="none") {
      allkn_references=references_in;
    } else {
      allkn_references="summarized_references";
    }
    const std::string indices_name="indices";
    const std::string weights_name="weights";
    if (connect_nodes=="snn") {
      if (robust_nn==false) {
        std::vector<std::string> nn_args1=nn_args;
        nn_args1.push_back("--references_in="+allkn_references);
        nn_args1.push_back("--indices_out="+indices_name);
        nn_args1.push_back("--distances_out="+weights_name);
        ws_->IndexAllReferencesQueries(&nn_args1);
        fl::ml::AllKN<boost::mpl::void_>::Run(ws_, nn_args1);           
        GraphDiffuser<WorkSpaceType>::
         template  MakeGraph<typename WorkSpaceType::DefaultSparseDoubleTable_t>(ws_, 
                  indices_name, 
                  weights_name, 
                  weight_policy,
                  "none",
                  true,
                  graph_name);
      } else {
        for(int32 i=0; i<iterations; ++i) {
          std::vector<std::string> nn_args1=nn_args;
          nn_args1.push_back("--references_in="+allkn_references
              +boost::lexical_cast<std::string>(i));
          nn_args1.push_back("--indices_out="+indices_name
              +boost::lexical_cast<std::string>(i));
          nn_args1.push_back("--distances_out="+weights_name
              +boost::lexical_cast<std::string>(i));
          ws_->IndexAllReferencesQueries(&nn_args1);
          fl::ml::AllKN<boost::mpl::void_>::Run(ws_, nn_args1); 
        }        
      }
    } else { 
      if (connect_nodes=="nn") {
        if (robust_nn==false) {
          std::vector<std::string> nn_args;
          nn_args=fl::ws::MakeArgsFromPrefix(args_, "allkn");
          nn_args.push_back("--references_in="+allkn_references);
          nn_args.push_back("--indices_out="+indices_name);
          nn_args.push_back("--distances_out="+weights_name);
          ws_->IndexAllReferencesQueries(&nn_args);
          fl::ml::AllKN<boost::mpl::void_>::Run(ws_, nn_args); 
          GraphDiffuser<WorkSpaceType>::template 
          MakeGraph<typename WorkSpaceType::DefaultSparseDoubleTable_t>(ws_, 
                    indices_name, 
                    weights_name, 
                    weight_policy,
                    "none",
                    false,
                    graph_name);
        } else {
          fl::logger->Die()<<"Robust nearest neighbor"
            <<NOT_SUPPORTED_MESSAGE;
          for(int32 i=0; i<iterations; ++i) {
            std::vector<std::string> nn_args;
            nn_args=fl::ws::MakeArgsFromPrefix(args_, "allkn");
            nn_args.push_back("--references_in="+allkn_references
                +boost::lexical_cast<std::string>(i));
            nn_args.push_back("--indices_out="+indices_name
                +boost::lexical_cast<std::string>(i));
            nn_args.push_back("--distances_out="+weights_name
                +boost::lexical_cast<std::string>(i));
            ws_->IndexAllReferencesQueries(&nn_args);
            fl::ml::AllKN<boost::mpl::void_>::Run(ws_, nn_args); 
          }        
        }
      } else {
        if (connect_nodes=="none") {
          // do nothing 
        } else {
          fl::logger->Die()<<"This option ("<<connect_nodes<<") "
            "for --connect_nodes is not supported";
        }
      }
    }
    // Now that we have a graph it is time to run the classification     
    if (vm["run_diffusion"].as<bool>()==true) {
      if (vm["symmetric_diffusion"].as<bool>()==false) {
        if (vm.count("left_labels_in")==0 &&
            vm.count("right_labels_in")==0 ) {
          fl::logger->Die()<<"When --symmetric_diffusion=0 then "
            "either --right_labels_in or --left_labels_in "
            "must be set";
        }
      } else {
        if (vm.count("left_labels_in")>0) {
          fl::logger->Die()<<"When --symmetric_diffusion=1  "
            " the diffusion in not on a bipartite graph so "
            " --left_labels must not be set";
        }
        if (vm.count("right_labels_in")==0) {
          fl::logger->Die()<<"if --run_diffusion=1 "
            "and --symmetric_diffusion=1 then "
            " --right_labels_in must be set";
        }
      }
      fl::logger->Message()<<"Ready to run diffusion"<<std::endl;
      if (vm["summary"].as<std::string>()=="none" &&
          vm["connect_nodes"].as<std::string>()=="none") {
        graph_name=vm["references_in"].as<std::string>();
        fl::logger->Message()<<"Using the --references_in="
          <<graph_name<< " as the graph"<<std::endl;
      } else {
        fl::logger->Message()<<"Using the generated graph"<<std::endl;
      }

      std::string decision_policy=vm["decision_policy"].as<std::string>();
      int32 iterations=vm["diffusion_iterations"].as<int32>();
      std::string left_normalization=vm["left_labels_norm"].as<std::string>();
      std::string right_normalization=vm["right_labels_norm"].as<std::string>();
      if (left_normalization!="none" &&
        left_normalization!="clip" &&
        left_normalization!="l1"   &&
        left_normalization!="l2") {
        fl::logger->Die()<<"The --left_labels_norm="<<left_normalization
           <<" is not valid";
      }
      if (right_normalization!="none" &&
        right_normalization!="clip" &&
        right_normalization!="l1"   &&
        right_normalization!="l2") {
        fl::logger->Die()<<"The --left_labels_norm="<<right_normalization
           <<" is not valid";
      }
      if (vm["symmetric_diffusion"].as<bool>()==false) {
        boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t>
          left_labels_table;
        boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t>
          right_labels_table;
        if (vm.count("left_labels_in")>0) {
          ws_->Attach(vm["left_labels_in"].as<std::string>(), 
              &left_labels_table);
        }
        if (vm.count("right_labels_in")>0) {
          ws_->Attach(vm["right_labels_in"].as<std::string>(), 
              &right_labels_table);
        }
        std::string right_labels_out;
        if (vm.count("right_labels_out")==0) {
          right_labels_out=ws_->GiveTempVarName();
        } else {
          right_labels_out=vm["right_labels_out"].as<std::string>();
        }
        std::string left_labels_out;
        if (vm.count("left_labels_out")==0) {
          left_labels_out=ws_->GiveTempVarName();
        } else {
          left_labels_out=vm["left_labels_out"].as<std::string>();
        }
       
       GraphDiffuser<WorkSpaceType>::DiffuseBipartite(ws_, 
            graph_name,
            decision_policy,
            iterations,
            right_labels_table,
            right_normalization,
            left_labels_table,
            left_normalization,
            right_labels_out,
            left_labels_out);
      } else {
        boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t>
          right_labels_table;
        if (vm.count("right_labels_in")>0) {
          ws_->Attach(vm["right_labels_in"].as<std::string>(), 
              &right_labels_table);
        }
        std::string right_labels_out;
        if (vm.count("right_labels_out")==0) {
          right_labels_out=ws_->GiveTempVarName();
        } else {
          right_labels_out=vm["right_labels_out"].as<std::string>();
        }
        std::string right_normalization=vm["right_labels_norm"].as<std::string>();
        GraphDiffuser<WorkSpaceType>::Diffuse(ws_,
            graph_name,
            decision_policy,
            iterations,
            right_labels_table,
            right_normalization,
            right_labels_out);
      }
    } else {
      fl::logger->Warning()<<"No diffusion selected"<<std::endl;
      if (vm.count("right_labels_out") ||
          vm.count("left_labels_out") || 
          vm.count("confidences_out")) {
        fl::logger->Die()<<"Since you are not running a diffusion you "
          "cannot set --right_labels_out --left_labels_out "
          "--confidences_out";
      }
    }
    fl::logger->Message()<<"Finished running diffusion"<<std::endl;
  }

  template<typename WorkSpaceType>
  int GraphDiffuser<boost::mpl::void_>::Run(
      WorkSpaceType *ws,
      const std::vector<std::string> &args) {

    bool found=false;
    std::string references_in;
    for(size_t i=0; i<args.size(); ++i) {
      if (fl::StringStartsWith(args[i],"--references_in=")) {
        found=true;
        std::vector<std::string> tokens=fl::SplitString(args[i], "=");
        if (tokens.size()!=2) {
          fl::logger->Die()<<"Something is wrong with the --references_in flag";
        }
        references_in=tokens[1];
        break;
      }
    }
    if (found==false) {
      Core<WorkSpaceType> core(ws, args);
      typename WorkSpaceType::DefaultTable_t t;
      core(t);
      return 1;
    }

    Core<WorkSpaceType> core(ws, args);
    fl::ws::BasedOnTableRun(ws, references_in, core);
    return 0;
}

}}

#endif
