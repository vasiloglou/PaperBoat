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

#ifndef PAPERBOAT_FASTLIB_INCLUDE_GRAPHVIS_DEFS_H_
#define PAPERBOAT_FASTLIB_INCLUDE_GRAPHVIS_DEFS_H_
#include <sys/time.h>
#include <algorithm>
#include <unistd.h>
#include "graphvis.h"
#include "mlpack/approximate_meanshift/approximate_meanshift.h"
#include "mlpack/kde/kde.h"
#include "mlpack/graph_diffuser/graph_diffuser.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "fastlib/math/fl_math.h"
extern "C" {
#include "ubigraph/UbigraphAPI.h"
}


template<typename WorkSpaceType>
fl::ml::GraphVis<WorkSpaceType>::GraphVis() {
  graph_table_=NULL;
  num_of_points_on_the_screen_=0;
}

template<typename WorkSpaceType>
template<bool IS_VISIBLE>
void fl::ml::GraphVis<WorkSpaceType>::MergeNodes(
    int32 node_to_remove,
    const std::map<int32, double> &nodes_link_to_node, // the list of
                      // all the points that their node point to the
                      // node that will be eliminated
    const std::map<index_t, int32> &point2vertex,
    std::map<int32, std::map<int32, double> > *nodes_point_to_nodes) {
  // first pick the replacement node
  typename std::map<int32, double>::const_iterator it=nodes_link_to_node.begin();
  if (nodes_link_to_node.size()>1) {
    std::advance(it, 
      fl::math::Random(int32(0), int32(nodes_link_to_node.size()-1)));
  }
  int32 replacement_node=it->first;
  double replacement_weight=it->second;

   // now make all points point to the replacement node.
  for(std::map<int32, double>::const_iterator it1=nodes_link_to_node.begin();
      it1!=nodes_link_to_node.end(); ++it1) {
    (*nodes_point_to_nodes)[it1->first].erase(node_to_remove);
    if (it1->first!=replacement_node
        && (*nodes_point_to_nodes)[replacement_node].count(it1->first)==0) {
      double weight=it1->second*replacement_weight;
      int32 edge_id=MakeEdge(it1->first, replacement_node, weight,IS_VISIBLE);
      if (edge_id!=-1) {
        (*nodes_point_to_nodes)[replacement_node][it1->first]=weight; 
        (*nodes_point_to_nodes)[it1->first][replacement_node]=weight; 
      }

      if ((*nodes_point_to_nodes)[it1->first].size()==0) {
        ubigraph_remove_vertex(it1->first);
        nodes_point_to_nodes->erase(it1->first);
      }
    } 
  }  
}

template<typename WorkSpaceType>
int32 fl::ml::GraphVis<WorkSpaceType>::MakeEdge(
    int32 vertex_id1, int32 vertex_id2, 
    double weight, bool visible) {
  if (weight<1e-4) {
    return -1;
  }
  int32 edge_id=ubigraph_new_edge(vertex_id1, vertex_id2);
  if (visible==false) {
    ubigraph_set_edge_attribute(edge_id, "visible", "false");
  }
  ubigraph_set_edge_attribute(edge_id, "strength", 
  boost::lexical_cast<std::string>(weight).c_str());
  return edge_id;
}

template<typename WorkSpaceType>
int32 fl::ml::GraphVis<WorkSpaceType>::MakeVertex(const std::string &shape, 
          const std::string &color, 
          int32 size) {
  int32 vertex_id=ubigraph_new_vertex();
  ubigraph_set_vertex_attribute(vertex_id, "color", color.c_str());
  ubigraph_set_vertex_attribute(vertex_id, "shape", shape.c_str());
  ubigraph_set_vertex_attribute(vertex_id, "size",
      boost::lexical_cast<std::string>(size).c_str());
  return vertex_id;
}

/**
 * @brief It adds a point to the graph on screen. If we have reached
 *        the maximum number of points on the screen then we drop the
 *        oldest point and we assign its links to its most popular neighbor
 */
template<typename WorkSpaceType>
template<bool IS_VISIBLE>
int32 fl::ml::GraphVis<WorkSpaceType>::AddPoint(
    const index_t point_id, 
    const bool lock_point,
    const int32 max_points_on_the_screen,
    std::map<index_t, int32> *point2vertex, 
    std::map<int32, std::map<int32, double> > *nodes_point_to_nodes,
    std::map<int64, index_t> *time2point_id,
    int32 *points_on_the_screen) {

  typedef typename NeighborTable_t::Point_t Point_t;
  if (*points_on_the_screen>=max_points_on_the_screen) {

    int counter=0;
    std::map<int64, index_t>::iterator it;
    // remove the 10 oldest points
    for(it=time2point_id->begin();
        it!=time2point_id->end(); ++it) {
      if (point2vertex->count(it->second)==0) {
        continue;
      }
      // now take all the vertices that point to this vertex
      // and then send them to a different vertex
      int32 node_to_remove=(*point2vertex)[it->second];
      this->template MergeNodes<IS_VISIBLE>(node_to_remove,
                       (*nodes_point_to_nodes)[node_to_remove], 
                       *point2vertex,
                       nodes_point_to_nodes);
      nodes_point_to_nodes->erase(node_to_remove);
      ubigraph_remove_vertex(node_to_remove);
      point2vertex->erase(it->second);
      counter++;
      (*points_on_the_screen)--;
      if (counter>=10) {
        break;
      }
    }
    if (it!=time2point_id->end()) {
      ++it;
    }
    time2point_id->erase(time2point_id->begin(), it);
  }
  timeval time_struct;
  int32 vertex_id1;
  if (point2vertex->count(point_id)==0) {
    if (lock_point==false) {
      gettimeofday(&time_struct, NULL);
      (*time2point_id)[time_struct.tv_usec]=point_id;
    } 
    vertex_id1=MakeVertex("sphere", "#ff0000", 2);
    (*point2vertex)[point_id]=vertex_id1;      
    *points_on_the_screen+=1;
  } else {
    vertex_id1=(*point2vertex)[point_id];
  }
  return vertex_id1;
}


template<typename WorkSpaceType>
template<bool IS_VISIBLE>
void fl::ml::GraphVis<WorkSpaceType>::VisualizeRandomSequential() {
  std::map<index_t, int32> point2vertex; 
  std::map<int64, index_t> time2point;
  std::map<int32, std::map<int32, double> > nodes_point_to_nodes;
  typedef typename NeighborTable_t::Point_t Point_t;
  Point_t point;
  int32 points_on_the_screen=0;
  std::vector<index_t> point_order(graph_table_->n_entries());
  // We create a vector with the order of plotting points
  for(size_t i=0; i<point_order.size(); ++i) {
    point_order[i]=i;
  }

  std::random_shuffle(point_order.begin(), point_order.end());

  while(true) {
    for(index_t i=0; i<graph_table_->n_entries(); ++i) {
      std::cout<<"visualizing point="<<i<<std::endl;
      graph_table_->get(point_order[i], &point);
      int32 vertex_id1=AddPoint<IS_VISIBLE>(point_order[i], 
          false,
          max_points_on_the_screen_,
          &point2vertex, 
          &nodes_point_to_nodes,
          &time2point, 
          &points_on_the_screen);
      for(typename Point_t::iterator it=point.begin();
          it!=point.end(); ++it) {
        int32 vertex_id2=AddPoint<IS_VISIBLE>(it.attribute(), 
            false,
            max_points_on_the_screen_,
          &point2vertex, 
          &nodes_point_to_nodes,
          &time2point, 
          &points_on_the_screen);
        // it is possible that so far the point_order[i] might have been evicted
        if (point2vertex.count(point_order[i])) {
          ubigraph_new_edge(vertex_id1, vertex_id2);
          nodes_point_to_nodes[vertex_id1][vertex_id2]=1.0;
          nodes_point_to_nodes[vertex_id2][vertex_id1]=1.0;
        }
      }
    }
    // Reshuffle the plot order
    std::random_shuffle(point_order.begin(), point_order.end());
  }
}

template<typename WorkSpaceType>
template<bool IS_VISIBLE>
void fl::ml::GraphVis<WorkSpaceType>::VisualizeApproxMeanShift(
    WorkSpaceType *ws,
    double edge_sparsity,
    const std::vector<std::string> &args) {
  
  std::map<index_t, int32> point2vertex; 
  std::map<int64, index_t> time2point;
  std::map<int32, std::map<int32, double> > nodes_point_to_nodes;

  typedef typename NeighborTable_t::Point_t Point_t;
  Point_t point;
  int32 points_on_the_screen=0;

 std::map<std::string, std::string> arg_map=fl::ws::GetArgumentPairs(args); 
 std::vector<std::string> ams_args=fl::ws::MakeArgsFromPrefix(args, "ams");
 std::vector<std::string> kde_args=fl::ws::MakeArgsFromPrefix(ams_args, "kde");
 std::map<std::string, std::string> kde_map=fl::ws::GetArgumentPairs(kde_args);
 kde_args.push_back("--references_in="+arg_map["--references_in"]);
 if (kde_map.count("--references_in")) {
   fl::logger->Die()<<"You are not allowed to set --references_in for ams:kde:";
 }
 std::string densities_name=ws->GiveTempVarName();
 kde_args.push_back("--densities_out="+densities_name);
 fl::ml::Kde<boost::mpl::void_>::Run(ws, kde_args);
 double bandwidth=boost::lexical_cast<double>(
     kde_map["--bandwidth"]);
 double square_bandwidth=bandwidth*bandwidth;
 std::map<std::string, std::string> ams_map=fl::ws::GetArgumentPairs(
      ams_args);
 std::string cluster_stat_name;
  if (ams_map.count("--cluster_statistics_out")==0) {
    cluster_stat_name=ws->GiveTempVarName();
  } else {
    cluster_stat_name=ams_map["--cluster_statistics_out"];
  }
  ams_args.push_back("--cluster_statistics_out="+cluster_stat_name);
  std::string clusters_name;
  if (ams_map.count("--clusters_out")==0) {
    clusters_name=ws->GiveTempVarName();
  } else {
    clusters_name=ams_map["--clusters_out"];
  }
  ams_args.push_back("--clusters_out="+clusters_name);
  std::string memberships_name;
  if (ams_map.count("--memberships_out")==0) {
    memberships_name=ws->GiveTempVarName();
  } else {
    memberships_name=ams_map["--memberships_out"];
  }
  ams_args.push_back("--memberships_out="+memberships_name);
  ams_args.push_back("--is_references_in_a_graph=1");
  ams_args.push_back("--densities_in="+densities_name);
  for(std::vector<std::string>::iterator it=ams_args.begin();
      it!=ams_args.end(); ++it) {
    std::cout<<*it<<",";
  }
  std::cout<<std::endl;
  fl::ml::ApproximateMeanShift<boost::mpl::void_>::Run(ws, ams_args);
  boost::shared_ptr<typename WorkSpaceType::IntegerTable_t> memberships_table;
  ws->Attach(memberships_name, &memberships_table);
  boost::shared_ptr<typename WorkSpaceType::IntegerTable_t> clusters_table;
  ws->Attach(clusters_name, &clusters_table);
  // Plot the centroids
  for(index_t i=0; i<clusters_table->n_entries(); ++i) {
    AddPoint<IS_VISIBLE>(
        clusters_table->get(i, index_t(0)),
        true,
        max_points_on_the_screen_,
        &point2vertex,
        &nodes_point_to_nodes,
        &time2point,
        &points_on_the_screen);
  }
  // Experimental
  // put a link between all centers
  for(index_t i=0; i<clusters_table->n_entries(); ++i) {
    for(index_t j=0; j<i; ++j) {
      int32 vertex_id1=point2vertex[clusters_table->get(i, index_t(0))];
      if (i!=j) {
        int32 vertex_id2=point2vertex[clusters_table->get(j, index_t(0))];
        MakeEdge(vertex_id1, vertex_id2, 0.001, false);
      }
    }
  }
  int32 n_clusters=clusters_table->n_entries();
  int32 active_cluster=0;
  int32 max_count=10;
  while(true) {
    index_t current_index=clusters_table->get(active_cluster, index_t(0));
    for(int32 counter=0; counter<max_count; ++counter) {
      int32 vertex_id1=AddPoint<IS_VISIBLE>(current_index, 
          false,
          max_points_on_the_screen_,
          &point2vertex, 
          &nodes_point_to_nodes,
          &time2point, 
          &points_on_the_screen);
      int32 edges_so_far=nodes_point_to_nodes[vertex_id1].size();
      graph_table_->get(current_index, &point);
      int32 allowed_edges=edge_sparsity*point.nnz();
      int32 edges_to_go=allowed_edges-edges_so_far;
      std::vector<index_t> indices;
      int32 counter1=0;
      for(typename NeighborTable_t::Point_t::iterator it=point.begin();
          it!=point.end(); ++it) {
        indices.push_back(it.attribute());
        if (counter1>edges_to_go) {
          continue;
        }
        if (fl::math::Random(0, 1)==0) {
          continue;
        }
        int32 vertex_id2=AddPoint<IS_VISIBLE>(it.attribute(), 
            false,
            max_points_on_the_screen_,
            &point2vertex, 
            &nodes_point_to_nodes,
            &time2point, 
            &points_on_the_screen);
        if (nodes_point_to_nodes[vertex_id1].count(vertex_id2)>0) {
          continue;
        } else {
          counter1++;
        }
        // You have to do another check in case there was an eviction
        if (point2vertex.count(current_index)==0) {
          vertex_id1=AddPoint<IS_VISIBLE>(current_index, 
            false,
            max_points_on_the_screen_,
            &point2vertex, 
            &nodes_point_to_nodes,
            &time2point, 
            &points_on_the_screen);
        }
        if (point2vertex.count(it.attribute())==0) {
          vertex_id2=AddPoint<IS_VISIBLE>(it.attribute(), 
            false,
            max_points_on_the_screen_,
            &point2vertex, 
            &nodes_point_to_nodes,
            &time2point, 
            &points_on_the_screen);
        }
        double weight=exp(-it.value()/square_bandwidth)/bandwidth;
        MakeEdge(vertex_id1, vertex_id2, weight,
            IS_VISIBLE);
        nodes_point_to_nodes[vertex_id1][vertex_id2]=weight;
        nodes_point_to_nodes[vertex_id2][vertex_id1]=weight;
      }
      if (indices.size()>0) {
        current_index=indices[fl::math::Random(size_t(0), size_t(indices.size()-1))];
      }
    }
    active_cluster=(active_cluster+1) % n_clusters;
  }
}

template<typename WorkSpaceType>
void fl::ml::GraphVis<WorkSpaceType>::Clear() {
  ubigraph_clear();
  num_of_points_on_the_screen_=0;
}



template<typename WorkSpaceType>
void fl::ml::GraphVis<WorkSpaceType>::set_graph_table(NeighborTable_t *graph_table) {
  graph_table_=graph_table;
} 


template<typename WorkSpaceType>
void fl::ml::GraphVis<WorkSpaceType>::set_point_vcolors(
    typename WorkSpaceType::IntegerTable_t   &colors) {
  vcolors_.resize(colors.n_entries());
  typename WorkSpaceType::IntegerTable_t::Point_t point;
  for(size_t i=0; i<colors.n_entries(); ++i) {
    colors.get(i, &point);
    char tbuf[20];
    sprintf(tbuf, "#%02x%02x%02x", 
        static_cast<unsigned int>(point[0]), 
        static_cast<unsigned int>(point[1]), 
        static_cast<unsigned int>(point[2]));
    vcolors_[i]=tbuf;
  }  
}


template<typename WorkSpaceType>
void fl::ml::GraphVis<WorkSpaceType>::set_max_points_on_screen(
    index_t max_points_on_the_screen) {
  max_points_on_the_screen_=max_points_on_the_screen;
}

template<typename WorkSpaceType>
void fl::ml::GraphVis<WorkSpaceType>::set_ubigraph_url(const std::string &url) {
  ubigraph_url_=url;
  if (fl::StringStartsWith(ubigraph_url_, "http://") 
      || fl::StringStartsWith(ubigraph_url_, "https://")) {
  
  } else {
    ubigraph_url_="http://"+ubigraph_url_;
  }
  if (fl::StringEndsWith(ubigraph_url_, "/RPC2")==false) {
    ubigraph_url_.append("/RPC2");
  }
  init_xmlrpc(ubigraph_url_.c_str());
}

template<typename WorkSpaceType>
template<typename TableType>
void fl::ml::GraphVis<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
    TableType&) {
  FL_SCOPED_LOG(GraphVis);
  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>(),
    "the reference data "
  )(
    "graph_in",
    boost::program_options::value<std::string>(),
    "data in graph format"
  )(
    "method",
    boost::program_options::value<std::string>()->default_value("ams"),
    "method to use for visualization:\n"
    "    rseq: random sequential\n"
    "    ams: approximate meanshift"  
  )(
    "node_colors_in",
    boost::program_options::value<std::string>(),
    "a table that contains 3 integer values between 0-255. The values "
    "correspond to RGB color values of the graph nodes"  
  )(
    "max_nodes_on_screen",
    boost::program_options::value<index_t>()->default_value(1000),
    "maximum number of graph nodes to be drawn on the screen" 
  )(
    "edge_sparsity",
    boost::program_options::value<double>()->default_value(0.5),
    "each node has n edges, but only a percentage of the edges can be "
    "drawn. --edge sparsity must be between 0 and 1 "
  )(
    "visible_edge",
    boost::program_options::value<bool>()->default_value(true),
    "if you set --visible_edge=1 edges will be shown on the visualization"
  )(
    "vertex_shape",
    boost::program_options::value<std::string>()->default_value("sphere"),
    "the shape of the vertex it can be:\n"
    "  sphere, cube, cone, torus, dodecachedron"
  )(
    "vertex_size",
    boost::program_options::value<int>()->default_value(2),
    "the size of the vertex symbol"
  )(
    "ubigraph_url",
    boost::program_options::value<std::string>()->default_value("http://localhost:20738/RPC2"),
    "the url of the ubigraph server"
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
  fl::ws::RequiredArgs(vm, "references_in");
  fl::ws::ImpossibleArgs(vm, "references_in", "graph_in");
  boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t> graph_table;
  std::string graph_name;
  if (vm.count("references_in")) {
     std::vector<std::string> graphd_args=fl::ws::MakeArgsFromPrefix(args_, "graphd");
     std::map<std::string, std::string> graphd_map=fl::ws::GetArgumentPairs(graphd_args);
     if (graphd_map.count("graph_out")) {
       graph_name=graphd_map["graph_out"];
     } else {
       graph_name=ws_->GiveTempVarName();
       graphd_args.push_back("--graph_out="+graph_name);
     }
     if (graphd_map.count("references_in")) {
       fl::logger->Die()<<"You are not allowed to set --graphd:references_in"
         " argument";
     }
     graphd_args.push_back("--references_in="+vm["references_in"].as<std::string>());
     fl::ml::GraphDiffuser<boost::mpl::void_>::Run(ws_, graphd_args);  
     ws_->Attach(graph_name, &graph_table);
  } else {
    if (vm.count("graph_in")==0) {
      fl::logger->Die()<<"You need to specify either references_in or --graph_in";     
    }
  }
  ws_->Attach(graph_name, &graph_table);
  GraphVis<WorkSpaceType> engine;  
  engine.set_graph_table(graph_table.get());
  if (vm.count("node_colors")) {
    boost::shared_ptr<typename WorkSpaceType::IntegerTable_t> colors_table;
    ws_->Attach(vm["node_colors"].as<std::string>(), &colors_table);
    // check if it is a correct matrix
    {
      if (colors_table->n_entries()!=graph_table->n_entries()) {
        fl::logger->Die()<<"Color table must have the same number of entries with the "
          "graph table";
      }
      if (colors_table->n_attributes()!=3) {
        fl::logger->Die()<<"Color table must have 3 attributes";
      }
      fl::logger->Message()<<"Checking color integrity"<<std::endl;
      for(index_t i=0; i<colors_table->n_entries(); ++i) {
        for(index_t j=0;j<3; ++j) {
          int color=colors_table->get(i, j);
          if (color<0 || color>255) {
            fl::logger->Die()<<"Entry ("<<i<<") in color table ("
             <<colors_table->filename()<<") is ("<<color<<") and it is"
             <<" outside of the acceptable bounds [0, 255]";
          }
        }
      }
    }
    engine.set_point_vcolors(*colors_table);
  }
  engine.set_max_points_on_screen(vm["max_nodes_on_screen"].as<index_t>());
  engine.set_ubigraph_url(vm["ubigraph_url"].as<std::string>());


  if (vm["method"].as<std::string>()=="rseq") {
    if (vm["visible_edge"].as<bool>()==true) {
      engine.template VisualizeRandomSequential<true>();
    } else {
      engine.template VisualizeRandomSequential<false>();
    }
  }
  if (vm["method"].as<std::string>()=="ams") {
    std::vector<std::string> ams_args=args_;
    ams_args.push_back("--ams:references_in="+graph_name);
    double edge_sparsity=vm["edge_sparsity"].as<double>();
    if (edge_sparsity<0 || edge_sparsity>1) {
      fl::logger->Die()<<"--edge_sparsity must be between 0 and 1";
    }
    if (vm["visible_edge"].as<bool>()==true) {
      engine.template VisualizeApproxMeanShift<true>(ws_,
            edge_sparsity,
            ams_args);
    } else {
      engine.template VisualizeApproxMeanShift<false>(ws_,
            edge_sparsity,
            ams_args);
    }
  }
}


template<typename WorkSpaceType>
int fl::ml::GraphVis<boost::mpl::void_>::Run(
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

  template<typename WorkSpaceType>
  fl::ml::GraphVis<boost::mpl::void_>::Core<WorkSpaceType>::Core(
     WorkSpaceType *ws, const std::vector<std::string> &args) :
   ws_(ws), args_(args)  {}


#endif
