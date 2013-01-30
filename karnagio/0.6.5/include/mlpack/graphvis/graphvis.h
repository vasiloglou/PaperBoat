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

#ifndef PAPERBOAT_FASTLIB_INCLUDE_GRAPHVIS_H_
#define PAPERBOAT_FASTLIB_INCLUDE_GRAPHVIS_H_
#include <vector>
#include <map>
#include <string>
#include "boost/tuple/tuple.hpp"
#include "fastlib/base/base.h"


namespace fl { namespace ml {
  template<typename WorkSpaceType>
  class GraphVis {
    public:
      typedef typename WorkSpaceType::DefaultSparseDoubleTable_t NeighborTable_t;
      GraphVis();
      template<bool IS_VISIBLE>
      int32 AddPoint(
          const index_t point_id, 
          const bool lock_point,
          const int32 max_points_on_the_screen,
          std::map<index_t, int32> *point2vertex, 
          std::map<int32, std::map<int32, double> > *nodes_point_to_nodes,
          std::map<int64, index_t> *time2point_id,
          int32 *points_on_the_screen);
      template<bool IS_VISIBLE>
      void MergeNodes(
          int32 node_to_remove,
          const std::map<int32, double> &points_link_to_point, // the list of
                      // all the points that their node point to the
                      // node that will be eliminated
          const std::map<index_t, int32> &point2vertex,
          std::map<int32, std::map<int32, double> > *nodes_point_to_nodes);
      int32 MakeEdge(int32 vertex_id1, int32 vertex_id2, 
          double weight, bool visible);
      int32 MakeVertex(const std::string &shape, 
          const std::string &color, 
          int32 size);
      template<bool IS_VISIBLE>
      void VisualizeRandomSequential();
      template<bool IS_VISIBLE>
      void VisualizeApproxMeanShift(
          WorkSpaceType* ws,
          double edge_sparsity,
          const std::vector<std::string> &args);
      void Clear();
      void set_graph_table(NeighborTable_t *graph_table);
      void set_point_vcolors(typename WorkSpaceType::IntegerTable_t 
          &colors);
      void set_max_points_on_screen(index_t num_of_points_on_screen);
      void set_ubigraph_url(const std::string &url);

    private:
      NeighborTable_t *graph_table_;
      std::vector<std::string> vcolors_;
      std::map<index_t, std::string> mcolors_;
      index_t num_of_points_on_the_screen_;
      index_t max_points_on_the_screen_;
      std::string ubigraph_url_;
  };

  template<>
  class GraphVis<boost::mpl::void_> {
    public:
      template<typename WorkSpaceType>
      struct Core {
        public:
          Core(WorkSpaceType *ws, const std::vector<std::string> &args);
          template<typename TableType>
          void operator()(TableType&);
        private:
          WorkSpaceType *ws_;
          std::vector<std::string> args_;
      };
    
      template<typename WorkSpaceType>
      static int Run(WorkSpaceType *data,
          const std::vector<std::string> &args);
 };


}}

#endif
