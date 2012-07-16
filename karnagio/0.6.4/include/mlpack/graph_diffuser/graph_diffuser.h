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

#ifndef PAPERBOAT_INCLUDE_MLPACK_GRAPH_DIFFUSER_GRAPH_DIFFUSER_H_
#define PAPERBOAT_INCLUDE_MLPACK_GRAPH_DIFFUSER_GRAPH_DIFFUSER_H_
#include "fastlib/base/base.h"

namespace fl { namespace ml {
  template<typename WorkSpaceType>
  class GraphDiffuser {
    public:
      /**
       * @brief Takes an edge and converts it to one
       */
      struct UnitEdge {
        template<typename T>
        void operator()(T *val) const {
          *val=1;
        }
      };

      /**
       * @brief takes the edge of the graph and inverts it
       */
      struct DistInvEdge {
        template<typename T>
        void operator()(T *val) const {
          FL_SCOPED_LOG(DistInvEdge);
          if (*val<1e-10) {
            fl::logger->Warning()<<"Infinite Edge value"<<std::endl;
          }
          *val=1 / *val;
        }
      };

      /**
       * @brief Takes the edge value and makes it gaussian
       */
      struct GaussEdge {
        GaussEdge() : h_(1.0) {}
        template<typename T>
        void operator()(T *val) const {
          *val=exp(-(*val)/h_);
        }
        void set_h(double h) {
          h_=h;
        }
        private:
          double h_;
      };

      /**
       * @brief Makes a graph bases on the nearest neighbor result
       * 
       */
      template<typename EdgeFunctorType, typename GraphTableType>
      static void MakeGraph(WorkSpaceType *ws, // WorkSpace that contains 
                                        // all the tables 
        const std::string &indices_name, // the table name of the
                                         // neighbor indices 
        const std::string &weights_name, // the table name of the
                                         // distances of neighbors
        const EdgeFunctorType *edge_functor, // function object that post
                                             // processes the edge values
                                             // You can use any of the one
                                             // defined above UnitEdge, DistInv
        const std::string &normalization, // Normalization method for 
                                          // the outgoing edge weights 
                                          // for every node 
        bool make_symmetric, // switch for making the graph symmetric
        const std::string &graph_name // the graph table name 
        );

      template<typename GraphTableType>
      static void MakeGraph(WorkSpaceType *ws, // WorkSpace that contains 
                                        // all the tables 
        const std::string &indices_name, // the table name of the
                                         // neighbor indices 
        const std::string &weights_name, // the table name of the
                                         // distances of neighbors
        const std::string &edge_option, // A string for the setting
                                        /// the function object
        const std::string &normalization, // Normalization method for 
                                          // the outgoing edge weights 
                                          // for every node 
        bool make_symmetric, // switch for making the graph symmetric
        const std::string &graph_name // the graph table name 
        );
      
      /**
       * @brief Diffusion policies
       * The following classes declare diffusion policies
       */
       class DotProdPolicy {
         public:
           void Init(double val);
           void Reset();
           template<typename PointType1, typename PointType2>
           void PointEval(PointType1 &p1, PointType2 &p2);
           template<typename T1, typename T2>
           void Update(T1 val1, T2 val2);
           double Result();
         private:
           double result_; 
       };

      /**
       * @brief This function diffuses labels on a graph.
       * After several iterations nodes settle to a specific value 
       */
      template<typename DecisionPolicyType>
      struct DiffuseStruct {
        public:
          DiffuseStruct(WorkSpaceType *ws, // workspace where the graph resides
            const std::string &graph_name,// name of the graph table
            DecisionPolicyType &decision_policy, // a decision policy for
                                                 // assigning a value to 
                                                 // a node based on the values
                                                 // of its neighbors
            int32 iterations,  // number of iterations to run the diffusion
            boost::shared_ptr<
              typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_right_table, 
                // predefined labels 
                // of the right nodes (Since this is a normal graph left and right are the same)
            const int right_normalization,
            const std::string &right_labels_out // table name for the right labels 
            );

          template<typename GraphTableType>
          void operator()(GraphTableType&);

        private:
          WorkSpaceType *ws;
          const std::string &graph_name;
          DecisionPolicyType &decision_policy;
          int32 iterations;  
          boost::shared_ptr<
            typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_right_table;
          const int right_normalization;
          const std::string &right_labels_out;
        
      };
      
      /**
       * @brief Same as before but this one is table agnostic.
       *  It also pre-encodes different decision policies
       */
      static void Diffuse(WorkSpaceType *ws, // workspace where the graph resides
          const std::string &graph_name,// name of the graph table
          const std::string &decision_policy, // a decision policy for
                                              // assigning a value to 
                                              // a node based on the values
                                              // of its neighbors
          int32 iterations,  // number of iterations to run the diffusion
          boost::shared_ptr<
            typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_right_table, 
              // predefined labels 
              // of the right nodes (Since this is a normal graph left and right are the same)
          const std::string &right_normalization,
          const std::string &right_labels_out // table name for the right labels 
          );

      /**
       * @brief This function diffuses labels on a bipartite graph
       */
      template<typename DecisionPolicyType>
      struct DiffuseBipartiteStruct {
        public:
           DiffuseBipartiteStruct(WorkSpaceType *ws, // workspace where the graph resides
             const std::string &graph_name, // name of the graph table
             DecisionPolicyType &decision_policy, // a decision policy for
                                                  // assigning a value to 
                                                  // a node based on the values
                                                  // of its neighbors
             int32 iterations, // number of iterations to run the diffusion
             boost::shared_ptr<
                typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_right_table, 
                  // predefined labels 
                  // on the right nodes
             const int right_normalization,
             boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_left_table, 
                // predefined labels 
                // on the left nodes
             const int left_normalization,
             const std::string &right_labels_out, // table name for the right nodes
             const std::string &left_labels_out); // table name for the left nodes

          template<typename GraphTableType>
          void operator()(GraphTableType&);

        private:
          WorkSpaceType *ws;
          const std::string &graph_name; 
          DecisionPolicyType &decision_policy;
          int32 iterations;
          boost::shared_ptr<
            typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_right_table;
          const int right_normalization;
          boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_left_table; 
          const int left_normalization;
          const std::string &right_labels_out;
          const std::string &left_labels_out;
      };

      /**
       * @brief It is the same as before with the only difference 
       *  that it is graph table agnostic and also pre-encodes
       *  decision policies
       */
      static void DiffuseBipartite(WorkSpaceType *ws, // workspace where the graph resides
          const std::string &graph_name, // name of the graph table
          const std::string &decision_policy, // a decision policy for
                                               // assigning a value to 
                                               // a node based on the values
                                               // of its neighbors
          int32 iterations, // number of iterations to run the diffusion
          boost::shared_ptr<
              typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_right_table, 
              // predefined labels 
              // on the right nodes
          const std::string &right_normalization,
          boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t> predefined_left_table, 
              // predefined labels 
              // on the left nodes
          const std::string &left_normalization,
          const std::string &right_labels_out, // table name for the right nodes
          const std::string &left_labels_out); // table name for the left nodes

      /**
       * @brief Auxiliary function, normalizes the edges of a graph so that 
       * is L1 or L2 normalized
       */
      static void Normalize(std::vector<std::pair<index_t, double> >::iterator begin,
          std::vector<std::pair<index_t, double> >::iterator end,
          int norm); 
     
      template<typename PointType>
      static void Normalize(
          PointType *point,
          int norm);

  };

  template<>
  class GraphDiffuser<boost::mpl::void_> {
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
