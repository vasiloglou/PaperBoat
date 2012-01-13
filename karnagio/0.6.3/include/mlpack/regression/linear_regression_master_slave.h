/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
Inc, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE Ismion Inc "AS IS" AND ANY
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

#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MASTER_SLAVE_H_
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MASTER_SLAVE_H_
#include "linear_regression.h"
#include <string>
#include <vector>
#include <deque>
#include "fastlib/communication/constants.h"
#include "fastlib/communication/client.h"
#include "fastlib/communication/server.h"
#include "boost/thread/mutex.hpp"
#include "boost/threadpool.hpp"


namespace fl {
namespace ml {
  template <typename TemplateArgs>
  class LinearRegressionMasterSlave;

  template<>
  class LinearRegressionMasterSlave<boost::mpl::void_> {
    public:
      /*
       * @brief use this version for the server regression
       */
      static bool ConstructBoostVariableMap(
        const std::vector<std::string> &args,
        std::vector<std::vector<std::string> > *task_arguments,
        boost::program_options::variables_map *vm);
  
      /*
       * @brief This is the Responder Class for the client
       */
       struct ResponderMaster : private boost::noncopyable {
         public:
           ResponderMaster(std::vector<std::string>  *args); 
           bool operator()(const boost::system::error_code& e,
                    fl::com::connection_ptr conn);
             
         private:
           std::vector<std::string> *args_;  
       };
      
       template<typename DataAccessType> 
       struct TaskMaster {
         public:
           TaskMaster(boost::mutex *host_port_mutex,
                boost::mutex *finished_tasks_mutex,
                std::deque<std::pair<std::string, std::string> > *host_ports,
                index_t *finished_tasks,
                index_t total_tasks,
                DataAccessType *data,
                ResponderMaster *responder); 
     
           void operator()();
  
         private:
           boost::mutex *host_port_mutex_;
           boost::mutex *finished_tasks_mutex_;
           index_t *finished_tasks_;
           index_t total_tasks_;
           std::deque<std::pair<std::string, std::string> > *host_ports_;  
           DataAccessType *data_;
           ResponderMaster *responder_;
       };
  
       /*
       * @brief This is the Responder Class for the Server
       */
       template<typename DataAccessType, typename BranchType>
       struct ResponderSlave : private boost::noncopyable {
         public:
           ResponderSlave(DataAccessType  *data); 
           bool operator()(const boost::system::error_code& e,
                    fl::com::connection_ptr conn);
             
         private:
           DataAccessType *data_;  
       };
  

      template<typename DataAccessType>
      static int MainMaster(DataAccessType *data,
          const std::vector<std::string> &args);

      template<typename DataAccessType, typename BranchType>
      static int MainSlave(DataAccessType *data,
          const std::vector<std::string> &args);

  };
  
}}

#endif

