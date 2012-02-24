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

#ifndef FL_LITE_INCLUDE_FASTLAB_WORKSPACE_TASK_H_
#define FL_LITE_INCLUDE_FASTLAB_WORKSPACE_TASK_H_
#include <vector>
#include <string>
#include "boost/thread.hpp"
#include "fastlib/base/logger.h"

namespace fl{ namespace ws {
  template<typename DataAccessType, 
           int (* ExecutorPtr)(DataAccessType *, 
                                const std::vector<std::string> &)>
  class Task {
    public:
      typedef int (* ExecutorPtr_t) (DataAccessType *, const std::vector<std::string> &);      
      static ExecutorPtr_t function_ptr;
      Task(DataAccessType *data, const std::vector<std::string> &args) :
        data_(data), args_(args) {
        }
      Task(const Task& other) : data_(other.data_), args_(other.args_) {
      }
      Task() {
      }
      ~Task() {
      }
      void operator()() {
        try{
          ExecutorPtr(data_, args_);
        }
        catch(...) {
          boost::mutex::scoped_lock lock(*global_exception_mutex);
          fl::global_exception=boost::current_exception();
        }
      }
      const std::vector<std::string> &args() const {
        return args_;
      }
      std::vector<std::string> &args() {
        return args_;
      }

    private:
      DataAccessType *data_;
      std::vector<std::string> args_;
  };
  
  template<typename DataAccessType, 
           int (* ExecutorPtr)(DataAccessType *, 
           const std::vector<std::string> &)>
  typename Task<DataAccessType, 
                ExecutorPtr>::
                  ExecutorPtr_t 
                  Task<
                    DataAccessType, 
                    ExecutorPtr
                  >::function_ptr=ExecutorPtr;
}}
#endif
