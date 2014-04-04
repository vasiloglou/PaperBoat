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
#ifndef FL_LITE_INCLUDE_FASTLAB_WORKSPACE_BASED_ON_TABLE_RUN_H_
#define FL_LITE_INCLUDE_FASTLAB_WORKSPACE_BASED_ON_TABLE_RUN_H_

#include "boost/mpl/for_each.hpp"
#include "boost/thread.hpp"
#include "fastlib/base/logger.h"

namespace fl { namespace ws {
  namespace BasedOnTableRunWS {
    template<typename WorkSpaceType, typename FunctorType>
    class TableSelector {
      public:
        TableSelector(WorkSpaceType *ws, FunctorType &func, 
            const std::string &table_name, bool *done) :
          ws_(ws), func_(func), table_name_(table_name), done_(done){}      
        
        template<typename T>
        void operator()(T&) {
          if (*done_==true) { 
            return;
          } else {
            try {
              ws_->template TryToAttach<T>(table_name_);
              T dummy;
              ws_->schedule(boost::template bind<void>(func_, boost::ref(dummy)));
              //ws_->schedule(boost::bind(&FunctorType::template operator()<T>, func_));
              *done_=true;
            }
            catch(const fl::TypeException &e) {
             
            }
            catch(const fl::Exception &e) {
              fl::global_exception=boost::current_exception();
              ws_->CancelAllTasks();
              *done_=true;
            }  
            catch(const boost::thread_interrupted &e) {
              fl::global_exception_mutex->lock();
              fl::global_exception=boost::current_exception();
              fl::global_exception_mutex->unlock();
              *done_=true;
            }
          }
        }
      private:
        WorkSpaceType *ws_;
        FunctorType &func_;
        const std::string &table_name_;
        bool *done_;
    };  
  }
  /**
   *  @brief When you need to schedule function that depends on TableType
   *         you should use the following function
   *   ws: is the workspace that has all the data your function needs
   *   table_name: is the table name that the workspace will find its type
   *               and choose the right version
   *   func: is a function object. It must implement the 
   *            template<typename TableType>
   *            void operator()(T&);
   *            WARNING !!!! it must be T& and not T, because Table is noncopyable
   *
   */
  template<typename WorkSpaceType, typename FunctorType>
  void BasedOnTableRun(WorkSpaceType *ws, 
      const std::string &table_name,
      FunctorType &func) {
    bool done=false;
    boost::mpl::for_each<typename WorkSpaceType::DataTables_t>(
        BasedOnTableRunWS::
        TableSelector<WorkSpaceType, FunctorType>(
          ws, func, table_name, &done));
    fl::global_exception_mutex->lock();
    bool detected_exception=fl::global_exception!=0;
    fl::global_exception_mutex->unlock();
    if (detected_exception) {
      fl::logger->Die()<<"Propagating termination";
    }
  }
}}

#endif
