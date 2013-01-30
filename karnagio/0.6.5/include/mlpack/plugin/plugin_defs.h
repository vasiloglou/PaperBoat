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

#ifndef PAPERBOAT_INCLUDE_MLPACK_PLUGIN_PLUGIN_DEFS_H_
#define PAPERBOAT_INCLUDE_MLPACK_PLUGIN_PLUGIN_DEFS_H_
#include <dlfcn.h>
#include "fastlib/base/logger.h"
#include "plugin.h"

namespace fl { namespace ml {

  template<typename WorkSpaceType>
  int Plugin<boost::mpl::void_>::Run(WorkSpaceType *ws,
          const std::vector<std::string> &args) {
    if (args.size()<2) {
      fl::logger->Die()<<"When calling plugin the syntax "
         "is plugin library method --arg1=<> --arg2=<>";

    }
    std::string library_name(args[0]);
    std::string function_name(args[1]);
    void *handle=dlopen(library_name.c_str(), RTLD_LAZY);
    if (!handle) {
      fl::logger->Die()<<
        "Cannot open library: " 
        << dlerror();
    }
    // reset errors
    dlerror();
    typedef void (*FunctionType)(WorkSpaceType *, 
        std::vector<std::string>&);
    FunctionType function=(FunctionType)dlsym(handle, function_name.c_str());
    const char *dlsym_error=dlerror();
    if (dlsym_error) {
      fl::logger->Die()<<"Cannot load symbol "<<function_name<<" "
        << dlsym_error;
    }
    std::vector<std::string> args1(args.begin()+2, args.end());
    function(ws, args1);
    dlclose(handle);
    return 0; 
  }

}}
#endif
