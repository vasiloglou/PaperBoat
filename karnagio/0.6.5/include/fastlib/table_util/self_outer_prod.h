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

#ifndef FASTLIB_TABLE_SELF_OUTER_PROD_H_
#define FASTLIB_TABLE_SELF_OUTER_PROD_H_
#include <vector>
#include <map>
#include <string>
#include "boost/mpl/void.hpp"
#include "boost/program_options.hpp"
#include "boost/shared_ptr.hpp"
#include "fastlib/base/base.h"

/**
 * @brief This file contains utilities for manipulating fl-lite files
 *  spliting them, picking cross validating samples etc.
 *
 */
namespace fl { namespace table {
 template<typename>
 class SelfOuterProd;

 template<>
 class SelfOuterProd<boost::mpl::void_> {
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
