/*
Copyright © 2010, Ismion Inc.
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

#include "fastlib/workspace/arguments.h"

 std::vector<std::string> fl::ws::MakeArgsFromPrefix(
     const std::vector<std::string> &vec, const std::string &prefix) {
   std::vector<std::string> result;
   std::string tok("--");
   tok.append(prefix);
   if (prefix!="") {
     tok.append(":");
   }

   for(unsigned int i=0; i<vec.size(); ++i) {
     std::vector<std::string> option_part;
     boost::algorithm::split(option_part, vec[i],
            boost::algorithm::is_any_of("="));
     std::vector<std::string> tokens;
     std::string::size_type ind=option_part[0].find(tok);
     std::string::size_type ind1=option_part[0].find(":");
    
     if ((ind!=std::string::npos && prefix!="") || (ind1==std::string::npos && prefix=="")) {
       std::string new_tok("--");
       new_tok.append(option_part[0].substr(
             tok.size(), std::string::npos));
       new_tok.append("=");
       new_tok.append(option_part[1]);
       result.push_back(new_tok);
     }
   }
   return result;
 }

