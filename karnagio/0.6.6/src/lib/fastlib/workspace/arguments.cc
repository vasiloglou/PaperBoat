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
#include "boost/algorithm/string/replace.hpp"
#include "fastlib/workspace/arguments.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/base/basic_types.h"
#include "fastlib/base/logger.h"

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
    
     if ((ind!=std::string::npos && prefix!="") || ((ind1==std::string::npos || ind1==2) && prefix=="")) {
       std::string new_tok("--");
       new_tok.append(option_part[0].substr(
             tok.size(), std::string::npos));
       if (option_part.size()==2) {
         new_tok.append("=");
         new_tok.append(option_part[1]);
       }
       result.push_back(new_tok);
     }
   }
   return result;
 }

 std::vector<std::string> fl::ws::RemoveArgsWithPrefix(
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
    
     if (!((ind!=std::string::npos && prefix!="") || ((ind1==std::string::npos || ind1==2) && prefix==""))) {
       result.push_back(vec[i]);
     }
   }
   return result;
 }

 std::map<std::string, std::string> fl::ws::GetArgumentPairs(
     const std::vector<std::string> &args) {
   std::map<std::string, std::string> argmap;
   for(std::vector<std::string>::const_iterator it=args.begin();
       it!=args.end(); ++it) {
     std::vector<std::string> tokens;
     tokens=fl::SplitString(*it, "=");
     if (fl::StringStartsWith(*it, "--")==false) {
       fl::logger->Die()<<"Argument "
         <<*it<<" does not start with --";
     }
     if (tokens.size()!=2) {
       fl::logger->Die()<<"Argument "<<*it<<" is wrong";
     }
     if (argmap.count(tokens[0])) {
       fl::logger->Die()<<"Argument "<<tokens[0]<<" has been defined "
         "multiple times";
     }
     argmap[tokens[0]]=tokens[1];
   }
   return argmap;
 }
 /**
  * @brief Defines argument that is required
  *        
  */
 void fl::ws::RequiredArgs(const boost::program_options::variables_map &vm,
     const std::string &flag // flag to be checked against 
     ) {
   if (vm.count(flag)==0) {
     fl::logger->Die()<<flag <<" is a required argument";    
   }
 }
 /**
  * @brief Defines arguments that at least one is required
  *        
  */
 void fl::ws::RequiredOrArgs(const boost::program_options::variables_map &vm,
     const std::string &flag // comma separated flags 
     ) {

   std::vector<std::string> tokens=fl::SplitString(flag, ",");
   bool found=false;
   for(size_t i=0; i<tokens.size(); ++i) {
     if (vm.count(tokens[i])!=0) {
       found=true;    
     }
   }
   if (found==false) {
     fl::logger->Die()<<"At least on of "<<flag <<" is a required argument"; 
   }
 }


 /**
  * @brief Defines arguments that are required for a specific flag
  *        
  */
 void fl::ws::RequiredArgs(const boost::program_options::variables_map &vm,
     const std::string &flag, // flag to be checked against 
     const std::string &required // comma separated string with required arguments
     ) {
   if (vm.count(flag)==0) {
     return;
   }
   std::vector<std::string> tokens=fl::SplitString(required, ",");
   for(size_t i=0; i<tokens.size(); ++i) {
     if (vm.count(tokens[i])==0) {
       fl::logger->Die()<<"When argument --"<<flag
         <<" is set it requires --"<<tokens[i]
         <<" to be defined too";
     }
   }
 }

 /**
  * @brief Defines arguments:value pairs 
  *        that are required for a specific flag:value
  *        
  */
 void fl::ws::RequiredArgValues(const boost::program_options::variables_map &vm,
     const std::string &flag, // flag to be checked against 
     const std::string &required // comma separated string with required arguments
     ) {

   std::vector<std::string> tokens=fl::SplitString(flag,":");
   if (tokens.size()!=2) {
     fl::logger->Die()<<"Something is wrong in the RequiredArgValues";
   }
   std::string argument=tokens[0];
   std::string value=tokens[1];
   if (vm.count(argument)==0) {
     return;
   } else {
     if (value!="*" && vm[argument].as<std::string>()!=value) {
       return;
     }
   }

   tokens.clear();
   tokens=fl::SplitString(required, ",");
   bool found=false;
   for(size_t i=0; i<tokens.size(); ++i) {
     std::vector<std::string> tokens1=fl::SplitString(tokens[i],":");
     if (tokens1.size()<2) {
       fl::logger->Die()<<"Something is wrong in the RequiredArgValues"; 
     }
     std::string arg=tokens1[0];
     if (vm.count(arg)==0) {
       fl::logger->Die()<<"When argument --"<<flag
         <<" is set it requires --"<<tokens[i]
         <<" to be defined too";
     } 
     if (tokens1[1]=="*") {
       found=true;
     } else {
       if (vm[arg].as<std::string>()==tokens1[1]) {
         found=true;
       }
     }
   }
   if (found==false) {
     std::string buffer;
     for(size_t i=0; i<tokens.size(); ++i) {
       buffer+=tokens[i]+",";
     }
     //boost::algorithm::replace_all(buffer, ":", "=");
     fl::logger->Die()<<"When argument --"<<flag
        <<" is set it requires the following flag"
        <<" to be defined in one of these values: "
        <<buffer;
    } 
  }

 /**
  * @brief Defines arguments that are impossible for a specific flag
  *        
  */
 void fl::ws::ImpossibleArgs(const boost::program_options::variables_map &vm,
     const std::string &flag, // flag to be checked against 
     const std::string &impossible // comma separated string with required arguments
     ) {
   if (vm.count(flag)==0) {
     return;
   }
   std::vector<std::string> tokens=fl::SplitString(impossible, ",");
   for(size_t i=0; i<tokens.size(); ++i) {
     if (vm.count(tokens[i])>0) {
       fl::logger->Die()<<"When argument --"<<flag
         <<" is set it requires --"<<tokens[i]
         <<" cannot be set too";
     }
   }
 }
 /**
  * @brief Defines arguments:value pairs 
  *        that are impossible for a specific flag:value
  *        
  */
 void fl::ws::ImpossibleArgValues(const boost::program_options::variables_map &vm,
     const std::string &flag, // flag to be checked against 
     const std::string &required // comma separated string with required arguments
     ) {

   std::vector<std::string> tokens=fl::SplitString(flag,":");
   if (tokens.size()!=2) {
     fl::logger->Die()<<"Something is wrong in the ImpossibleArgValues";
   }
   std::string argument=tokens[0];
   std::string value=tokens[1];
   if (vm.count(argument)==0) {
     return;
   } else {
     if (vm[argument].as<std::string>()!=value) {
       return;
     }
   }

   tokens.clear();
   tokens=fl::SplitString(required, ",");
   for(size_t i=0; i<tokens.size(); ++i) {
     std::vector<std::string> tokens1=fl::SplitString(tokens[i], ":");
     if (tokens1.size()<2) {
       fl::logger->Die()<<"Something is wrong in the ImpossibleArgValues"; 
     }
     std::string arg=tokens1[0];
     if (vm.count(arg)==0) {
       fl::logger->Die()<<"When argument --"<<flag
         <<" is set it requires --"<<tokens[i]
         <<" to be defined too";
     } 
     for(size_t i=1; i<tokens1.size(); ++i) {
       if (vm[arg].as<std::string>()==tokens1[i]) {
         fl::logger->Die()<<"When argument --"<<argument
            <<" is set to "<<value
         <<" , --"<<arg<<" cannot be set to "<<tokens1[i];
       } 
     }
   }
 }


 std::vector<std::string> fl::ws::GetFileSequence(const std::string &table_name,
        const boost::program_options::variables_map &vm) {
 
   if (vm.count(table_name+"_in")==0 && vm.count(table_name+"_prefix_in")==0 
       && vm.count(table_name+"_out")==0 && vm.count(table_name+"_prefix_out")==0) {
     fl::logger->Die()<<"You are required to specify --"+table_name+"_in or "
       +"--"+table_name+"_prefix_in or "+table_name+"_out or "
       +"--"+table_name+"_prefix_out";
   }
   if (vm.count(table_name+"_prefix_in")!=0 && vm.count(table_name+"_num_in")==0) {
     fl::logger->Die()<<"If you specify "+table_name+"_prefix_in then you should "
       "specify --"+table_name+"_num_in";
   }
   if (vm.count(table_name+"_prefix_out")!=0 && vm.count(table_name+"_num_out")==0) {
     fl::logger->Die()<<"If you specify "+table_name+"_prefix_out then you should "
       "specify --"+table_name+"_num_out";
   }

   std::vector<std::string> results;
   if (vm.count(table_name+"_in")==true) {
      std::string arg_value=vm[table_name+"_in"].as<std::string>();
      results=fl::SplitString(arg_value,",");
   } else {
     if (vm.count(table_name+"_prefix_in")==true) {
       std::string table_prefix=vm[table_name+"_prefix_in"].as<std::string>();
       int32 results_num=0;
       try{
         results_num=vm[table_name+"_num_in"].as<int32>();
       }
       catch(...) {
         fl::logger->Die()<<"There is somethng wrong with the flag --"+table_name+"_num_in";
       }
       results.resize(results_num);
       for(size_t i=0; i<results.size(); ++i) {
         results[i]=table_prefix+boost::lexical_cast<std::string>(i);
       }
     }
   }  
   if (vm.count(table_name+"_out")==true) {
      std::string arg_value=vm[table_name+"_out"].as<std::string>();
      results=fl::SplitString(arg_value,",");
   } else {
     if (vm.count(table_name+"_prefix_out")==true) {
       std::string table_prefix=vm[table_name+"_prefix_out"].as<std::string>();
       int32 results_num=vm[table_name+"_num_out"].as<int32>();
       results.resize(results_num);
       for(size_t i=0; i<results.size(); ++i) {
         results[i]=table_prefix+boost::lexical_cast<std::string>(i);
       }
     }
   }

   return results;
 }

  std::vector<std::string> fl::ws::GetFileSequence(const std::string &table_name,
        std::map<std::string, std::string> &vm) {
 
   if (vm.count(table_name+"_in")==0 && vm.count(table_name+"_prefix_in")==0 
       && vm.count(table_name+"_out")==0 && vm.count(table_name+"_prefix_out")==0) {
     fl::logger->Die()<<"You are required to specify --"+table_name+"_in or "
       +"--"+table_name+"_prefix_in or "+table_name+"_out or "
       +"--"+table_name+"_prefix_out";
   }
   if (vm.count(table_name+"_prefix_in")!=0 && vm.count(table_name+"_num_in")==0) {
     fl::logger->Die()<<"If you specify "+table_name+"_prefix_in then you should "
       "specify --"+table_name+"_num_in";
   }
   if (vm.count(table_name+"_prefix_out")!=0 && vm.count(table_name+"_num_out")==0) {
     fl::logger->Die()<<"If you specify "+table_name+"_prefix_out then you should "
       "specify --"+table_name+"_num_out";
   }

   std::vector<std::string> results;
   if (vm.count(table_name+"_in")==true) {
      std::string arg_value=vm[table_name+"_in"];
      results=fl::SplitString(arg_value,",");
   } else {
     if (vm.count(table_name+"_prefix_in")==true) {
       std::string table_prefix=vm[table_name+"_prefix_in"];
       int32 results_num=boost::lexical_cast<int32>(vm[table_name+"_num_in"]);
       results.resize(results_num);
       for(size_t i=0; i<results.size(); ++i) {
         results[i]=table_prefix+boost::lexical_cast<std::string>(i);
       }
     }
   }  
   if (vm.count(table_name+"_out")==true) {
      std::string arg_value=vm[table_name+"_out"];
      results=fl::SplitString(arg_value,",");
   } else {
     if (vm.count(table_name+"_prefix_out")==true) {
       std::string table_prefix=vm[table_name+"_prefix_out"];
       int32 results_num=boost::lexical_cast<int32>(vm[table_name+"_num_out"]);
       results.resize(results_num);
       for(size_t i=0; i<results.size(); ++i) {
         results[i]=table_prefix+boost::lexical_cast<std::string>(i);
       }
     }
   }

   return results;
 }

