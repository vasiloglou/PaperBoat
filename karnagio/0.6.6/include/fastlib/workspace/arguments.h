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

#ifndef INCLUDE_FASTLIB_WORKSPACE_ARGUMENTS_H_
#define INCLUDE_FASTLIB_WORKSPACE_ARGUMENTS_H_
#include <vector>
#include <map>
#include <stdarg.h>
#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "boost/program_options.hpp"

namespace fl {namespace ws {
 class Arguments {
   public:
     template<typename T>
     void Add(const std::string &key, const T& value) {
       Add(key, boost::lexical_cast<std::string>(value));
     }

     void Add(const std::string &key, const std::string &value) {
       std::string token("--");
       token.append(key).append("=").append(value);
       args_.push_back(token);
     }
     void Add(const std::vector<std::string> &some_args) {
       args_.insert(args_.begin(), some_args.begin(), some_args.end());
     }

     const std::vector<std::string> & args() const {
       return args_;
     }
   private:
     std::vector<std::string> args_;  
 };

 std::vector<std::string> MakeArgsFromPrefix(
     const std::vector<std::string> &vec, const std::string &prefix);

 std::vector<std::string> RemoveArgsWithPrefix(
     const std::vector<std::string> &vec, const std::string &prefix);

 std::map<std::string, std::string> GetArgumentPairs(
     const std::vector<std::string> &args);

 
 /**
  * @brief Required argument
  */
 void RequiredArgs(const boost::program_options::variables_map &vm,
     const std::string &flag); // flag to be checked against
/**
  * @brief At least one of the flags must be present
  */
 void RequiredOrArgs(const boost::program_options::variables_map &vm,
     const std::string &flags); // flag to be checked against

 /**
   * @brief Defines arguments that are required for a specific flag
   *        
   */
 void RequiredArgs(const boost::program_options::variables_map &vm,
     const std::string &flag, // flag to be checked against 
     const std::string &required // comma separated string with required arguments
     );

  /**
   * @brief Defines arguments:value pairs 
   *        that are required for a specific flag:value
   *        
   */
  void RequiredArgValues(const boost::program_options::variables_map &vm,
      const std::string &flag, // flag to be checked against 
      const std::string &required // comma separated string with required arguments
      ); 

  /**
   * @brief Defines arguments that are impossible for a specific flag
   *        
   */
  void ImpossibleArgs(const boost::program_options::variables_map &vm,
      const std::string &flag, // flag to be checked against 
      const std::string &impossible // comma separated string with required arguments
      ); 

  /**
   * @brief Defines arguments:value pairs 
   *        that are impossible for a specific flag:value
   *        
   */
  void ImpossibleArgValues(const boost::program_options::variables_map &vm,
     const std::string &flag, // flag to be checked against 
     const std::string &required // comma separated string with required arguments
     );


   /**
    * @brief In Paperboat a sequence of tables can be imported/exported in two ways
    *        --my_tables_prefix_in=table1,other_table,this_table
    *        or 
    *        --my_tables_in=prefix --my_tables_num_in=3
    *          this will import tables prefix0 prefix1 prefix2
    *          for output tables you can use
    *        --my_tables_out=table1,other_table,this_table
    *        or 
    *        --my_tables_out=prefix --my_tables_num_out=3
    */
    std::vector<std::string> GetFileSequence(const std::string &table_name,
        const boost::program_options::variables_map &vm);
    std::vector<std::string> GetFileSequence(const std::string &table_name,
        std::map<std::string, std::string> &vm);
}}
#endif
