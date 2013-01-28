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
#include <iostream> 
#include <sys/time.h>
#include <sstream>
#include <map>
#include <string>
#include "boost/lexical_cast.hpp"
#include "workspace_dev.h"
#include "boost/shared_ptr.hpp"
#include "set_datum.h"
#include "fastlib/base/logger.h"
#include "pbservice.h"
#include "mlpack/allkn/allkn.h"
 
namespace fl {  namespace hpcc {
  IPluginContext * parentCtx = NULL;
 
  PAPERBOAT_LIB_API void setPluginContext(IPluginContext * _ctx) {fl::hpcc::parentCtx = _ctx; }
                                                               
  PAPERBOAT_LIB_API bool getECLPluginDefinition(ECLPluginDefinitionBlock *pb) {
    if (pb->size != sizeof(ECLPluginDefinitionBlock)) {
      return false;
      pb->magicVersion = PLUGIN_VERSION;
      pb->version = PAPERBOAT_LIB_VERSION " $Name$ $Id$";
      pb->moduleName = "PAPERBOAT";
      pb->ECL = EclDefinition;
      //pb->Hole = HoleDefinition;
      pb->flags = PLUGIN_IMPLICIT_MODULE;
      pb->description = "Paperboat library";
      return true;
    }
    return false;
  } 
  
 std::map<std::string, boost::shared_ptr<WorkSpace> > ws; 
 std::stringstream str_stream;
  
  
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GenSessionId(
      size32_t &__lenResult,
      char*    &__result) {
    timeval tv;
    gettimeofday(&tv, NULL);
    std::string random_tag =
    boost::lexical_cast<std::string>(tv.tv_sec)+
    boost::lexical_cast<std::string>(tv.tv_usec);   
    // we will store it in a dataset so we need an extra 4 bytes
    // for the length of the string first
    __result=(char*)rtlMalloc(random_tag.size()); 
    int random_tag_size=random_tag.size();
    memcpy((char*)__result, random_tag.data(), random_tag.size());
    __lenResult=random_tag.size();
    std::cout<<"random tag="<<random_tag<<std::endl;
  }
  
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL MakeWorkSpace(
      size32_t        lenSession_id,
      const char*     session_id) {
    ::fl::logger->SetLogger("debug");
    ::fl::logger->Init(&fl::hpcc::str_stream); 
    ::fl::logger->Message()<<"PaperBoat Library initialized";
 
    std::string session(session_id, lenSession_id);
    fl::hpcc::ws[session].reset(new WorkSpace());
    fl::hpcc::ws[session]->set_schedule_mode(2);
  }
  
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GetLog(
      size32_t &__lenResult,
      char* &__result) {
    __result=(char*)rtlMalloc(fl::hpcc::str_stream.str().size()); 
    memcpy(__result, fl::hpcc::str_stream.str().data(), fl::hpcc::str_stream.str().size()); 
    __lenResult=fl::hpcc::str_stream.str().size();
   ::fl::logger->SetLogger("silent");
  
  }
  
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL LoadAllTables(
      size32_t       lenRealws,
      const void*    realws,
      // DATASET(Types.RealLDatum) realws=empty1,
      size32_t       lenUint8ws,
      const void*    uint8ws,
      // DATASET(Types.Uint8LDatum) uint8ws=empty2,
      size32_t       lenInt32ws,
      const void*    int32ws,
      // DATASET(Types.Int32LDatum) int32ws=empty3,
      size32_t lenArguments,
      const char *arguments, 
      // STRING arguments,
      size32_t lenSession_id,
      const char *session_id
      //STRING session_id
      ) {
    
     std::string args(arguments, lenArguments); 
     std::string session(session_id, lenSession_id);
     if (fl::hpcc::ws.count(session)==0) {
       ::fl::logger->Warning()<<"You are using an invalid session_id"<<std::endl;
     }
     fl::hpcc::ws[session]->LoadAllDenseHPCCDataSets<SetDatum<double> >(args, 
        static_cast<const char*>(realws), 
        lenRealws); 
                    
     fl::hpcc::ws[session]->LoadAllDenseHPCCDataSets<SetDatum<uint8> >(args, 
        static_cast<const char*>(uint8ws), 
        lenUint8ws); 
     fl::hpcc::ws[session]->LoadAllDenseHPCCDataSets<SetDatum<int32> >(args, 
        static_cast<const char*>(int32ws), 
        lenInt32ws); 
  
  }
  
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GetRealTables(
      size32_t &__lenResult,
      void* &__result,
      size32_t lenArguments,
      const char *arguments,
      size32_t lenSession_tag,
      const char *session_tag) {
      
    if (::fl::global_exception) {
      __lenResult=0;
     return;
    }
    std::string session_id(session_tag, lenSession_tag);
    std::string args(arguments, lenArguments);
    if (fl::hpcc::ws.count(session_id)==0) {
      ::fl::logger->Warning()<<"You are using an invalid session_id"<<std::endl;
    }
    if (fl::hpcc::ws.count(session_id)!=0) {
      fl::hpcc::ws[session_id]->ExportAllDenseHPCCDataSets<SetDatum<double> >(
         args,
         &__result,
         &__lenResult); 
    } else {
      __result=NULL;
      __lenResult=0;
    }
  }
    
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL  GetUint8Tables( 
      size32_t &__lenResult,
      void* &__result,
      size32_t lenArguments,
      const char *arguments,
      size32_t lenSession_tag,
      const char *session_tag) {
  
    if (fl::global_exception) {
      __lenResult=0;
     return;
    }
    std::string session_id(session_tag, lenSession_tag);
    std::string args(arguments, lenArguments);

    if (ws.count(session_id)==0) {
      ::fl::logger->Warning()<<"You are using an invalid session_id"<<std::endl;
    }
    if (ws.count(session_id)!=0) {
      ::fl::hpcc::ws[session_id]->ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<uint8> >(
           args,
           &__result,
           &__lenResult); 
    } else {
      __result=NULL;
      __lenResult=0;
    }
  }
  
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GetInt32Tables( 
      size32_t &__lenResult,
      void* &__result,
      size32_t lenArguments,
      const char *arguments,
      size32_t lenSession_tag,
      const char *session_tag) {
      
    if (fl::global_exception) {
      __lenResult=0;
     return;
    }
    std::string session_id(session_tag, lenSession_tag);
    std::string args(arguments, lenArguments);

    if (ws.count(session_id)==0) {
      ::fl::logger->Warning()<<"You are using an invalid session_id"<<std::endl;
    }
    if (ws.count(session_id)!=0) {
      ws[session_id]->ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<int32> >(
           args,
           &__result,
           &__lenResult); 
    } else {
      __result=NULL;
      __lenResult=0;
    }
  }
  
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GetUint32Tables(
      size32_t &__lenResult,
      void* &__result,
      size32_t lenArguments,
      const char *arguments,
      size32_t lenSession_tag,
      const char *session_tag) {
       
    if (global_exception) {
      __lenResult=0;
     return;
    }
    std::string session_id(session_tag, lenSession_tag);
    std::string args(arguments, lenArguments);

    if (fl::hpcc::ws.count(session_id)==0) {
      ::fl::logger->Warning()<<"You are using an invalid session_id"<<std::endl;
    }
    if (fl::hpcc::ws.count(session_id)!=0) {  
      ws[session_id]->ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<uint32> >(
           args,
           &__result,
           &__lenResult); 
    } else {
      __result=NULL;
      __lenResult=0;
    }
  }  
  
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GetInt64Tables(
      size32_t &__lenResult,
      void* &__result,
      size32_t lenArguments,
      const char *arguments,
      size32_t lenSession_tag,
      const char *session_tag) {
      
    if (global_exception) {
      __lenResult=0;
     return;
    }
    std::string session_id(session_tag, lenSession_tag);
    std::string args(arguments, lenArguments);

    if (ws.count(session_id)==0) {
      ::fl::logger->Warning()<<"You are using an invalid session_id"<<std::endl;
    }
    if (ws.count(session_id)!=0) { 
      ws[session_id]->ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<int64> >(
         args,
         &__result,
         &__lenResult); 
    } else {
      __result=NULL;
      __lenResult=0;
    }
  } 
    
  
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL Allkn(
      size32_t lenArguments,
      const char *arguments, 
      // STRING arguments,
      size32_t lenSession_tag,
      const char *session_tag
      //STRING session_id
      ) {
   
    std::string session_id(session_tag, lenSession_tag);
    std::string args(arguments, lenArguments); 

    if (ws.count(session_id)==0) {
      ::fl::logger->Warning()<<"You are using an invalid session_id"<<std::endl;
    }
    try {
      std::vector<std::string> vec;
      vec=SplitString(args, " ");
      ws[session_id]->IndexAllReferencesQueries(vec);
      fl::ml::AllKN<boost::mpl::void_>::Run(fl::hpcc::ws[session_id].get(), vec);
     
    } 
    catch(...) {
      boost::mutex::scoped_lock lock(*fl::global_exception_mutex);
      global_exception=boost::current_exception();
    } 
  }
}} 
