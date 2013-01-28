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
#ifndef PAPERBOAT_LIB_PBSERVICE_H_
#define PAPERBOAT_LIB_PBSERVICE_H_

#ifdef _WIN32
#define PAPERBOAT_LIB_CALL _cdecl
#ifdef PAPERBOAT_LIB_EXPORTS
#define PAPERBOAT_LIB_API __declspec(dllexport)
#else
#define PAPERBOATLIB_API __declspec(dllimport)
#endif
#else
#define PAPERBOAT_LIB_CALL
#define PAPERBOAT_LIB_API
#endif

#include "pbservice_common.h"

  
static const char * const HoleDefinition =   
  "SYSTEM\n"
  "MODULE (SYSTEM)\n"
  " FUNCTION StringFind(string src, string search, unsigned4 instance),unsigned4,c,name('elStringFind')\n"
  "END\n";
                            
                              
                            
static const char * const EclDefinition =
    "export ExampleLib := SERVICE\n"
    " unsigned integer4 StringFind(const string src, const string tofind, unsigned4 instance) : c, pure,entrypoint='elStringFind'; \n"
    "END;";

namespace fl { namespace hpcc { 
extern "C" {

  PAPERBOAT_LIB_API bool getECLPluginDefinition(ECLPluginDefinitionBlock *pb);
  PAPERBOAT_LIB_API void setPluginContext(IPluginContext * _ctx);                     
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GenSessionId(
    size32_t &__leResult,
    char*    &__result);
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL MakeWorkSpace(
    size32_t       lenSession_id,
    const char*    session_id);
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GetLog(
    size32_t &__lenResult,
    char* &__result);
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL LoadAllTables(
    size32_t       lenRealws,
    const void*    realws,
    // DATASET(Types.RealLDatum) realws=empty1,
    size32_t       lenUint8ws,
    const void*    uint8ws,
    // DATASET(Types.Uint8LDatum) uint8ws=empty2,
    size32_t       lenInt32ws,
    const void*    uint32ws,
    // DATASET(Types.Int32LDatum) int32ws=empty3,
    size32_t lenArguments,
    const char *arguments,
    // STRING arguments,
    size32_t lenSession_id,
    const char *session_id
    //STRING session_id
    );
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GetRealTables(
    size32_t &__lenResult,
    void* &__result,
    size32_t lenArguments,
    const char *arguments,
    size32_t lenSession_tag,
    const char *session_tag);
 PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL  GetUint8Tables( 
    size32_t &__lenResult,
    void* &__result,
    size32_t lenArguments,
    const char *arguments,
    size32_t lenSession_tag,
    const char *session_tag);
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GetInt32Tables( 
    size32_t &__lenResult,
    void* &__result,
    size32_t lenArguments,
    const char *arguments,
    size32_t lenSession_tag,
    const char *session_tag);
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GetUint32Tables(
    size32_t &__lenResult,
    void* &__result,
    size32_t lenArguments,
    const char *arguments,
    size32_t lenSession_tag,
    const char *session_tag);
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL GetInt64Tables(
    size32_t &__lenResult,
    void* &__result,
    size32_t lenArguments,
    const char *arguments,
    size32_t lenSession_tag,
    const char *session_tag);
  PAPERBOAT_LIB_API void PAPERBOAT_LIB_CALL Allkn(
    size32_t lenArguments,
    const char *arguments, 
    // STRING arguments,
    size32_t lenSession_id,
    const char *session_id
    //STRING session_id
    );
}
}}

#endif 
