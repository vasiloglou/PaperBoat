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

#ifndef PAPERBOAT_HPCC_SRC_MACROS_H_
#define PAPERBOAT_HPCC_SRC_MACROS_H_

#define PB_ECL_LOAD_DATA_MACRO \
      std::string session(session_id, lenSession_id);\
      ws[session].reset(new fl::hpcc::WorkSpace());\
      ws[session]->set_schedule_mode(2);\
      std::string args(arguments, lenArguments); \
      ws[session]->LoadAllDenseHPCCDataSets<fl::hpcc::SetDatum<double> >(args, \
          static_cast<const char*>(realws), \
          lenRealws); \
                      \
      ws[session]->LoadAllDenseHPCCDataSets<fl::hpcc::SetDatum<uint8> >(args, \
          static_cast<const char*>(uint8ws), \
          lenUint8ws); \
      ws[session]->LoadAllDenseHPCCDataSets<fl::hpcc::SetDatum<int32> >(args, \
          static_cast<const char*>(int32ws), \
          lenInt32ws); \
                       \
      ws[session]->LoadAllSparseHPCCDataSets<fl::hpcc::SetDatum<double> >(args, \
          static_cast<const char*>(realws), \
          lenRealws); \
      ws[session]->LoadAllSparseHPCCDataSets<fl::hpcc::SetDatum<uint8> >(args, \
          static_cast<const char*>(uint8ws), \
          lenUint8ws); \
                       \
      ws[session]->LoadAllSparseHPCCDataSets<fl::hpcc::SetDatum<int32> >(args, \
          static_cast<const char*>(int32ws), \
          lenInt32ws); \
      std::vector<std::string> vec = fl::SplitString(args, " ");

#define PB_ECL_EXPORT_LOG_MACRO \
      __result=(char*)rtlMalloc(str_stream.str().size()); \
      memcpy(__result, str_stream.str().data(), str_stream.str().size()); \
      __lenResult=str_stream.str().size();\

#endif
