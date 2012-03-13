EXPORT Definitions() :=  MACRO
  #option('compileOptions', ' -I/e/ismion/hg/PaperBoat/hpcc/karnagio/include -I/e/ismion/hg/PaperBoat/hpcc/karnagio/src/lib -I/e/ismion/hg/PaperBoat/hpcc/src -g');
  #option('linkOptions', '-L/e/ismion/hg/PaperBoat/hpcc/bin/debug/,-lpaperboat,-lboost_thread-mt,-lboost_program_options-mt,-llapack -lblas');

  STRING DefineWorkSpace() := BEGINC++
    #ifndef PAPERBOAT_WORKSPACE
      #define PAPERBOAT_WORKSPACE
      #include "workspace/workspace_dev.h"
      #include "boost/shared_ptr.hpp"
      #include <map>
      #include <string>
      std::map<std::string, boost::shared_ptr<fl::hpcc::WorkSpace> > ws; 
      std::stringstream str_stream;
    #endif
    #include "workspace/macros.h"
    #body
    fl::logger->SetLogger("debug");
    fl::logger->Init(&str_stream); 
    std::string message="PaperBoat Library initialized";
    __result=(char *)rtlMalloc(message.size());
    memcpy(__result, message.data(), message.size());
    __lenResult=message.size();
  ENDC++;

  DefineWorkSpace();

ENDMACRO;
