EXPORT Definitions() :=  MACRO
  #option('compileOptions', 
      ' -I/usr/include/karnagio/include'
      +' -I/usr/local/include/karnagio/include'
      +' -I/usr/include/karnagio/src/' 
      +' -I/usr/local/include/karnagio/src'
      +' -I/usr/include/ecl-pb-glue'
      +' -I/usr/local/include/ecl-pb-glue');
  #option('linkOptions', '-lecl-paperboat,'
                         +'-lboost_thread-mt,'
                         +'-lboost_program_options-mt,'
                         +'-llapack,'
                         +'-lblas'
                         );

  STRING DefineWorkSpace() := BEGINC++
    #ifndef PAPERBOAT_WORKSPACE
      #define PAPERBOAT_WORKSPACE
      #include "workspace_dev.h"
      #include "boost/shared_ptr.hpp"
      #include <map>
      #include <string>
      namespace fl { namespace hpcc {
        static std::map<std::string, boost::shared_ptr<fl::hpcc::WorkSpace> > ws; 
        std::stringstream str_stream;
      }}
    #endif
    #include "macros.h"
    #body
    fl::logger->SetLogger("debug");
    fl::logger->Init(&fl::hpcc::str_stream); 
    std::string message="PaperBoat Library initialized";
    __result=(char *)rtlMalloc(message.size());
    memcpy(__result, message.data(), message.size());
    __lenResult=message.size();
  ENDC++;

  DefineWorkSpace();

ENDMACRO;
