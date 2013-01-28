EXPORT UnDefinitions() :=  MACRO

  STRING UnDefineWorkSpace() := BEGINC++
    #ifndef PAPERBOAT_WORKSPACE
      #define PAPERBOAT_WORKSPACE
      #include "workspace_dev.h"
    #endif
    #include "macros.h"
    #body
    PB_ECL_EXPORT_LOG_MACRO  
    fl::logger->SetLogger("silent");
  ENDC++;

  UnDefineWorkSpace();

ENDMACRO;
