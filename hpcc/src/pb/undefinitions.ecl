EXPORT UnDefinitions() :=  MACRO

  STRING UnDefineWorkSpace() := BEGINC++
    #ifndef PAPERBOAT_WORKSPACE
      #define PAPERBOAT_WORKSPACE
      #include "workspace/workspace_dev.h"
    #endif
    #include "workspace/macros.h"
    #body
    PB_ECL_EXPORT_LOG_MACRO  
    fl::logger->SetLogger("silent");
  ENDC++;

  UnDefineWorkSpace();

ENDMACRO;
