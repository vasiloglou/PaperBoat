IMPORT * FROM $;

empty1:=DATASET([], Types.RealLDatum);
empty2:=DATASET([], Types.Uint8LDatum);
empty3:=DATASET([], Types.Int32LDatum);


EXPORT Karnagio(DATASET(Types.RealLDatum) realws=empty1,
         DATASET(Types.Uint8LDatum) uint8ws=empty2,
         DATASET(Types.Int32LDatum) int32ws=empty3) := MODULE 

  SHARED DATASET(Types.RealLDatum) GetRealTables(STRING arguments, 
      STRING session_tag) :=BEGINC++
    #option once
    #option pure 
    
    #include "workspace/set_datum.h"
    #body
    if (fl::global_exception) {
      __lenResult=0;
     return;
    }
    std::string session_id(session_tag, lenSession_tag);
    std::string args(arguments, lenArguments);
    fl::hpcc::ws[session_id]->ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<double> >(
         args,
         &__result,
         &__lenResult); 
  ENDC++;
  
  SHARED DATASET(Types.Uint8LDatum) GetUint8Tables(STRING arguments, 
      STRING session_id) :=BEGINC++
    #option pure
    #option once
    #include "workspace/set_datum.h"
    #body

    if (fl::global_exception) {
      __lenResult=0;
     return;
    }
    std::string session_id(session_tag, lenSession_tag);
    std::string args(arguments, lenArguments);
    fl::hpcc::ws[session_id]->ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<uint8> >(
         args,
         &__result,
         &__lenResult); 
  ENDC++;
  SHARED DATASET(Types.Int32LDatum) GetInt32Tables(STRING arguments,
      STRING session_tag) :=BEGINC++
    #option once
    #option pure

    #include "workspace/set_datum.h"
    #body
    
    if (fl::global_exception) {
      __lenResult=0;
     return;
    }
    std::string session_id(session_tag, lenSession_tag);
    std::string args(arguments, lenArguments);
    fl::hpcc::ws[session_id]->ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<int32> >(
         args,
         &__result,
         &__lenResult); 
  ENDC++;

  SHARED DATASET(Types.UInt32LDatum) GetUInt32Tables(STRING arguments,
      STRING session_tag) :=BEGINC++
    #option once
    #option pure

    #include "workspace/set_datum.h"
    #body
     
    if (fl::global_exception) {
      __lenResult=0;
     return;
    }
    std::string session_id(session_tag, lenSession_tag);
    std::string args(arguments, lenArguments);
    fl::hpcc::ws[session_id]->ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<uint32> >(
         args,
         &__result,
         &__lenResult); 
  ENDC++;

  SHARED DATASET(Types.Int64LDatum) GetInt64Tables(STRING arguments, 
      STRING session_tag) :=BEGINC++
    #option pure
    #option once
    #include "workspace/set_datum.h"
    #body
    
    if (fl::global_exception) {
      __lenResult=0;
     return;
    }
    std::string session_id(session_tag, lenSession_tag);
    std::string args(arguments, lenArguments);
    fl::hpcc::ws[session_id]->ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<int64> >(
         args,
         &__result,
         &__lenResult); 
  ENDC++;
  
  SHARED STRING GenSession() := BEGINC++
      #include <iostream> 
      #include <sys/time.h>
      #body
      timeval tv;
      gettimeofday(&tv, NULL);
      std::string random_tag =boost::lexical_cast<std::string>(tv.tv_usec);   
      __result=(char*)rtlMalloc(random_tag.size()); 
      memcpy(__result, random_tag.data(), random_tag.size());
      __lenResult=random_tag.size();
  ENDC++;

  EXPORT Allkn(STRING arguments) := MODULE
    SHARED INTEGER PbAllkn(DATASET(Types.RealLDatum) realws,
          DATASET(Types.Uint8LDatum) uint8ws,
          DATASET(Types.Int32LDatum) int32ws,
          STRING arguments,
          STRING session_id) := BEGINC++

      #include <sstream>
      #include "workspace/workspace.h"
      #include "fastlib/base/logger.h"
      #include "mlpack/allkn/allkn.h"
      #include "workspace/set_datum.h"
      #include "workspace/macros.h"

      #body 
     
      try {
        PB_ECL_LOAD_DATA_MACRO
        fl::hpcc::ws[session]->IndexAllReferencesQueries(vec);
        fl::ml::AllKN<boost::mpl::void_>::Run(fl::hpcc::ws[session].get(), vec);
        return 0;
      } 
      catch(...) {
        boost::mutex::scoped_lock lock(*fl::global_exception_mutex);
        fl::global_exception=boost::current_exception();
        return 1;
      } 
    ENDC++;
        
    SHARED STRING session_id := GenSession();
    EXPORT INTEGER call :=PbAllkn(realws, uint8ws, int32ws, arguments, session_id);
    EXPORT DATASET(Types.RealLDatum) real_result := GetRealTables(arguments, session_id);
    EXPORT DATASET(Types.Uint8LDatum) uint8_result := GetUint8Tables(arguments, session_id);
    EXPORT DATASET(Types.Int32LDatum) int32_result := GetInt32Tables(arguments, session_id);    
    EXPORT DATASET(Types.UInt32LDatum) uint32_result := GetUInt32Tables(arguments, session_id);    

  END;

  EXPORT Kde(STRING arguments) := MODULE
    SHARED INTEGER PbKde(DATASET(Types.RealLDatum) realws,
          DATASET(Types.Uint8LDatum) uint8ws,
          DATASET(Types.Int32LDatum) int32ws,
          STRING arguments,
          STRING session_id) := BEGINC++

      #include <sstream>
      #include "workspace/workspace.h"
      #include "fastlib/base/logger.h"
      #include "mlpack/kde/kde.h"
      #include "workspace/set_datum.h"
      #include "workspace/macros.h"

      #body

      try {
        PB_ECL_LOAD_DATA_MACRO
        fl::hpcc::ws[session]->IndexAllReferencesQueries(vec);
        fl::ml::Kde<boost::mpl::void_>::Run(fl::hpcc::ws[session].get(), vec);
        return 0;
      } 
      catch(...) {
        boost::mutex::scoped_lock lock(*fl::global_exception_mutex);
        fl::global_exception=boost::current_exception();
        return 1;
      }
    ENDC++;

    SHARED STRING session_id := GenSession();
    EXPORT INTEGER call :=PbKde(realws, uint8ws, int32ws, arguments, session_id);
    EXPORT DATASET(Types.RealLDatum) real_result := GetRealTables(arguments, session_id);
    EXPORT DATASET(Types.Uint8LDatum) uint8_result := GetUint8Tables(arguments, session_id);
    EXPORT DATASET(Types.Int32LDatum) int32_result := GetInt32Tables(arguments, session_id);    
    EXPORT DATASET(Types.UInt32LDatum) uint32_result := GetUInt32Tables(arguments, session_id);    

  END;

  EXPORT Npr(STRING arguments) := MODULE
   SHARED INTEGER PbNpr(DATASET(Types.RealLDatum) realws,
          DATASET(Types.Uint8LDatum) uint8ws,
          DATASET(Types.Int32LDatum) int32ws,
          STRING arguments,
          STRING session_id) := BEGINC++

      #include <sstream>
      #include "workspace/workspace.h"
      #include "fastlib/base/logger.h"
      #include "mlpack/nonparametric_regression/nonparametric_regression.h"
      #include "workspace/set_datum.h"
      #include "workspace/macros.h"

      #body

      try {
        PB_ECL_LOAD_DATA_MACRO
        fl::hpcc::ws[session]->IndexAllReferencesQueries(vec);
        fl::ml::NonParametricRegression<boost::mpl::void_>::Run(fl::hpcc::ws[session].get(), vec);
        return 0;
      } 
      catch(...) {
        boost::mutex::scoped_lock lock(*fl::global_exception_mutex);
        fl::global_exception=boost::current_exception();
        return 1;
      }
    ENDC++;

    SHARED STRING session_id := GenSession();
    EXPORT INTEGER call :=PbNpr(realws, uint8ws, int32ws, arguments, session_id);
    EXPORT DATASET(Types.RealLDatum) real_result := GetRealTables(arguments, session_id);
    EXPORT DATASET(Types.Uint8LDatum) uint8_result := GetUint8Tables(arguments, session_id);
    EXPORT DATASET(Types.Int32LDatum) int32_result := GetInt32Tables(arguments, session_id);    
    EXPORT DATASET(Types.UInt32LDatum) uint32_result := GetUInt32Tables(arguments, session_id);    

  END;


  EXPORT KMeans(STRING arguments) := MODULE
    SHARED INTEGER PbKMeans(DATASET(Types.RealLDatum) realws,
          DATASET(Types.Uint8LDatum) uint8ws,
          DATASET(Types.Int32LDatum) int32ws,
          STRING arguments,
          STRING session_id) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/clustering/kmeans.h"
      #body

      try {
        PB_ECL_LOAD_DATA_MACRO
        fl::hpcc::ws[session]->IndexAllReferencesQueries(vec);
        fl::ml::KMeans<boost::mpl::void_>::Run(fl::hpcc::ws[session].get(), vec);
        return 0;
      } 
      catch(...) {
        boost::mutex::scoped_lock lock(*fl::global_exception_mutex);
        fl::global_exception=boost::current_exception();
        return 1;
      }
    ENDC++;

    SHARED STRING session_id := GenSession();
    EXPORT INTEGER call :=PbKMeans(realws, uint8ws, int32ws, arguments, session_id);
    EXPORT DATASET(Types.RealLDatum) real_result := GetRealTables(arguments, session_id);
    EXPORT DATASET(Types.Uint8LDatum) uint8_result := GetUint8Tables(arguments, session_id);
    EXPORT DATASET(Types.Int32LDatum) int32_result := GetInt32Tables(arguments, session_id);    
    EXPORT DATASET(Types.UInt32LDatum) uint32_result := GetUInt32Tables(arguments, session_id);    

  END;

  EXPORT Regression(STRING arguments) := MODULE
    SHARED INTEGER PbLasso(DATASET(Types.RealLDatum) realws,
          DATASET(Types.Uint8LDatum) uint8ws,
          DATASET(Types.Int32LDatum) int32ws,
          STRING arguments,
          STRING session_id) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/regression/linear_regression.h"
      #body

      try {
        PB_ECL_LOAD_DATA_MACRO
        fl::ml::LinearRegression<boost::mpl::void_>::Run(fl::hpcc::ws[session].get(), vec);
        return 0;
      } 
      catch(...) {
        boost::mutex::scoped_lock lock(*fl::global_exception_mutex);
        fl::global_exception=boost::current_exception();
        return 1;      
     }
    ENDC++;

    SHARED STRING session_id := GenSession();
    EXPORT INTEGER call :=PbLasso(realws, uint8ws, int32ws, arguments, session_id);
    EXPORT DATASET(Types.RealLDatum) real_result := GetRealTables(arguments, session_id);
    EXPORT DATASET(Types.Uint8LDatum) uint8_result := GetUint8Tables(arguments, session_id);
    EXPORT DATASET(Types.Int32LDatum) int32_result := GetInt32Tables(arguments, session_id);    
    EXPORT DATASET(Types.UInt32LDatum) uint32_result := GetUInt32Tables(arguments, session_id);    

  END;


  EXPORT Lasso(STRING arguments) := MODULE
    SHARED INTEGER PbLasso(DATASET(Types.RealLDatum) realws,
          DATASET(Types.Uint8LDatum) uint8ws,
          DATASET(Types.Int32LDatum) int32ws,
          STRING arguments,
          STRING session_id) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/lasso/lasso.h"
      #body

      try {
        PB_ECL_LOAD_DATA_MACRO
        fl::ml::Lasso<boost::mpl::void_>::Run(fl::hpcc::ws[session].get(), vec);
        return 0;
      } 
      catch(...) {
        boost::mutex::scoped_lock lock(*fl::global_exception_mutex);
        fl::global_exception=boost::current_exception();
        return 1;       
      }
    ENDC++;

    SHARED STRING session_id := GenSession();
    EXPORT INTEGER call :=PbLasso(realws, uint8ws, int32ws, arguments, session_id);
    EXPORT DATASET(Types.RealLDatum) real_result := GetRealTables(arguments, session_id);
    EXPORT DATASET(Types.Uint8LDatum) uint8_result := GetUint8Tables(arguments, session_id);
    EXPORT DATASET(Types.Int32LDatum) int32_result := GetInt32Tables(arguments, session_id);    
    EXPORT DATASET(Types.UInt32LDatum) uint32_result := GetUInt32Tables(arguments, session_id);    

  END;

  EXPORT Nmf(STRING arguments) := MODULE
    SHARED INTEGER PbNmf(DATASET(Types.RealLDatum) realws,
          DATASET(Types.Uint8LDatum) uint8ws,
          DATASET(Types.Int32LDatum) int32ws,
          STRING arguments,
          STRING session_id) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/nmf/nmf.h"
      #body

      try {
        PB_ECL_LOAD_DATA_MACRO
        fl::ml::Nmf::Run(fl::hpcc::ws[session].get(), vec);
        return 0;
      } 
      catch(...) {
        boost::mutex::scoped_lock lock(*fl::global_exception_mutex);
        fl::global_exception=boost::current_exception();
        return 1;       
      }
    ENDC++;

    SHARED STRING session_id := GenSession();
    EXPORT INTEGER call :=PbNmf(realws, uint8ws, int32ws, arguments, session_id);
    EXPORT DATASET(Types.RealLDatum) real_result := GetRealTables(arguments, session_id);
    EXPORT DATASET(Types.Uint8LDatum) uint8_result := GetUint8Tables(arguments, session_id);
    EXPORT DATASET(Types.Int32LDatum) int32_result := GetInt32Tables(arguments, session_id);    
    EXPORT DATASET(Types.UInt32LDatum) uint32_result := GetUInt32Tables(arguments, session_id);      
  END;

  EXPORT Svd(STRING arguments) := MODULE
    SHARED INTEGER PbSvd(DATASET(Types.RealLDatum) realws,
          DATASET(Types.Uint8LDatum) uint8ws,
          DATASET(Types.Int32LDatum) int32ws,
          STRING arguments,
          STRING session_id) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/svd/svd.h"
      #body

      try {
        PB_ECL_LOAD_DATA_MACRO
        fl::ml::Svd<boost::mpl::void_>::Run(fl::hpcc::ws[session].get(), vec);
        return 0;
      } 
      catch(...) {
        boost::mutex::scoped_lock lock(*fl::global_exception_mutex);
        fl::global_exception=boost::current_exception();
        return 1;       
      }
    ENDC++;

    SHARED STRING session_id := GenSession();
    EXPORT INTEGER call :=PbSvd(realws, uint8ws, int32ws, arguments, session_id);
    EXPORT DATASET(Types.RealLDatum) real_result := GetRealTables(arguments, session_id);
    EXPORT DATASET(Types.Uint8LDatum) uint8_result := GetUint8Tables(arguments, session_id);
    EXPORT DATASET(Types.Int32LDatum) int32_result := GetInt32Tables(arguments, session_id);    
    EXPORT DATASET(Types.UInt32LDatum) uint32_result := GetUInt32Tables(arguments, session_id);      
  END;

  EXPORT Svm(STRING arguments) := MODULE
    SHARED INTEGER PbSvm(DATASET(Types.RealLDatum) realws,
          DATASET(Types.Uint8LDatum) uint8ws,
          DATASET(Types.Int32LDatum) int32ws,
          STRING arguments,
          STRING session_id) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/svm/svm.h"
      #body

      try {
        PB_ECL_LOAD_DATA_MACRO
        fl::hpcc::ws[session]->IndexAllReferencesQueries(vec);
        fl::ml::Svm<boost::mpl::void_>::Run(fl::hpcc::ws[session].get(), vec);
        return 0;
      } 
      catch(...) {
        boost::mutex::scoped_lock lock(*fl::global_exception_mutex);
        fl::global_exception=boost::current_exception();
        return 1;       
      }
    ENDC++;

    SHARED STRING session_id := GenSession();
    EXPORT INTEGER call :=PbSvm(realws, uint8ws, int32ws, arguments, session_id);
    EXPORT DATASET(Types.RealLDatum) real_result := GetRealTables(arguments, session_id);
    EXPORT DATASET(Types.Uint8LDatum) uint8_result := GetUint8Tables(arguments, session_id);
    EXPORT DATASET(Types.Int32LDatum) int32_result := GetInt32Tables(arguments, session_id);    
    EXPORT DATASET(Types.UInt32LDatum) uint32_result := GetUInt32Tables(arguments, session_id);    
  END;

  EXPORT OrthoRangeSearch(STRING arguments) := MODULE
    SHARED INTEGER PbOrthoRangeSearch(DATASET(Types.RealLDatum) realws,
          DATASET(Types.Uint8LDatum) uint8ws,
          DATASET(Types.Int32LDatum) int32ws,
          STRING arguments,
          STRING session_id) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/ortho_range_search/ortho_range_search.h"
      #body

      try {
        PB_ECL_LOAD_DATA_MACRO
        fl::ml::OrthoRangeSearch<boost::mpl::void_>::Run(fl::hpcc::ws[session].get(), vec);
        return 0;
      } 
      catch(...) {
        boost::mutex::scoped_lock lock(*fl::global_exception_mutex);
        fl::global_exception=boost::current_exception();
        return 1;       
      }
    ENDC++;

    SHARED STRING session_id := GenSession();
    EXPORT INTEGER call :=PbOrthoRangeSearch(realws, uint8ws, int32ws, arguments, session_id);
    EXPORT DATASET(Types.RealLDatum) real_result := GetRealTables(arguments, session_id);
    EXPORT DATASET(Types.Uint8LDatum) uint8_result := GetUint8Tables(arguments, session_id);
    EXPORT DATASET(Types.Int32LDatum) int32_result := GetInt32Tables(arguments, session_id);    
    EXPORT DATASET(Types.UInt32LDatum) uint32_result := GetUInt32Tables(arguments, session_id);     
  END;

END;


