IMPORT * FROM ML;


#option('compileOptions', ' -I/e/ismion/hg/PaperBoat/hpcc/karnagio/include -I/e/ismion/hg/PaperBoat/hpcc/karnagio/src/lib -I/e/ismion/hg/PaperBoat/hpcc/src -g');
#option('linkOptions', '-L/e/ismion/hg/PaperBoat/hpcc/bin/debug/,-lpaperboat,-lboost_thread-mt,-lboost_program_options-mt,-llapack -lblas');

DatumMacro(IdType, ValueType, DatumName) := MACRO
  DatumName := RECORD
    IdType id;
	  ML.Types.t_FieldNumber number;
	  ValueType value;
  END;
ENDMACRO;

DatumMacro(ML.Types.t_RecordID, ML.Types.t_FieldReal, RealDatum);
DatumMacro(ML.Types.t_RecordID, UNSIGNED1, Uint8Datum);
DatumMacro(ML.Types.t_RecordID, INTEGER, Int64Datum);

LDatumMacro(DatumType, DatumName) := MACRO
  DatumName := RECORD
    (DatumType)
    UNSIGNED1 file_id;
  END;
ENDMACRO;

LDatumMacro(RealDatum, RealLDatum);
LDatumMacro(Uint8Datum, Uint8LDatum);
LDatumMacro(Int64Datum, Int64LDatum);
  INTEGER DefineWorkSpace() := BEGINC++
    #ifndef PAPERBOAT_WORKSPACE
      #define PAPERBOAT_WORKSPACE
      #include "workspace/workspace_dev.h"
      fl::hpcc::WorkSpace ws; 
      std::stringstream str_stream;
    #endif
    #include "workspace/macros.h"
    #body
    fl::logger->SetLogger("debug");
    fl::logger->Init(&str_stream);
    ws.set_schedule_mode(2);
  ENDC++;
  DefineWorkSpace();

empty1:=DATASET([], RealLDatum);
empty2:=DATASET([], Uint8LDatum);
empty3:=DATASET([], Int64LDatum);
PaperBoat(DATASET(RealLDatum) realws=empty1,
          DATASET(Uint8LDatum) uint8ws=empty2,
          DATASET(Int64LDatum) int64ws=empty3) := MODULE 


  EXPORT DATASET(RealLDatum) GetRealTables(STRING arguments) :=BEGINC++
    std::string args(arguments, lenArguments);
    ws.ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<double> >(
         args,
         &__result,
         &__lenResult); 
  ENDC++;
  EXPORT DATASET(Uint8LDatum) GetUint8Tables(STRING arguments) :=BEGINC++
    std::string args(argumens, lenArguments);
    ws.ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<uint8> >(
         args,
         &__result,
         &__lenResult); 
  ENDC++;
  EXPORT DATASET(Int64LDatum) GetInt64Tables(STRING arguments) :=BEGINC++
    std::string args(argumens, lenArguments);
    ws.ExportAllDenseHPCCDataSets<fl::hpcc::SetDatum<uint64> >(
         args,
         &__result,
         &__lenResult); 
  ENDC++;

  EXPORT Allkn(STRING arguments) := MODULE
    STRING PbAllkn(DATASET(RealLDatum) realws,
          DATASET(Uint8LDatum) uint8ws,
          DATASET(Int64LDatum) int64ws,
          STRING arguments) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/allkn/allkn.h"
      #body
      PB_ECL_LOAD_DATA_MACRO
      ws.IndexAllReferencesQueries(vec);
      fl::ml::AllKN<boost::mpl::void_>::Run(&ws, vec);
      PB_ECL_EXPORT_LOG_MACRO 
    ENDC++;
    EXPORT STRING message := PbAllkn(realws, uint8ws, int64ws, arguments);
    EXPORT DATASET(RealLDatum) real_result := GetRealTables(arguments);
    EXPORT DATASET(Uint8LDatum) uint8_result := GetUint8Tables(arguments);
    EXPORT DATASET(Int64LDatum) int64_result := GetInt64Tables(arguments);    
  END;

  EXPORT Kde(STRING arguments) := MODULE
    STRING PbKde(DATASET(RealLDatum) realws,
          DATASET(Uint8LDatum) uint8ws,
          DATASET(Int64LDatum) int64ws,
          STRING arguments) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/kde/kde.h"
      #body
      PB_ECL_LOAD_DATA_MACRO
      ws.IndexAllReferencesQueries(vec);
      fl::ml::Kde<boost::mpl::void_>::Run(&ws, vec);
      PB_ECL_EXPORT_LOG_MACRO 
    ENDC++;
    EXPORT STRING message := PbKde(realws, uint8ws, int64ws, arguments);
    EXPORT DATASET(RealLDatum) real_result := GetRealTables(arguments);
    EXPORT DATASET(Uint8LDatum) uint8_result := GetUint8Tables(arguments);
    EXPORT DATASET(Int64LDatum) int64_result := GetInt64Tables(arguments);    
  END;

 
  EXPORT KMeans(STRING arguments) := MODULE
    STRING PbKMeans(DATASET(RealLDatum) realws,
          DATASET(Uint8LDatum) uint8ws,
          DATASET(Int64LDatum) int64ws,
          STRING arguments) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/kmeans/kmeans.h"
      #body
      PB_ECL_LOAD_DATA_MACRO
      ws.IndexAllReferencesQueries(vec);
      fl::ml::KMeans<boost::mpl::void_>::Run(&ws, vec);
      PB_ECL_EXPORT_LOG_MACRO 
    ENDC++;
    EXPORT STRING message := PbKMeans(realws, uint8ws, int64ws, arguments);
    EXPORT DATASET(RealLDatum) real_result := GetRealTables(arguments);
    EXPORT DATASET(Uint8LDatum) uint8_result := GetUint8Tables(arguments);
    EXPORT DATASET(Int64LDatum) int64_result := GetInt64Tables(arguments);    
  END;

  EXPORT Lasso(STRING arguments) := MODULE
    STRING PbLasso(DATASET(RealLDatum) realws,
          DATASET(Uint8LDatum) uint8ws,
          DATASET(Int64LDatum) int64ws,
          STRING arguments) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/lasso/lasso.h"
      #body
      PB_ECL_LOAD_DATA_MACRO
      fl::ml::Lasso<boost::mpl::void_>::Run(&ws, vec);
      PB_ECL_EXPORT_LOG_MACRO 
    ENDC++;
    EXPORT STRING message := PbLasso(realws, uint8ws, int64ws, arguments);
    EXPORT DATASET(RealLDatum) real_result := GetRealTables(arguments);
    EXPORT DATASET(Uint8LDatum) uint8_result := GetUint8Tables(arguments);
    EXPORT DATASET(Int64LDatum) int64_result := GetInt64Tables(arguments);    
  END;

  EXPORT Nmf(STRING arguments) := MODULE
    STRING PbNmf(DATASET(RealLDatum) realws,
          DATASET(Uint8LDatum) uint8ws,
          DATASET(Int64LDatum) int64ws,
          STRING arguments) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/nmf/nmf.h"
      #body
      PB_ECL_LOAD_DATA_MACRO
      fl::ml::Nmf<boost::mpl::void_>::Run(&ws, vec);
      PB_ECL_EXPORT_LOG_MACRO 
    ENDC++;
    EXPORT STRING message := PbNmf(realws, uint8ws, int64ws, arguments);
    EXPORT DATASET(RealLDatum) real_result := GetRealTables(arguments);
    EXPORT DATASET(Uint8LDatum) uint8_result := GetUint8Tables(arguments);
    EXPORT DATASET(Int64LDatum) int64_result := GetInt64Tables(arguments);    
  END;

  EXPORT Regression(STRING arguments) := MODULE
    STRING PbRegression(DATASET(RealLDatum) realws,
          DATASET(Uint8LDatum) uint8ws,
          DATASET(Int64LDatum) int64ws,
          STRING arguments) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/regression/linear_regression.h"
      #body
      PB_ECL_LOAD_DATA_MACRO
      fl::ml::LinearRegression<boost::mpl::void_>::Run(&ws, vec);
      PB_ECL_EXPORT_LOG_MACRO 
    ENDC++;
    EXPORT STRING message := PbRegression(realws, uint8ws, int64ws, arguments);
    EXPORT DATASET(RealLDatum) real_result := GetRealTables(arguments);
    EXPORT DATASET(Uint8LDatum) uint8_result := GetUint8Tables(arguments);
    EXPORT DATASET(Int64LDatum) int64_result := GetInt64Tables(arguments);    
  END;

  EXPORT Svd(STRING arguments) := MODULE
    STRING PbSvd(DATASET(RealLDatum) realws,
          DATASET(Uint8LDatum) uint8ws,
          DATASET(Int64LDatum) int64ws,
          STRING arguments) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/svd/svd.h"
      #body
      PB_ECL_LOAD_DATA_MACRO
      fl::ml::Svd<boost::mpl::void_>::Run(&ws, vec);
      PB_ECL_EXPORT_LOG_MACRO 
    ENDC++;
    EXPORT STRING message := PbSvd(realws, uint8ws, int64ws, arguments);
    EXPORT DATASET(RealLDatum) real_result := GetRealTables(arguments);
    EXPORT DATASET(Uint8LDatum) uint8_result := GetUint8Tables(arguments);
    EXPORT DATASET(Int64LDatum) int64_result := GetInt64Tables(arguments);    
  END;

  EXPORT Svm(STRING arguments) := MODULE
    STRING PbSvm(DATASET(RealLDatum) realws,
          DATASET(Uint8LDatum) uint8ws,
          DATASET(Int64LDatum) int64ws,
          STRING arguments) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/svm/svm.h"
      #body
      PB_ECL_LOAD_DATA_MACRO
      fl::ml::Svm<boost::mpl::void_>::Run(&ws, vec);
      PB_ECL_EXPORT_LOG_MACRO 
    ENDC++;
    EXPORT STRING message := PbSvm(realws, uint8ws, int64ws, arguments);
    EXPORT DATASET(RealLDatum) real_result := GetRealTables(arguments);
    EXPORT DATASET(Uint8LDatum) uint8_result := GetUint8Tables(arguments);
    EXPORT DATASET(Int64LDatum) int64_result := GetInt64Tables(arguments);    
  END;

  EXPORT OrthoRangeSearch(STRING arguments) := MODULE
    STRING PbOrthoRangeSearch(DATASET(RealLDatum) realws,
          DATASET(Uint8LDatum) uint8ws,
          DATASET(Int64LDatum) int64ws,
          STRING arguments) := BEGINC++
      #include <sstream>
      #include "workspace/set_datum.h"
      #include "mlpack/ortho_range_search/ortho_range_search.h"
      #body
      PB_ECL_LOAD_DATA_MACRO
      fl::ml::OrthoRangeSearch<boost::mpl::void_>::Run(&ws, vec);
      PB_ECL_EXPORT_LOG_MACRO 
    ENDC++;
    EXPORT STRING message := PbOrthoRangeSearch(realws, uint8ws, int64ws, arguments);
    EXPORT DATASET(RealLDatum) real_result := GetRealTables(arguments);
    EXPORT DATASET(Uint8LDatum) uint8_result := GetUint8Tables(arguments);
    EXPORT DATASET(Int64LDatum) int64_result := GetInt64Tables(arguments);    
  END;

END;

DATASET(RealLDatum) TestPBWorkSpace(DATASET(RealLDatum) realws, 
        INTEGER n_attributes,
        INTEGER n_entries) := BEGINC++
  #include "workspace/workspace.h"
  #include "workspace/numeric_datum.h"
  #include "workspace/set_datum.h"
  
#body
  ws.set_schedule_mode(2);
  ws.LoadDenseHPCCDataSet<fl::hpcc::SetDatum<double> >(
          "paparia",
          n_attributes,
          n_entries,
          realws);

  ws.ExportDenseHPCCDataSet<fl::hpcc::SetDatum<double> >(std::string("paparia"),&__result ,&__lenResult);
ENDC++;


x:=DATASET([{0, 0, 0.1, 4},
            {0, 1, 0.2, 4},
            {1, 0, 1.1, 4},
            {1, 1, 1.2, 4},
            {2, 0, 2.1, 4},
            {2, 1, 2.2, 4},
            {3, 0, 3.1, 4},
            {3, 1, 3.2, 4},
            {4, 0, 4.1, 4},
            {4, 1, 4.2, 4}], RealLDatum);

y:=TestPBWorkSpace(x, 5, 2);
z:=PaperBoat(x,,).Allkn(' --references_in=dense$double$4 '
               +' --k_neighbors=1 '
               +' --distances_out=dense$double$3');
OUTPUT(z.message);
OUTPUT(z.real_result);
