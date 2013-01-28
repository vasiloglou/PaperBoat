IMPORT TYPES;
 
empty1:=DATASET([], Types.RealLDatum);
empty2:=DATASET([], Types.Uint8LDatum);
empty3:=DATASET([], Types.Int32LDatum);

EXPORT PB := SERVICE

  STRING GenSessionId() : LIBRARY='eclpb', entrypoint='GenSessionId';
  MakeWorkSpace(STRING session_id) : LIBRARY='eclpb', entrypoint='MakeWorkSpace';
  STRING GetLog() : LIBRARY='eclpb', entrypoint='GetLog';
  LoadAllTables(
    DATASET(Types.RealLDatum) realws=empty1,
    DATASET(Types.Uint8LDatum) uint8ws=empty2,
    DATASET(Types.Int32LDatum) int32ws=empty3,
    STRING arguments,
    STRING session_id) : c, action, LIBRARY='eclpb', entrypoint='LoadAllTables';
 DATASET(Types.RealLDatum) GetRealTables(
   STRING arguments,    
   STRING session_id) : c, action, LIBRARY='eclpb', entrypoint='GetRealTables';
 DATASET(Types.Uint8LDatum) GetUint8Tables(
   STRING arguments,    
   STRING session_id) : c, action, LIBRARY='eclpb', entrypoint='GetUint8Tables';
 DATASET(Types.Int32LDatum) GetInt32Tables(
   STRING arguments,    
   STRING session_id) : c, action, LIBRARY='eclpb', entrypoint='GetInt32Tables';
 DATASET(Types.Uint32LDatum) GetUint32Tables(
   STRING arguments,    
   STRING session_id) : c, action, LIBRARY='eclpb', entrypoint='GetUint32Tables' ;
 DATASET(Types.Int64LDatum) GetInt64Tables(
   STRING arguments,    
   STRING session_id) : c, action, LIBRARY='eclpb', entrypoint='GetInt64Tables';
 Allkn(STRING arguments,
   STRING session_id
   ) : c, action, LIBRARY='eclpb', entrypoint='Allkn'; 
END; 


