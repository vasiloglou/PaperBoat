//IMPORT PB;
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
   STRING session_id) : c, pure, LIBRARY='eclpb', entrypoint='GetUint8Tables';
 DATASET(Types.Int32LDatum) GetInt32Tables(
   STRING arguments,    
   STRING session_id) : c, action, LIBRARY='eclpb', entrypoint='GetInt32Tables';
 DATASET(Types.Uint32LDatum) GetUint32Tables(
   STRING arguments,    
   STRING session_id) : c, pure, LIBRARY='eclpb', entrypoint='GetUint32Tables' ;
 DATASET(Types.Int64LDatum) GetInt64Tables(
   STRING arguments,    
   STRING session_id) : c, action, LIBRARY='eclpb', entrypoint='GetInt64Tables';
 Allkn(STRING arguments,
   STRING session_id
   ) : c, action, LIBRARY='eclpb', entrypoint='Allkn'; 
END; 



x:=DATASET(
[{0,0,-0.90265,0},
{0,1,-0.91719,0},
{0,2,0.74029,0},
{1,0,-0.079012,0},
{1,1,-0.79218,0},
{1,2,2.4023,0},
{2,0,-0.79442,0},
{2,1,0.40984,0},
{2,2,-0.65182,0},
{3,0,0.4469,0},
{3,1,0.86663,0},
{3,2,-0.15092,0},
{4,0,-0.52346,0},
{4,1,0.44427,0},
{4,2,0.33556,0}], Types.RealLDatum);



session_id:='hi';//PB.GenSessionId();
PB.MakeWorkSpace(session_id);
OUTPUT(session_id);
arguments:=' --references_in=dense$double$0 '
               +' --k_neighbors=1'
               +' --distances_out=dense$double$3'
               +' --indices_out=dense$uint32$4';
PB.LoadAllTables(x,,,arguments,session_id);
PB.Allkn(arguments, session_id);

distances:=PB.GetRealTables(arguments, session_id);
indices:=PB.GetUint32Tables(arguments, session_id);
pblog:=PB.GetLog();
distances;
