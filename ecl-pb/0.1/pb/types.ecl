IMPORT ML;

DatumMacro(IdType, ValueType) := FUNCTIONMACRO
  RETURN RECORD
    IdType id;
	  ML.Types.t_FieldNumber number;
	  ValueType value;
  END;
ENDMACRO;

RealDatum := DatumMacro(ML.Types.t_RecordID, ML.Types.t_FieldReal);
Uint8Datum := DatumMacro(ML.Types.t_RecordID, UNSIGNED1);
Int64Datum := DatumMacro(ML.Types.t_RecordID, INTEGER);
UInt64Datum := DatumMacro(ML.Types.t_RecordID, UNSIGNED);
Int32Datum := DatumMacro(ML.Types.t_RecordID, INTEGER4);
UInt32Datum := DatumMacro(ML.Types.t_RecordID, UNSIGNED4);


LDatumMacro(DatumType) := FUNCTIONMACRO
  RETURN RECORD
    (DatumType)
    UNSIGNED1 file_id;
  END;
ENDMACRO;

EXPORT Types := MODULE
  EXPORT RealLDatum := LDatumMacro(RealDatum);
  EXPORT Uint8LDatum := LDatumMacro(Uint8Datum);
  EXPORT Int64LDatum := LDatumMacro(Int64Datum);
  EXPORT UInt64LDatum := LDatumMacro(UInt64Datum);
  EXPORT Int32LDatum := LDatumMacro(Int32Datum);
  EXPORT UInt32LDatum := LDatumMacro(UInt32Datum);


END;

