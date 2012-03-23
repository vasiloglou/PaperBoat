  IMPORT PB;  

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
      {4,2,0.33556,0}], PB.Types.RealLDatum);
  
   y:=DATASET(
     [{0,0,-0.265,1},
      {0,1,-0.719,1},
      {0,2,0.7029,1},
      {1,0,-0.012,1},
      {1,1,-0.218,1},
      {1,2,2.4013,1},
      {2,0,-0.442,1},
      {2,1,0.1984,1},
      {2,2,-0.182,1},
      {3,0,0.4469,1},
      {3,1,0.6663,1},
      {3,2,-0.092,1},
      {4,0,-0.346,1},
      {4,1,0.4427,1},
      {4,2,0.3556,1}], PB.Types.RealLDatum);

   w:=DATASET(
      [{0, 0, 0.31, 2},
       {1, 0, -0.4, 2},
       {2, 0, 0.24, 2},
       {3, 0, -0.1, 2},
       {4, 0, -0.9, 2}
      ], PB.Types.RealLDatum);
   b:=DATASET(
      [{0,0,1.265,3},
      {1,0,1.719,3},
      {2,0,1.7029,3}], PB.Types.RealLDatum);


  PB.Definitions();  // All the paperboat calls start with this initializer
  
  z:=PB.Karnagio(x+y+w+b,,).Npr(
    ' --references_in=dense$double$0'
    +' --targets_in=dense$double$2'
    +' --queries_in=dense$double$1'
    +' --bandwidths_in=dense$double$3'
    +' --relative_error=0.1'
    +' --predictions_out=dense$double$4'
    +' --reliabilities_out=dense$double$5'
    +' --run_mode=eval'
  );
  z.call;
  OUTPUT(z.real_result);
  

  PB.UnDefinitions();


