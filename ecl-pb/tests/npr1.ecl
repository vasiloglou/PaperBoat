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
      [{0, 0, 0.31, 1},
       {1, 0, -0.4, 1},
       {2, 0, 0.24, 1},
       {3, 0, -0.1, 1},
       {4, 0, -0.9, 1}
      ], PB.Types.RealLDatum);

  PB.Definitions();  // All the paperboat calls start with this initializer
  
  z:=PB.Karnagio(x+y,,).Npr(
    ' --references_in=dense$double$0'
    +' --targets_in=dense$double$1'
    +' --train_algorithm=stoc'
    +' --bandwidths_init=-1'
    +' --relative_error=0.1'
    +' --ref_split_factor=0.8'
    +' --query_split_factor=0.1'
  );
  z.call;
  OUTPUT(z.real_result);
  

  PB.UnDefinitions();


