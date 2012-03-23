  IMPORT PB;  
              
  x:=DATASET(
     [{0,0,1,0},
      {0,1,2,0},
      {0,2,4,0},
      {1,0,3,0},
      {1,1,2,0},
      {1,2,1,0},
      {2,0,2,0},
      {2,1,4,0},
      {2,2,5,0},
      {3,0,3,0},
      {3,1,2,0},
      {3,2,1,0},
      {4,0,3,0},
      {4,1,2,0},
      {4,2,1,0}], PB.Types.RealLDatum);

  PB.Definitions();  // All the paperboat calls start with this initializer
  
  z:=PB.Karnagio(x,,).Nmf(
      ' --references_in=dense$double$0'
      +' --w_factor_out=dense$double$1'
      +' --h_factor_out=dense$double$2'
      +' --k_rank=2');
  z.call;
  OUTPUT(z.real_result);
  PB.UnDefinitions();
