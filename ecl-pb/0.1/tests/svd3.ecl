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
      {4,2,1,0}], PB.Types.Uint8LDatum);

  PB.Definitions();  // All the paperboat calls start with this initializer
  
  z:=PB.Karnagio(,x,).Svd(
      ' --references_in=sparse$uint8$0' // As we have mentioned ECL does not
                                         // discriminate between sparse and 
                                         // dense. You choose how the 
                                         // paperboat will handle them 
      +' --rec_error=0'         // WARNING if your data is high dimensional then
                                // computing reconstruction error is N^2 and very 
                                // costly
      +' --algorithm=randomized' 
      +' --smoothing_p=2'  // if your dataset has a very slow decay of
                           // eigenvalues you need to increase this value
      +' --svd_rank=2'
      +' --lsv_out=dense$double$1'
      +' --rsv_out=dense$double$2'
      +' --sv_out=dense$double$3');
  z.call;
  OUTPUT(z.real_result);

  PB.UnDefinitions();

