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

  PB.Definitions();  // All the paperboat calls start with this initializer
  
  z:=PB.Karnagio(x,,).Svd(
      ' --references_in=sparse$double$0' // As we have mentioned ECL does not
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


