  IMPORT PB;

  PB.Definitions();
  x:=DATASET(
    [{0,0,0.90265,0},
    {0,1,0.91719,0},
    {0,2,-0.74029,0},
    {1,0,0.079012,0},
    {1,1,0.79218,0},
    {1,2,-2.4023,0},
    {2,0,0.79442,0},
    {2,1,-0.40984,0},
    {2,2,0.65182,0},
    {3,0,-0.4469,0},
    {3,1,0.86663,0},
    {3,2,0.15092,0},
    {4,0,0.52346,0},
    {4,1,-0.44427,0},
    {4,2,-0.33556,0}], PB.Types.RealLDatum);

  y:=DATASET(
   [{0,0,-0.90265,1},
    {0,1,-0.91719,1},
    {0,2,0.74029,1},
    {1,0,-0.079012,1},
    {1,1,-0.79218,1},
    {1,2,2.4023,1},
    {2,0,-0.79442,1},
    {2,1,0.40984,1},
    {2,2,-0.65182,1},
    {3,0,0.4469,1},
    {3,1,0.86663,1},
    {3,2,-0.15092,1},
    {4,0,-0.52346,1},
    {4,1,0.44427,1},
    {4,2,0.33556,1}], PB.Types.RealLDatum);
  

  z:=PB.Karnagio(x+y,,).Kde(' --references_in=dense$double$0 '
               +' --bandwidth_selection=monte_carlo'
               +' --kernel=gaussian'
               );

  OUTPUT(z.call);
  OUTPUT(z.real_result);
  OUTPUT(z.uint32_result);
  OUTPUT(z.int32_result);

  PB.UnDefinitions();


