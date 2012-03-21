IMPORT PB;

PB.Definitions();

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

z:=PB.Karnagio(x,,).Kde(' --references_in=dense$double$0 '
             +' --kernel=gaussian'
             +' --bandwidth=1'
             +' --iterations=2'
             +' --relative_error=0.01'
             +' --algorithm=dual'
             +' --densities_out=dense$double$1');

OUTPUT(z.call);
OUTPUT(z.real_result);

PB.UnDefinitions();


