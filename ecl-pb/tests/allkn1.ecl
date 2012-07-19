IMPORT PB;


PB.Definitions();

//DATASET(PB.Types.RealLDatum) xx := DATASET('3gaussians.thor', PB.Types.RealLDatum, CSV(SEPARATOR(','))); x:=xx(id<=10);
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


/*
x:=DATASET([{0, 0, 0.1, 4},
            {0, 1, 0.2, 4},
            {1, 0, 1.1, 4},
            {1, 1, 1.2, 4},
            {2, 0, 2.1, 4},
            {2, 1, 2.2, 4},
            {3, 0, 3.1, 4},
            {3, 1, 3.2, 4},
            {4, 0, 4.1, 4},
            {4, 1, 4.2, 4}], PB.Types.RealLDatum);
*/

z:=PB.Karnagio(x,,).Allkn(' --references_in=dense$double$0 '
               +' --k_neighbors=1'
               +' --distances_out=dense$double$3'
               +' --indices_out=dense$uint32$4');

OUTPUT(z.real_result);
OUTPUT(z.uint32_result);

PB.UnDefinitions();
