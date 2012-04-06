import os

exec_path="../../../debug.build/bin/"
executable=exec_path+"moe"
command=executable
command+=" --references_in=/home/nvasil/datasets/moe_test/2experts_500_500_4 "
command+=" --k_clusters=2 "
command+=" --iterations=45 "
command+=" --n_restarts=3 "
command+=" --memberships_out=/home/nvasil/memberships "
command+=" --scores_out=/home/nvasil/scores "
command+=" --expert=regression:1 "
command+=" --expert_args=--prediction_index_prefix:0,--algorithm:naive,--exclude_bias_term:1"
print command
os.system(command);

exec_path="../../../release.build/bin/"
executable=exec_path+"moe"
command=executable
command+=" --references_in=/home/nvasil/datasets/moe_test/3experts_500_500_500_4 "
command+=" --k_clusters=3 "
command+=" --iterations=15 "
command+=" --memberships_out=/home/nvasil/memberships "
command+=" --scores_out=/home/nvasil/scores "
command+=" --expert=regression:1 "
command+=" --expert_args=--prediction_index_prefix:0,--algorithm:naive,--exclude_bias_term:1"
print command
os.system(command);
