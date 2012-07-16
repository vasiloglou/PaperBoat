import os
import os.path
import threading
import test_suite
import optparse

parser = optparse.OptionParser();

parser.add_option("--version_dir", action="store", type="string", 
    dest="version_dir", default="../branches/0.6.4/", 
    help="the version directory to build/use the binaries")
parser.add_option("--logfile", action="store", type="string", 
    dest="logfile", default="test_results",
    help="The file to append the results")
(options, args)=parser.parse_args();

# open log file for appending output
fout=open(options.logfile, "a")

version_dir=options.version_dir
build_mode=["config-debug"]
bin_dir={"debug":(version_dir+"debug.build/bin"), \
    "release":(version_dir+"release.build/bin")}
targets="moe"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "moe", fout)

# Start the tests
for directory in bin_dir.values():
  moe1=directory+"/moe --references_in="+             \
      dataset_dir+"/moe/moe_easy.pb"    +             \
     " --predefined_memberships_in="+"/moe/moe_easy_memberships.pb "+ \
     " --k_clusters=4 "+                              \
     " --expert=regression "+                         \
     " --expert_args=--algorithm:naive,--exclude_bias_term:0,--prediction_index_prefix:1 "+ \
     " --log_expert=0 "+                              \
     " --memberships_out=memberships "+               \
     " --final_expert_args=--coeffs_out:coeffs "+     \
  os.system(moe + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, moe, "SUCCESS"
  else:
    print >> fout, moe, "FAILED"
    print moe, "FAILED"
  os.remove("temp")
  if os.path.exists("memberships0")==True:
    os.remove("memberships*")
  if os.path.exists("coeffs0")==True:  
    os.remove("coeffs*")
 
print >> fout, "[moe] Test finished"
fout.close()
t.cancel()

