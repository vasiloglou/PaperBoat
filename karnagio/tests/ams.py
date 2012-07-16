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
targets="ams"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "ams", fout)

# Start the tests
for directory in bin_dir.values():
  ams1=directory+"/ams --references_in="+             \
      dataset_dir+"3gaussians/3gaussians.txt "+       \
      "--graphd:allkn:k_neighbors=15 "+               \
      "--kde:bandwidth=0.9 " +                        \
      "--clusters_out=clusters "+                     \
      "--memberships_out=memberships"
  os.system(ams1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, ams1, "SUCCESS"
  else:
    print >> fout, ams1, "FAILED"
    print ams1, "FAILED"
  os.remove("temp")
  if os.path.exists("clusters")==True:
    os.remove("clusters")
  if os.path.exists("memberships")==True:  
    os.remove("memberships")
 
print >> fout, "[ams] Test finished"
fout.close()
t.cancel()
