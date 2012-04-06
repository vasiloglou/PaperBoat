import os
import os.path
import threading
import test_suite
import optparse

parser = optparse.OptionParser();

parser.add_option("--version_dir", action="store", type="string", 
    dest="version_dir", default="../branches/0.6.3/", 
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
targets="npr"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "npr", fout)

# Start the tests
for directory in bin_dir.values():
  npr1=directory+"/npr --references_in="+                     \
      dataset_dir+"/3gaussians/3gaussians_regression "+       \
      " --queries_in="+dataset_dir+"/3gaussians/3gaussians_regression_small " \
      +" --bandwidths_in="+dataset_dir+"/3gaussians/bandwidths" \
      +" --relative_error=0.1"                                  \
      +" --predictions_out=predictions"                         \
      +" --reliabilities_out=reliabilities"                     \
      +" --run_mode=eval";
  os.system(npr1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, npr1, "SUCCESS"
  else:
    print >> fout, npr1, "FAILED"
    print npr1, "FAILED"

  os.remove("temp")
  if os.path.exists("reliabilities")==True:
    os.remove("reliabilities")
  else:
    print >>fout, npr1, "(reliabilities) FAILED"
  if os.path.exists("predictions")==True:
    os.remove("predictions")
  else:
    print >> fout, npr1, "(predictions) FAILED"


