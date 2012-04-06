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
targets="svd"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "svd", fout)

# Start the tests
for directory in bin_dir.values():
  svd1=directory+"/svd --references_in="+           \
      dataset_dir+"/random/random_1kx6.txt "+       \
      " --algorithm=covariance"                     \
      +" --svd_rank=2"                              \
      +" --rec_error=1"                             \
      +" --lsv_out=lsv"                             \
      +" --rsv_out=rsv"                             \
      +" --sv_out=sv"                             
  os.system(svd1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, svd1, "SUCCESS"
  else:
    print >> fout, svd1, "FAILED"
    print svd1, "FAILED"
  os.remove("temp")
  if os.path.exists("lsv"):
    os.remove("lsv")
  else:
    print >> fout, svd1, "(lsv) FAILED"
  if os.path.exists("rsv"):
    os.remove("rsv")
  else:
    print >> fout, svd1, "(rsv) FAILED"
  if os.path.exists("sv"):
    os.remove("sv")
  else:
    print >> fout, svd1, "(sv) FAILED"
 
  svd2=directory+"/svd --references_in="+            \
       dataset_dir+"/random/random_1kx6.txt "+       \
       " --algorithm=randomized"                     \
       +" --svd_rank=2"                              \
       +" --smoothing_p=2"                           \
       +" --rec_error=0"                             \
       +" --lsv_out=lsv"                             \
       +" --rsv_out=rsv"                             \
       +" --sv_out=sv"                             
  os.system(svd2 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, svd2, "SUCCESS"
  else:
    print >> fout, svd2, "FAILED"
    print svd2, "FAILED"
  if os.path.exists("lsv"):
    os.remove("lsv")
  else:
    print >> fout, svd1, "(lsv) FAILED"
  if os.path.exists("rsv"):
    os.remove("rsv")
  else:
    print >> fout, svd1, "(rsv) FAILED"
  if os.path.exists("sv"):
    os.remove("sv")
  else:
    print >> fout, svd1, "(sv) FAILED"
  
