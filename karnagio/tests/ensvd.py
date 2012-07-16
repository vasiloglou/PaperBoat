import os
import os.path
import threading
import test_suite
import optparse

parser = optparse.OptionParser();

parser.add_option("--version_dir", action="store", type="string", 
    dest="version_dir", default="../branches/0.6.3/", 
    help="the version directory to build/use the binaries")
parser.add_option("--dataset_dir", action="store", type="string",
    dest="dataset_dir", default="./datasets", 
    help="The directory that contains the datasets")
parser.add_option("--logfile", action="store", type="string", 
    dest="logfile", default="test_results",
    help="The file to append the results")
(options, args)=parser.parse_args();

# open log file for appending output
fout=open(options.logfile, "a")

version_dir=options.version_dir
build_mode=["config-debug"]
bin_dir={"debug":(version_dir+"debug.build/bin")}
targets="ensvd"
dataset_dir=options.dataset_dir;
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the watchdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "ensvd", fout)

# Start the tests
for directory in bin_dir.values():
  ensvd1=directory+"/ensvd --references_prefix_in="+\
      dataset_dir+"/3gaussians/3gaussians_chunk   "+\
      " --references_num_in=2"+                     \
      " --svd:svd_rank=3"+                          \
      " --svd:algorithm=covariance"+                \
      " --lsv_prefix_out=lsv "+                     \
      " --lsv_num_out=2 "+                          \
      " --sv_out=sv "+                              \
      " --rsv_prefix_out=rsv "+                     \
      " --rsv_num_out=2 "

  print ensvd1
  os.system(ensvd1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, ensvd1, "SUCCESS"
  else:
    print >> fout, ensvd1, "FAILED"
  #os.remove("temp")
  if os.path.exists("sv")==True:
    os.remove("sv")
  if os.path.exists("rsv0")==True:  
    os.remove("rsv*")
  if os.path.exists("lsv0")==True:  
    os.remove("lsv*")
  
print >> fout, "[ensvd] Test finished"
fout.close()
t.cancel()

