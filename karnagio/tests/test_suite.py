import os
import os.path

# this routine will check and see if there was any terminate message in 
# the log, Also it will check to see if the outputs were generated
# if they were empty. Also it will evaluate the performance of the 
# algorithm if there is any performance bug. 
def EveluateRun(logfile, outfiles):
  pass

def EvaluateBuild(logfile):
  pass

def ExpireTest(test_name, fhandle):
  print >>fhandle, test_name, "Failed to terminate within reasonable time"
  fhandle.close()
  exit()

def EvaluateBuild(logfile):
  return True

def EvaluateRun(logfile, outfiles, expected_bytes):
  fin=open(logfile, "r")
  log=fin.read()
  fin.close()
  if log.find("[TERMINATING]")!=-1:
    return False
  if len(expected_bytes)==0:
      expected_bytes=[1000 for i in range(len(outfiles))]
  for i in range(len(outfiles)):
    info = os.stat(outfiles[i])
    if info.st_size<expected_bytes[i]:
      return False
  return True

def Build(version_dir, test_dir, targets, fout):
  configure="cd "+version_dir+" && make config-debug 2>&1 >cmake_log"
  if (os.path.exists(version_dir+"/debug.build")==False):
    os.system(configure)
  else:
    os.system("cd "+version_dir+"/debug.build && cmake . 2>&1 >cmake_log")

  build="cd "+version_dir+"debug.build && make -j3 "+targets
  os.system(build + " 2>&1  > temp")
  if EvaluateBuild(version_dir+"debug.build/temp")==True:
    print  >> fout, "[",targets, "] debug build SUCCESS"
  else:
    print >> fout, "[", targets, "] debug build FAIL"
  os.remove(version_dir+"debug.build/temp")

  os.system("cd "+test_dir)
  configure="cd "+version_dir+" && make config-release 2>&1 >cmake_log"
  if (os.path.exists(version_dir+"/release.build")==False):
    os.system(configure)
  else:
    os.system("cd "+version_dir+"/release.build && cmake . 2>&1 >cmake_log")

  build="cd "+version_dir+"release.build && make -j3 "+targets
  os.system(build + " 2>&1 > temp")
  if EvaluateBuild(version_dir+"release.build/temp")==True:
    print >> fout, "[",targets,"] released build SUCCESS"
  else:
    print >> fout, "[",targets,"] release build FAIL"
  os.remove(version_dir+"release.build/temp")
  os.system("cd "+test_dir)


