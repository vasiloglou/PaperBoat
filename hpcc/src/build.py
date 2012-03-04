import os

command="eclcc pb2ecl_test1.ecl " \
        +"-Wc,-I../karnagio/include,-I../karnagio/src " \
        +"-v "\
        +"-flinkOptions=-L../karnagio/debug.build/lib/," \
        +"-lfastlib-dbg," \
        +"-lboost_thread-mt,"\
        +"-lboost_program_options-mt,"\
        +"-llapack,"\
        +"-lblas "
os.system(command)
#os.system("/usr/bin/g++ a.out.cpp "\
#          +"-fvisibility=hidden -DUSE_VISIBILITY=1 -fPIC -pipe -O0 " \
#          +"-I../karnagio/include " \
#          +"-I../karnagio/src " \
#          +"-m64  -c  -I/opt/HPCCSystems/componentfiles/cl/include")
#os.system("/usr/bin/g++ "\
#          +"-L. -Wl,-E -fPIC -pipe -O0  "\
#          +"-L/opt/HPCCSystems/lib -Wl,-rpath "
#          +"-Wl,/opt/HPCCSystems/lib  "\
#          +"a.out.o "\
#          +"-leclrtl -la.out.res.o -lhthor "\
#          +"-L../karnagio/debug.build/lib "\
#          +"-lfastlib-dbg "\
#          +"-lboost_thread-mt "\
#          +"-lboost_program_options-mt "\
#          +"-llapack " \
#          +"-lblas " \
#          +"-o a.out")
