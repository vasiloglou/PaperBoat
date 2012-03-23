INTEGER F2(INTEGER x) := BEGINC++
   #include <iostream>
   int k=0;
   #body

   return x;
 ENDC++;

F2(0);

 INTEGER F3(INTEGER x) := BEGINC++
   #include <iostream>
   #body

   return x;
 ENDC++;


MyModule := MODULE
  EXPORT MyModule1(INTEGER x) := MODULE
     EXPORT INTEGER F1(INTEGER x) := BEGINC++
       #include <iostream>
       #body
       k++;
       std::cout<<"***"<<k<<std::endl;
       return x;
     ENDC++;
     EXPORT INTEGER FF1(INTEGER x) := FUNCTION
       return x;
     END;
     EXPORT INTEGER FF2(INTEGER x) := FUNCTION
       return x;
     END;
     EXPORT INTEGER FF3(INTEGER x) := FUNCTION
       return x;
     END;
    
    EXPORT INTEGER a:=F1(x);
    EXPORT INTEGER b:=F2(a);
    EXPORT INTEGER c:=F3(a);
  END;
END;

z:=MyModule.MyModule1(1);
OUTPUT(z.a);
OUTPUT(z.b);
OUTPUT(z.c);
