# libgqd: QD library for Windows.

## INTRODUCTION
QD library has written by Yozo Hida, Xiaoye S. Li and David H. Bailey and others in LBNL.  
GQD was developed by Mian Lu and others based on QD.  
libgqd is ported for Windows based on GQD.  

## What is different?
*Class-rized.  
*Switchable between Accuracy-priority and Speed-priority (compile time)  
*Some bugs fixed.  

##Road map
*IEEE754-2008 compliant (expects precision, some limitations and restrictions)  
*Switchable between accuracy priority and Speed priority (in compile time)  

##To Do
*Write tests  


## Known Issues
libgqd project set as static library(.lib), but Nvidia not provide to create user-defined library as Drop-in library.
(According to sample, it must to use through the function pointers currently. It is not my preference.)
So it can not use as static library such as other host some.  
To use this library, include in source code directly such as test project please.
I did not find solution currently.

## License
```
3-clause BSD license.
```

## Extarnal Links
```
Mian Lu's repository:lumianph (https://github.com/lumianph/gpuprec)
Mian Lu's old page for GQD: (https://code.google.com/p/gpuprec/)
LBNL High-Precision Software Directory (http://crd-legacy.lbl.gov/~dhbailey/mpdist/)
```



