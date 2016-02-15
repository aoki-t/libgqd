
//#include <float.h>
//#include <stdio.h>
#include <iostream>

#include "kernel.h"


union trans{
	unsigned __int64 asInt64;
	double asDouble;
};



int main() {
	std::cout << "Start GQD tests=================================================" << std::endl;
	std::cout << std::endl;
	Initialize();
	trans inf, qnan;
	inf.asDouble = std::numeric_limits<double>::infinity();
	qnan.asDouble = std::numeric_limits<double>::quiet_NaN();
	std::cout << std::hex;
	std::cout << inf.asInt64 << std::endl;
	std::cout << qnan.asInt64 << std::endl;



	checkInline();


	//addWithCudaq();

	//addWithCuda();


	IEEE_check();

	Finalize();
	std::cout << "Exit GQD tests==================================================" << std::endl;
	return 0;
}

