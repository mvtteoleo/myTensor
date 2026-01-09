// Compile with
//  g++ t1.cpp -std=c++23 -O3
#include "include/tensors.hpp"
#include <cassert>
#include <iostream>

int main (int argc, char *argv[]) { 
    numPDE::Vector<double> a(3, 0.0);
    numPDE::Vector<double> b(3, 3.0);
    numPDE::Vector<double> c(3, 5.0);
      
    numPDE::Array<double, 3> d(8.0);
    /*
    numPDE::Array<double, 3>   e(5.0);  
    numPDE::Array<double, 3>   f(5.0); 
    */
    numPDE::ElementProxy<double, 3, false>   e( b );  
    numPDE::ElementProxy<double, 3, false>   f( d ); 
    
    numPDE::Vector<double> g (  a + b + c + d + e - 2.0*f);

    assert(g[0] ==   a[0] + b[0] + c[0] + d[0] + e[0] - 2.0*f[0] );
    for(const auto v : g){ std::cout << v ; }

    return 0;
}
