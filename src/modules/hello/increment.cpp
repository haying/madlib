/* ----------------------------------------------------------------------- *//**
 *
 * @file increment.cpp
 *
 *//* ----------------------------------------------------------------------- */

#include <dbconnector/dbconnector.hpp>

#include "increment.hpp"

namespace madlib {

// Use Eigen
using namespace dbal::eigen_integration;

namespace modules {

namespace hello {

AnyType
hello_increment::run(AnyType &args) {
    int arg = args[0].getAs<int>();
    arg ++;

    return arg;
}

} // namespace hello

} // namespace modules

} // namespace madlib
