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
hello_array_len::run(AnyType &args) {
    MappedColumnVector arg = args[0].getAs<MappedColumnVector>();
    MappedColumnVector to_be_mapped;
    to_be_mapped.rebind(arg.memoryHandle());

    return to_be_mapped.size();
}

} // namespace hello

} // namespace modules

} // namespace madlib
