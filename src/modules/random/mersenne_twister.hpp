/* ----------------------------------------------------------------------- *//**
 *
 * @file mersenne_twister.cpp
 *
 *//* ----------------------------------------------------------------------- */

#ifndef MADLIB_MODULES_RANDOM_MERSENNE_TWISTER_HPP
#define MADLIB_MODULES_RANDOM_MERSENNE_TWISTER_HPP

#include <boost/tr1/random.hpp>

namespace std {
    // Import names from TR1.
    
    // Currently provided by boost.
    using tr1::mt19937;
}

namespace madlib {

namespace modules {

namespace random {

std::mt19937& mersenneTwister();

} // namespace random

} // namespace modules

} // namespace madlib

#endif // defined(MADLIB_MODULES_RANDOM_MERSENNE_TWISTER_HPP)
