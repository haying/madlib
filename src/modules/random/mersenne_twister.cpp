/* ----------------------------------------------------------------------- *//**
 *
 * @file PseudoRandomNumberGenerator_impl.cpp
 *
 * @brief Wrap Boost's Mersenne Twister PRNG
 *
 *//* ----------------------------------------------------------------------- */

#include <boost/tr1/random.hpp>

namespace std {
    // Import names from TR1.
    
    // Currently provided by boost.
    using tr1::random_device;
}

#include "mersenne_twister.hpp"

namespace madlib {

namespace modules {

namespace random {

std::mt19937&
mersenneTwister() {
    static bool sInitialized = false;
    static std::mt19937 sMersenneTwister;
    
    if (!sInitialized) {
        // We assume that seeds generated with random_device are completely
        // uncorrelated and thus reasonable choice even for multiple
        // pseudo-random number streams that may be greated with different seeds
        // (e.g., in different processes or on different machines).
        
        std::random_device rd;
        sMersenneTwister.seed(rd());
        sInitialized = true;
    }
    return sMersenneTwister;
}

} // namespace random

} // namespace modules

} // namespace madlib
