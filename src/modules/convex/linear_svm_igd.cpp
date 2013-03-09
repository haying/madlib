/* ----------------------------------------------------------------------- *//**
 *
 * @file linear_svm_igd.cpp
 *
 * @brief Linear Support Vector Machine functions
 *
 *//* ----------------------------------------------------------------------- */

#include <dbconnector/dbconnector.hpp>

#include "linear_svm_igd.hpp"

#include "task/linear_svm.hpp"
#include "algo/igd.hpp"
#include "algo/loss.hpp"

#include "type/tuple.hpp"
#include "type/model.hpp"
#include "type/state.hpp"

namespace madlib {

namespace modules {

namespace convex {

// This 2 classes contain public static methods that can be called
typedef IGD<GLMIGDState<MutableArrayHandle<double> >, GLMIGDState<ArrayHandle<double> >,
        LinearSVM<GLMModel, GLMTuple > > LinearSVMIGDAlgorithm;

typedef Loss<GLMIGDState<MutableArrayHandle<double> >, GLMIGDState<ArrayHandle<double> >,
        LinearSVM<GLMModel, GLMTuple > > LinearSVMLossAlgorithm;

/**
 * @brief Perform the linear support vector machine transition step
 *
 * Called for each tuple.
 */
AnyType
linear_svm_igd_transition::run(AnyType &args) {
    // The real state.
    // For the first tuple: args[0] is nothing more than a marker that
    // indicates that we should do some initial operations.
    // For other tuples: args[0] holds the computation state until last tuple
    GLMIGDState<MutableArrayHandle<double> > state = args[0];

    // initilize the state if first tuple
    if (state.algo.numRows == 0) {
        if (!args[3].isNull()) {
            GLMIGDState<ArrayHandle<double> > previousState = args[3];
            state.allocate(*this, previousState.task.dimension);
            state = previousState;
            double stepsize = args[5].getAs<double>();
            state.task.stepsize = stepsize;
        } else {
            // configuration parameters
            uint32_t dimension = args[4].getAs<uint32_t>();
            double stepsize = args[5].getAs<double>();

            state.allocate(*this, dimension); // with zeros
            state.task.stepsize = stepsize;
        }
        // resetting in either case
        state.reset();
    }

    // tuple
    using madlib::dbal::eigen_integration::MappedColumnVector;
    GLMTuple tuple;
    tuple.indVar.rebind(args[1].getAs<MappedColumnVector>().memoryHandle());
    tuple.depVar = args[2].getAs<bool>() ? 1. : -1.;

    // Now do the transition step
    LinearSVMIGDAlgorithm::transition(state, tuple);
    LinearSVMLossAlgorithm::transition(state, tuple);
    state.algo.numRows ++;

    return state;
}

/**
 * @brief Perform the perliminary aggregation function: Merge transition states
 */
AnyType
linear_svm_igd_merge::run(AnyType &args) {
    GLMIGDState<MutableArrayHandle<double> > stateLeft = args[0];
    GLMIGDState<ArrayHandle<double> > stateRight = args[1];

    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (stateLeft.algo.numRows == 0) { return stateRight; }
    else if (stateRight.algo.numRows == 0) { return stateLeft; }

    // Merge states together
    LinearSVMIGDAlgorithm::merge(stateLeft, stateRight);
    LinearSVMLossAlgorithm::merge(stateLeft, stateRight);
    // The following numRows update, cannot be put above, because the model
    // averaging depends on their original values
    stateLeft.algo.numRows += stateRight.algo.numRows;

    return stateLeft;
}

/**
 * @brief Perform the linear support vector machine final step
 */
AnyType
linear_svm_igd_final::run(AnyType &args) {
    // We request a mutable object. Depending on the backend, this might perform
    // a deep copy.
    GLMIGDState<MutableArrayHandle<double> > state = args[0];

    // Aggregates that haven't seen any data just return Null.
    if (state.algo.numRows == 0) { return Null(); }

    // finalizing
    LinearSVMIGDAlgorithm::final(state);
    // LinearSVMLossAlgorithm::final(state); // empty function call causes a warning

    // debug code
    // dberr << "loss: " << state.algo.loss << std::endl;

    return state;
}

AnyType
internal_linear_svm_igd_result::run(AnyType &args) {
    GLMIGDState<ArrayHandle<double> > state = args[0];

    AnyType tuple;
    tuple << state.task.model
        << static_cast<double>(state.algo.loss);

    return tuple;
}

/**
 * @brief Return the difference in RMSE between two states
 */
AnyType
internal_linear_svm_igd_distance::run(AnyType &args) {
    GLMIGDState<ArrayHandle<double> > stateLeft = args[0];
    GLMIGDState<ArrayHandle<double> > stateRight = args[1];

    return std::abs((stateLeft.algo.loss - stateRight.algo.loss)
            / stateRight.algo.loss);
}

/**
 * @brief Return the coefficients of the state
 */
AnyType
internal_linear_svm_igd_coef::run(AnyType &args) {
    GLMIGDState<ArrayHandle<double> > state = args[0];

    return state.task.model;
}

// This 2 classes contain public static methods that can be called
typedef IGD<GLMIGDBBState, GLMIGDBBState, LinearSVM<GLMModel, GLMTuple > > LinearSVMIGDBBAlgorithm;

// typedef Loss<GLMIGDBBState, GLMIGDBBState, LinearSVM<GLMModel, GLMTuple > > LinearSVMLossBBAlgorithm;

AnyType
linear_svm_igd_bb_transition::run(AnyType &args) {
    MappedColumnVector stepsizes = args[5].getAs<MappedColumnVector>();
    MutableArrayHandle<double> storage = args[0].getAs<MutableArrayHandle<double> >();
    std::vector<GLMIGDBBState> states;
    uint32_t dimension = args[4].getAs<uint32_t>();
    uint32_t arraySize = GLMIGDBBState::arraySize(dimension) + 1;

    if (storage[0] == 0) {
        if (!args[3].isNull()) {
            MappedColumnVector previousState = args[3].getAs<MappedColumnVector>();
            storage = this->allocateArray<double, dbal::AggregateContext, dbal::DoZero,
                    dbal::ThrowBadAlloc>(arraySize * stepsizes.size());
            for (uint32_t i = 0; i < stepsizes.size(); i ++) {
                states.push_back(GLMIGDBBState(&storage[i * arraySize], dimension));
                states[i].task.model = previousState;
                states[i].task.stepsize = stepsizes(i);
                states[i].algo.incrModel = previousState;
            }
        } else {
            // configuration parameters
            storage = this->allocateArray<double, dbal::AggregateContext, dbal::DoZero,
                    dbal::ThrowBadAlloc>(arraySize * stepsizes.size());
            for (uint32_t i = 0; i < stepsizes.size(); i ++) {
                states.push_back(GLMIGDBBState(&storage[i * arraySize], dimension));
                states[i].task.dimension = dimension;
                states[i].task.stepsize = stepsizes(i);
            }
        }
    } else {
        for (uint32_t i = 0; i < stepsizes.size(); i ++) {
            states.push_back(GLMIGDBBState(&storage[i * arraySize], dimension));
        }
    }

    // tuple
    using madlib::dbal::eigen_integration::MappedColumnVector;
    GLMTuple tuple;
    tuple.indVar.rebind(args[1].getAs<MappedColumnVector>().memoryHandle());
    tuple.depVar = args[2].getAs<bool>() ? 1. : -1.;

    for (uint32_t i = 0; i < stepsizes.size(); i ++) {
        // Now do the transition step
        LinearSVMIGDBBAlgorithm::transition(states[i], tuple);
        states[i].algo.numRows ++;
    }

    return storage;
}

AnyType
linear_svm_igd_bb_final::run(AnyType &args) {
    MutableArrayHandle<double> storage = args[0].getAs<MutableArrayHandle<double> >();
    std::vector<GLMIGDBBState> states;
    uint32_t numOfStepsizes = 0;

    // Aggregates that haven't seen any data just return Null.
    if (storage[0] == 0) { return Null(); }

    HandleTraits<ArrayHandle<double> >::ReferenceToUInt32 dimension;
    dimension.rebind(&storage[0]);
    uint32_t arraySize = GLMIGDBBState::arraySize(dimension) + 1;
    numOfStepsizes = storage.size() / arraySize;

    uint32_t loss_arraySize = GLMLossBBState::arraySize(dimension) + 1;
    MutableArrayHandle<double> loss_storage = this->allocateArray<double,
            dbal::AggregateContext, dbal::DoZero, dbal::ThrowBadAlloc>(
            loss_arraySize * numOfStepsizes);
    std::vector<GLMLossBBState> loss_states;

    for (uint32_t i = 0; i < numOfStepsizes; i ++) {
        states.push_back(GLMIGDBBState(&storage[i * arraySize], dimension));
        // finalizing
        LinearSVMIGDBBAlgorithm::final(states[i]);
        // prepare for loss best ball
        loss_states.push_back(GLMLossBBState(&loss_storage[i * loss_arraySize], dimension));
        loss_states[i].task.model = states[i].task.model;
        loss_states[i].task.stepsize = states[i].task.stepsize;
    }

    return loss_storage;
}

AnyType
linear_svm_igd_min_transition::run(AnyType &args) {
    GLMIGDState<ArrayHandle<double> > state = args[0];

    if (state.algo.numRows == 0) {
        ArrayHandle<double> next = args[1].getAs<ArrayHandle<double> >();
        MutableArrayHandle<double> ret = this->allocateArray<double, dbal::AggregateContext,
                dbal::DoZero, dbal::ThrowBadAlloc>(next.size());
        for (size_t i = 0; i < next.size(); i ++) {
            ret[i] = next[i];
        }
        return ret;
    }

    GLMIGDState<ArrayHandle<double> > nextState = args[1];

    if (state.algo.loss > nextState.algo.loss) { return args[1]; }
    else { return args[0]; }
}

} // namespace convex

} // namespace modules

} // namespace madlib

