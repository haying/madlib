/* ----------------------------------------------------------------------- *//**
 *
 * @file linear_svm_loss.cpp
 *
 * @brief Linear Support Vector Machine functions
 *
 *//* ----------------------------------------------------------------------- */

#include <dbconnector/dbconnector.hpp>

#include "linear_svm_loss.hpp"

#include "task/linear_svm.hpp"

#include "type/tuple.hpp"
#include "type/state.hpp"

namespace madlib {

namespace modules {

namespace convex {

/**
 * @brief Compute the linear support vector machine loss
 */
AnyType
linear_svm_loss::run(AnyType &args) {
    using madlib::dbal::eigen_integration::MappedColumnVector;

    MappedColumnVector model = args[0].getAs<MappedColumnVector>();

    GLMTuple tuple;
    tuple.indVar.rebind(args[1].getAs<MappedColumnVector>().memoryHandle());
    tuple.depVar = args[2].getAs<bool>() ? 1. : -1.;

    return LinearSVM<MappedColumnVector, GLMTuple >::loss(model, tuple.indVar, tuple.depVar);
}

/**
 * @brief Return the prediction reselt
 */
AnyType
linear_svm_predict::run(AnyType &args) {
    using madlib::dbal::eigen_integration::MappedColumnVector;
    MappedColumnVector model = args[0].getAs<MappedColumnVector>();
    MappedColumnVector indVar = args[1].getAs<MappedColumnVector>();

    double p = LinearSVM<MappedColumnVector, GLMTuple>::predict(model, indVar);

    return (p > 0.);
}

/**
 * @brief Solve best ball of the linear support vector machine loss using batching
 */
AnyType
linear_svm_best_ball_transition::run(AnyType &args) {
    using madlib::dbal::eigen_integration::MappedColumnVector;
    using madlib::dbal::eigen_integration::ColumnVector;

    GLMBestBallState<MutableArrayHandle<double> > state = args[0];

    GLMTuple tuple;
    tuple.indVar.rebind(args[1].getAs<MappedColumnVector>().memoryHandle());
    tuple.depVar = args[2].getAs<bool>() ? 1. : -1.;

    MappedColumnVector model = args[3].getAs<MappedColumnVector>();
    MappedColumnVector direction = args[4].getAs<MappedColumnVector>();
    MappedColumnVector stepsizes = args[5].getAs<MappedColumnVector>();

    if (state.numRows == 0) {
        state.allocate(*this, stepsizes.size());
        state.reset();
    }

    uint32_t i;
    for (i = 0; i < state.dimension; i ++) {
        ColumnVector model_to_try = model + stepsizes(i) * direction;
        double l = LinearSVM<ColumnVector, GLMTuple >::loss(model_to_try, tuple.indVar, tuple.depVar);
        state.loss_list(i) += l;
    }

    state.numRows ++;

    return state;
}

AnyType
linear_svm_best_ball_merge::run(AnyType &args) {
    GLMBestBallState<MutableArrayHandle<double> > stateLeft = args[0];
    GLMBestBallState<ArrayHandle<double> > stateRight = args[1];

    if (stateLeft.numRows == 0) { return stateRight; }
    else if (stateRight.numRows == 0) { return stateLeft; }

    stateLeft.loss_list += stateRight.loss_list;
    stateLeft.numRows += stateRight.numRows;

    return stateLeft;
}

AnyType
linear_svm_best_ball_final::run(AnyType &args) {
    GLMBestBallState<ArrayHandle<double> > state = args[0];

    return state.loss_list;
}

AnyType
linear_svm_greedy_step_size::run(AnyType &args) {
    MappedColumnVector loss_list = args[0].getAs<MappedColumnVector>();
    MappedColumnVector stepsizes = args[1].getAs<MappedColumnVector>();

    uint32_t i, mini = 0;
    double min = loss_list(0);
    for (i = 1; i < stepsizes.size(); i ++) {
        if (min > loss_list(i)) {
            min = loss_list(i);
            mini = i;
        }
    }

    return stepsizes(mini);
}

} // namespace convex

} // namespace modules

} // namespace madlib

