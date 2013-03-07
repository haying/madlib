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

} // namespace convex

} // namespace modules

} // namespace madlib

