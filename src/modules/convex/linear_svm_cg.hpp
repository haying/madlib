/* ----------------------------------------------------------------------- *//**
 *
 * @file linear_svm_cg.hpp
 *
 *//* ----------------------------------------------------------------------- */

/**
 * @brief Linear support vector machine (conjugate gradient): Transition function
 */
DECLARE_UDF(convex, linear_svm_cg_transition)

/**
 * @brief Linear support vector machine (conjugate gradient): State merge function
 */
DECLARE_UDF(convex, linear_svm_cg_merge)

/**
 * @brief Linear support vector machine (conjugate gradient): Final function
 */
DECLARE_UDF(convex, linear_svm_cg_final)

/**
 * @brief Linear support vector machine (conjugate gradient): Search direction
 */
DECLARE_UDF(convex, linear_svm_cg_direction)
DECLARE_UDF(convex, linear_svm_cg_update)

/**
 * @brief Linear support vector machine (conjugate gradient): Difference in
 *     log-likelihood between two transition states
 */
DECLARE_UDF(convex, internal_linear_svm_cg_distance)

/**
 * @brief Linear support vector machine (conjugate gradient): Convert
 *     transition state to model coefficients
 */
DECLARE_UDF(convex, linear_svm_cg_coef)

