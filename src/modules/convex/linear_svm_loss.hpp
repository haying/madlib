/* ----------------------------------------------------------------------- *//**
 *
 * @file linear_svm_loss.hpp
 *
 *//* ----------------------------------------------------------------------- */

/**
 * @brief Linear support vector machine: Loss function
 */
DECLARE_UDF(convex, linear_svm_loss)

/**
 * @brief Linear support vector machine: Prediction
 */
DECLARE_UDF(convex, linear_svm_predict)

DECLARE_UDF(convex, linear_svm_best_ball_transition)
DECLARE_UDF(convex, linear_svm_best_ball_merge)
DECLARE_UDF(convex, linear_svm_best_ball_final)

DECLARE_UDF(convex, linear_svm_greedy_step_size)

