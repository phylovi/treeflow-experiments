#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import treeflow
import treeflow.beagle
import treeflow.substitution_model
import treeflow.sequences
import treeflow.tree_processing
import treeflow.model
import numdifftools as nd
import time

tfd = tfp.distributions

DATA_ROOT = "../../data/"
NEWICK_FILE = DATA_ROOT + "wnv/wnv_seed_6.nwk"
FASTA_FILE = DATA_ROOT + "wnv/wnv.fasta"
FREQUENCIES = np.array(
    [
        0.26744051135162256,
        0.22286688874964067,
        0.2787013207712062,
        0.23099127912752943,
    ]
)
KAPPA = 14.52346114599242
RELAXED_SD = 1.0


def cast_float(x):
    return tf.convert_to_tensor(x, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)


def build_likelihood_and_gradient(use_libsbn):
    tree, taxon_names = treeflow.tree_processing.parse_newick(NEWICK_FILE)
    branch_lengths = treeflow.sequences.get_branch_lengths(tree)

    if use_libsbn:
        likelihood, instance = treeflow.beagle.log_prob_conditioned_branch_only(
            FASTA_FILE,
            treeflow.substitution_model.HKY(),
            frequencies=FREQUENCIES,
            kappa=KAPPA,
            rescaling=True,
            newick_file=NEWICK_FILE,
        )
    else:
        alignment = treeflow.sequences.get_encoded_sequences(FASTA_FILE, taxon_names)
        topology = tree["topology"]

        likelihood, instance = treeflow.sequences.log_prob_conditioned_branch_only(
            alignment,
            topology,
            category_count=1,
            subst_model=treeflow.substitution_model.HKY(),
            category_weights=cast_float([1.0]),
            category_rates=cast_float([1.0]),
            frequencies=FREQUENCIES,
            kappa=KAPPA,
        )
    likelihood = tf.function(likelihood)
    likelihood(branch_lengths)  # Call to do function mode tracing

    @tf.function
    def clock_likelihood_func(clock_rate):
        return likelihood(branch_lengths * clock_rate)

    @tf.function
    def strict_clock_func(log_clock_rate):
        return tfp.math.value_and_gradient(
            lambda log_clock_rate: -1.0
            * likelihood(branch_lengths * tf.exp(log_clock_rate)),
            log_clock_rate,
        )

    @tf.function
    def relaxed_clock_func(all_log_clock_rates):
        num_branch_lengths = branch_lengths.get_shape()[0]
        # assert all_log_clock_rates.shape == num_branch_lengths + 1

        def function(all_log_clock_rates):
            grand_log_rate, log_clock_rates = tf.split(
                all_log_clock_rates, [1, num_branch_lengths]
            )
            return (
                -1.0
                * likelihood(branch_lengths * tf.exp(grand_log_rate + log_clock_rates))
                + 0.5
                * tf.reduce_sum(log_clock_rates * log_clock_rates)
                / RELAXED_SD
                / RELAXED_SD
            )

        return tfp.math.value_and_gradient(function, all_log_clock_rates)

    return (
        tree,
        strict_clock_func,
        relaxed_clock_func,
        clock_likelihood_func,
        likelihood,
        instance,
    )


(
    tree,
    strict_clock,
    relaxed_clock,
    cl,
    log_likelihood,
    inst,
) = build_likelihood_and_gradient(use_libsbn=True)

(
    tree2,
    strict_clock2,
    relaxed_clock2,
    cl2,
    log_likelihood2,
    inst2,
) = build_likelihood_and_gradient(use_libsbn=False)

test_clock_rate_value = cast_float(0.0064)
print(cl(test_clock_rate_value))
print(cl2(test_clock_rate_value))

strict_start = tf.constant([0.1], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)
strict_result = tfp.optimizer.bfgs_minimize(
    strict_clock, initial_position=strict_start, tolerance=1e-8
)
relaxed_start = tf.constant([0.0], shape=207, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)
relaxed_result = tfp.optimizer.lbfgs_minimize(
    relaxed_clock, initial_position=relaxed_start, tolerance=1e-8, max_iterations=1000
)


def numeric_gradient(log_clock_rate):
    dfun = nd.Gradient(
        lambda log_clock_rate: -1.0 * cl(np.exp(log_clock_rate)).numpy()
    )  # only first 2 dimensions
    return dfun(log_clock_rate)


print(numeric_gradient(-1.0))
print(strict_clock(cast_float(-1.0))[1])

print(strict_result.position)
print(relaxed_result.position)


def st_time(func):
    def st_func(*args, **keyArgs):
        t1 = time.time()
        r = func(*args, **keyArgs)
        t2 = time.time()
        print("Function=%s, Time=%s" % (func.__name__, t2 - t1))
        return r

    return st_func


@st_time
def loop_lbfgs_libsbn():
    for x in range(5):
        tfp.optimizer.lbfgs_minimize(
            relaxed_clock,
            initial_position=relaxed_start,
            tolerance=1e-8,
            max_iterations=1000,
        )


# @st_time
# def loop_lbfgs_tf():
#     for x in range(5):
#         tfp.optimizer.lbfgs_minimize(relaxed_clock2, initial_position=relaxed_start,
#                                      tolerance=1e-8, max_iterations=1000)


@st_time
def loop_like_libsbn():
    for x in range(5):
        log_likelihood(treeflow.sequences.get_branch_lengths(tree))


@st_time
def loop_like_tf():
    for x in range(5):
        log_likelihood2(treeflow.sequences.get_branch_lengths(tree))


loop_lbfgs_libsbn()
loop_like_libsbn()

loop_like_tf()  # Does not return correct log_likelihood
# loop_lbfgs_tf()
