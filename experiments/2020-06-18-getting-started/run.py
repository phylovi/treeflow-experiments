#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Probability & Treeflow Demo
# Author: Christiaan Swanepoel

import click
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import treeflow
import treeflow.beagle
import treeflow.substitution_model
import treeflow.sequences
import treeflow.tree_processing
import treeflow.model
import treeflow.vi
from treeflow.coalescent import ConstantCoalescent

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
    ],
    dtype=np.float32,
)
KAPPA = 14.52346114599242


def cast_float(x):
    return tf.convert_to_tensor(x, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)


def build_q_and_log_posterior(use_libsbn):
    tree, taxon_names = treeflow.tree_processing.parse_newick(NEWICK_FILE)
    topology = tree["topology"]
    taxon_count = len(taxon_names)
    sampling_times = tf.convert_to_tensor(
        tree["heights"][:taxon_count], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF
    )

    prior = tfd.JointDistributionNamed(
        dict(
            clock_rate=tfd.LogNormal(loc=cast_float(0.0), scale=cast_float(3.0)),
            pop_size=tfd.LogNormal(loc=cast_float(0.0), scale=cast_float(3.0)),
            tree=lambda pop_size: ConstantCoalescent(
                taxon_count, pop_size, sampling_times
            ),
        )
    )
    prior_sample = prior.sample()

    q_prior, q_prior_vars = treeflow.model.construct_prior_approximation(
        prior, prior_sample
    )
    q_tree, q_tree_vars = treeflow.model.construct_tree_approximation(NEWICK_FILE)
    q = tfd.JointDistributionNamed(dict(tree=q_tree, **q_prior))
    q.sample()

    subst_model = treeflow.substitution_model.HKY()
    if use_libsbn:
        likelihood, instance = treeflow.beagle.log_prob_conditioned_branch_only(
            FASTA_FILE,
            subst_model,
            frequencies=FREQUENCIES,
            kappa=KAPPA,
            rescaling=True,
            newick_file=NEWICK_FILE,
        )
    else:
        tree, taxon_names = treeflow.tree_processing.parse_newick(NEWICK_FILE)
        alignment = treeflow.sequences.get_encoded_sequences(FASTA_FILE, taxon_names)

        likelihood, instance = treeflow.sequences.log_prob_conditioned_branch_only(
            alignment,
            topology,
            category_count=1,
            subst_model=subst_model,
            category_weights=cast_float([1.0]),
            category_rates=cast_float([1.0]),
            frequencies=cast_float(FREQUENCIES),
            kappa=cast_float(KAPPA),
        )

    wrapped_likelihood = lambda z: likelihood(
        treeflow.sequences.get_branch_lengths(z["tree"])
        * tf.expand_dims(z["clock_rate"], -1)
    )
    wrapped_likelihood(q.sample())

    log_posterior = lambda **z: (prior.log_prob(z) + wrapped_likelihood(z))
    log_posterior(**q.sample())

    return q, log_posterior


TRIAL_COUNT = 1
USE_TF_PROFILER = False
USE_LIBSBN = False
USE_FUNCTION_MODE = True

q, log_posterior = build_q_and_log_posterior(USE_LIBSBN)

if USE_TF_PROFILER:
    tf.profiler.experimental.start("logdir")

with click.progressbar(range(TRIAL_COUNT), label="Trials") as bar:
    for _ in bar:
        q_tmp = q.copy()
        optimizer = tf.optimizers.Adam(learning_rate=0.0001)
        num_steps = 5
        if USE_FUNCTION_MODE:
            loss = tfp.vi.fit_surrogate_posterior(
                log_posterior, q_tmp, optimizer, num_steps
            )
        else:
            loss = treeflow.vi.fit_surrogate_posterior(
                log_posterior, q_tmp, optimizer, num_steps
            )

if USE_TF_PROFILER:
    tf.profiler.experimental.stop()

pd.Series(loss).to_csv("loss.csv", header=False, index=False)
