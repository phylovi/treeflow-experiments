#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Probability & Treeflow Demo
# Author: Christiaan Swanepoel

import click
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.beagle
import treeflow.substitution_model
import treeflow.sequences
import treeflow.tree_processing
import treeflow.model
from treeflow.coalescent import ConstantCoalescent

tfd = tfp.distributions

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

DATA_ROOT = "../../data/"
NEWICK_FILE = DATA_ROOT + "wnv/wnv_seed_6.nwk"
FASTA_FILE = DATA_ROOT + "wnv/wnv.fasta"
FREQUENCIES = np.array(
    [
        0.26744051135162256,
        0.22286688874964067,
        0.2787013207712062,
        0.23099127912752943,
    ], dtype=treeflow.DEFAULT_FLOAT_DTYPE_NP
)
KAPPA = 14.52346114599242

cast = lambda x: tf.convert_to_tensor(x, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)
def build_q_and_log_posterior(use_libsbn):
    tree, taxon_names = treeflow.tree_processing.parse_newick(NEWICK_FILE)
    topology = treeflow.tree_processing.update_topology_dict(tree["topology"])
    taxon_count = len(taxon_names)
    sampling_times = tf.convert_to_tensor(
        tree["heights"][:taxon_count], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF
    )

    prior = tfd.JointDistributionNamed(
        dict(
            clock_rate=tfd.LogNormal(cast(0.0), cast(3.0)),
            pop_size=tfd.LogNormal(cast(0.0), cast(3.0)),
            tree=lambda pop_size: ConstantCoalescent(
                taxon_count, pop_size, sampling_times
            ),
        )
    )

    q_prior = treeflow.model.construct_prior_approximation(prior)
    q_tree = treeflow.model.construct_tree_approximation(NEWICK_FILE)
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
        alignment = treeflow.sequences.get_encoded_sequences(
            FASTA_FILE, taxon_names
        )

        likelihood, instance = treeflow.sequences.log_prob_conditioned_branch_only(
            alignment,
            topology,
            category_count=1,
            subst_model=subst_model,
            category_weights=tf.convert_to_tensor([1.0], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
            category_rates=tf.convert_to_tensor([1.0], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
            frequencies=FREQUENCIES,
            kappa=KAPPA,
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

q, log_posterior = build_q_and_log_posterior(USE_LIBSBN)

if USE_TF_PROFILER:
    tf.profiler.experimental.start("logdir")

with click.progressbar(range(TRIAL_COUNT), label="Trials") as bar:
    for _ in bar:
        q_tmp = q.copy()
        loss = tfp.vi.fit_surrogate_posterior(
            log_posterior, q_tmp, tf.optimizers.Adam(learning_rate=0.0001), 5
        )

if USE_TF_PROFILER:
    tf.profiler.experimental.stop()

pd.Series(loss).to_csv("loss.csv", header=False, index=False)
