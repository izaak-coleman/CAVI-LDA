"""main file for LDA inference."""
import sys
import pandas as pd
import numpy as np
from scipy.special import digamma, gammaln

from parse_wc import parse_to_docs, filter_docs

# start from same seed to aid debug
np.random.seed(42)

# Number of topcs
K = 20
# value at which we assume convergence
CONVERGED = 1


def safe_ln(x, minval=10**(-200)):
    return np.log(x.clip(lower=minval))


def view_topics(vocab, lambdas, n_terms=10):
    for k in range(K):
        top = lambdas.loc[k].argsort(kind='stable')[-n_terms:] # get top n_terms
        top = lambdas.loc[k].iloc[top].index
        terms = '; '.join(vocab.loc[top][0])
        print(f'Topic {k}: {terms}')


def init_lambdas(n_topics, vocab_sz, word_counts, prior=None):
    # init to prior.
    if prior:
        return pd.DataFrame(np.full((n_topics, vocab_sz), prior), columns=word_counts.columns)
    # else, randomly init
    lambdas = np.full((n_topics, vocab_sz), 0.0)
    for i in range(n_topics):
        for j in range(vocab_sz):
            lambdas[i,j] = np.random.uniform(0.0000001, 1)
    return pd.DataFrame(lambdas, columns=word_counts.columns)


def init_gammas(n_docs, n_topics, prior=np.empty((0,))):
    # init to 1 if no prior, otherwise, init to prior
    gammas = np.full((n_docs, n_topics), 1.0)
    if len(prior) != 0:
        for k in range(n_topics):
            gammas[:,k] = prior[k]
    return pd.DataFrame(gammas)


def elbo(word_counts, lambdas, gammas, E_q_log_beta, E_q_log_theta, eta, alphas):
    """Compute the mean field ELBO"""
    # set up
    n_docs, vocab_sz = word_counts.shape
    JOINT = 0.0
    NEG_ENTROPY = 0.0

    # compute expected joint term E_q[log p(beta)]
    JOINT += ((eta - 1) * E_q_log_beta).sum(axis=1).sum()
    # compute expected joint term E_q[log p(theta)]
    JOINT += ((alphas - 1) * E_q_log_theta).sum(axis=1).sum()
    # compute:
    # 1) expected joint term E_q[log p(z_ij | theta_i)],
    # 2) expected joint term E_q[log p(x_ij | z_ij, beta)]
    # 3) negative entropy term E_q[log z_ij | phi_ij]
    for i in range(n_docs):
        # get phi_i matrix
        phi_i = np.exp(E_q_log_beta.add(E_q_log_theta.iloc[i], axis=0))
        phi_i = phi_i / phi_i.sum(axis=0) # normalize
        # compute 1) E_q[log p(z_ij | theta_i)]
        JOINT += phi_i.mul(E_q_log_theta.iloc[i], axis=0).mul(word_counts.iloc[i], axis=1).sum().sum()
        # compute 2) E_q[log p(x_ij | z_ij, beta)]
        JOINT += (phi_i*E_q_log_beta).mul(word_counts.iloc[i], axis=1).sum().sum()
        # compute 3) negative entropy term E_q[log z_ij | phi_ij]
        NEG_ENTROPY += (phi_i*safe_ln(phi_i)).mul(word_counts.iloc[i], axis=1).sum().sum()

    # compute -E_q[log q(theta_i; gamma_i)]
    NEG_ENTROPY += (
            -gammaln(gammas.sum(axis=1)) + (gammaln(gammas)).sum(axis=1) - ((gammas - 1) * E_q_log_theta).sum(axis=1)
    ).sum()
    # compute -E_q[log q(beta_i; lambda_i)]
    NEG_ENTROPY += (
            -gammaln(lambdas.sum(axis=1)) + (gammaln(lambdas)).sum(axis=1) - ((lambdas - 1) * E_q_log_beta).sum(axis=1)
    ).sum()

    return JOINT - NEG_ENTROPY


def run_CAVI(word_counts, eta, alphas, vocab, view_topic_it=False):

    # set up the run
    n_docs, vocab_sz = word_counts.shape
    elbos = list()
    # initialize variational parameter matrices
    lambdas = init_lambdas(K, vocab_sz, word_counts)
    gammas = init_gammas(n_docs, K)
    # dummy elbo
    prev_elbo, new_elbo = -np.inf, 0
    iteration = 0

    # iterate
    while abs(prev_elbo - new_elbo) > CONVERGED:
        print(f'CAVI iteration {iteration}...')

        # store last iters elbo
        prev_elbo = new_elbo
        if view_topic_it:
            view_topics(vocab, lambdas)

        # compute E_q[log beta_ik] and E_q [log theta_ik]
        E_q_log_beta = digamma(lambdas).sub(digamma(lambdas.sum(axis=1)), axis=0)
        E_q_log_theta = digamma(gammas).sub(digamma(gammas.sum(axis=1)), axis=0)
        # get the ELBO
        new_elbo = elbo(word_counts, lambdas, gammas, E_q_log_beta, E_q_log_theta, eta, alphas)
        elbos.append(new_elbo)
        print(f'Iteration {iteration} ELBO: {elbos[-1]}  ELBO DIFF: {new_elbo - prev_elbo}')

        # Update the variational params in the following loop.
        # Here, I compute the summations gamma_ik = alpha + E[m_ik] on the fly.
        # This is also true for the lambda_kv = eta + E[n_kv].
        # This avoids the need to store a big phi_ikv matrix.
        # Also, use of word counts avoids storing an even bigger phi_ikj matrix (j being the jth word in the doc)

        # Initialize new variational params to their respective priors (eta, alphas)
        lambdas = init_lambdas(K,vocab_sz, word_counts, prior=eta)
        gammas = init_gammas(n_docs, K, prior=alphas)
        for i in range(n_docs):
            # compute matrix phi_i (a k X v matrix encoding phi_ikv)
            phi_i = np.exp(E_q_log_beta.add(E_q_log_theta.iloc[i], axis=0))
            phi_i = phi_i / phi_i.sum(axis=0)  # normalize
            # scale according to word counts
            phi_i_scaled = phi_i.mul(word_counts.iloc[i], axis=1)
            # Add the phis to the lambdas adding the ith docs contribution to E[n_kv]
            lambdas += phi_i_scaled
            # Add the phis to the gamma_i to get E[m_ik]
            gammas.iloc[i] += phi_i_scaled.sum(axis=1)

        iteration += 1
        # Finished iteration

    # convergence reached.
    return lambdas, gammas, pd.DataFrame(elbos, columns=['ELBO'])


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage: <exe> <ap.dat> <vocab> <run_name> <eta> <alpha>')
        sys.exit()

    ap_dat, vocab, run_name, eta, alpha = sys.argv[1:]
    eta = float(eta)
    alpha = float(alpha)

    # read in the vocabulary into vocab vector
    vocab = pd.read_csv(vocab, header=None)

    # Read the ap dat file into pandas dataframe that is Docs X Vocab
    # each entry ij is the number of times vocab term j occurs in doc i
    word_counts = parse_to_docs(ap_dat)

    # filter words. Remove stopwords and infrequent words,
    # and then drop those from the matrix.
    # Make sure the indexes of the matrix still match the correct
    # index in the vocab vector after dropping.
    word_counts = filter_docs(word_counts)

    # specify alphas
    alphas = np.full((K,), alpha)

    with open(f'{run_name}.log', 'w') as f:
        f.write(f'ETA {eta}\n')
        f.write(f"ALPHAS {','.join([str(a) for a in alphas])}\n")
        f.write(f"K {K}\n")

    # Launch CAVI
    lambdas, gammas, elbos = run_CAVI(word_counts, eta, alphas, vocab, view_topic_it=True)

    # Write data
    lambdas.index = [f'topic_{i}' for i in lambdas.index]
    lambdas.columns = [f'term_{i}' for i in lambdas.columns]
    lambdas.to_csv(f'{run_name}.lambdas.csv')
    gammas.index = [f'doc_{i}' for i in gammas.index]
    gammas.columns = [f'topic_{i}' for i in gammas.columns]
    gammas.to_csv(f'{run_name}.gammas.csv')
    elbos.index  = [f'Iteration_{i}' for i in elbos.index]
    elbos.to_csv(f'{run_name}.elbos.csv')

