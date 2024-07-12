import networkx as nx
import numpy as np
import scipy
import torch
import torch.distributions
import torch_geometric
from scipy.sparse import csr_matrix

# from scipy.special import gamma, gammaln

from torch_geometric.data import Data
from torch.special import gammaln


from modules.transforms.liftings.graph2simplicial.clique_lifting import (
    SimplicialCliqueLifting,
)


class LatentCliqueCover:
    """Latent clique cover model for network data corresponding to the
    Partial Observability Setting of the Random Clique Cover paper:
    http://proceedings.mlr.press/v115/williamson20a/williamson20a.pdf

    """

    def __init__(self, data: Data, init="edges"):
        self.data = data
        self.init = init

        # prior parameters
        self.alpha_params = [1.0, 1.0]
        self.sigma_params = [1.0, 1.0]
        self.pie_params = [1.0, 1.0]

        self._init_params()
        self._init_Z()
        self.adj = self.get_adj()

    @property
    def num_nodes(self):
        return self.data.num_nodes

    @property
    def num_edges(self):
        return self.data.edge_index.shape[1]

    @property
    def device(self):
        return self.data.edge_index.device

    def get_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=self.device)
        edges = self.data.edge_index.T
        adj[edges[:, 0], edges[:, 1]] = 1
        adj[edges[:, 1], edges[:, 0]] = 1
        # for i in range(self.num_nodes):
        #     adj[i, i] = 1
        return adj

    @property
    def edges(self):
        _edges = self.data.edge_index.T
        N = self.data.num_nodes
        #  add self loops
        nodes = torch.arange(N, device=self.device)
        _edges = torch.cat([_edges, torch.stack([nodes, nodes], 1)])
        return _edges

    def _init_params(self):
        self.alpha = 1
        self.sigma = 0.5
        self.c = 1
        self.lamb = self.num_edges
        self.pie = 0.9

    def _init_Z(self):
        num_edges = self.data.edge_index.shape[1]
        num_nodes = self.data.num_nodes
        edges = self.data.edge_index.T

        # initialize Z to one single
        if self.init == "edges":
            # self.Z = np.zeros((self.num_edges, self.num_nodes))
            self.Z = torch.zeros((num_edges, num_nodes), device=self.device)
            for i in range(self.num_edges):
                self.Z[i, edges[i][0]] = 1
                self.Z[i, edges[i][1]] = 1
        else:
            raise ValueError("init not recognized")

    def num_cliques(self):
        return self.Z.shape[0]

    # @torch.compile # < not supported in python 3.11 yet
    @torch.no_grad()
    def sample(
        self,
        num_iters=1000,
        num_sm=10,
        dot_every=100,
        sample_hypers=True,
        do_gibbs=True,
        verbose=False,
    ):
        # Gibbs seems to matter here
        for iter in range(num_iters):
            if do_gibbs:
                self.gibbs()
                if verbose and iter % dot_every == 0:
                    print("iter ", iter, ", gibbs done.")

            if sample_hypers:
                self.sample_hypers()
                if verbose and iter % dot_every == 0:
                    print("iter ", iter, ", sample_hypers done.")

            for _ in range(num_sm):
                self.splitmerge()
                if verbose and iter % dot_every == 0:
                    print("iter ", iter, ", splitmerge done.")

            if iter % dot_every == 0:
                print("iter ", iter, ", K=", self.num_cliques())

    def membership_counts(self):
        """Returns the number of nodes in each clique."""
        return self.Z.sum(1)

    def loglik(self, alpha=None, sigma=None, c=None, alpha_only=False, include_K=False):
        """Efficient implementation of the Stable Beta-Indian Buffet Process likelihood

        The likelihood is computed as

        P(Z1,...,Zn) = alpha * exp( - alpha * A * B) * C * D^K
                   A = sum_i=1^n Gam(i - 1 + c + sigma) / Gam(i + c)
                   B = Gam(1 + c) / Gam(c + sigma)
                   C = prod_k=1^K Gam(mk - sigma) * Gam(n - mk + c + sigma)
                   D = Gam(1 + c) / Gam(c + sigma) / Gam(1 - sigma) / Gam(n + c)

        Or, equivalently

        logP(Z1,...,Zn) = log(alpha) - alpha * A * B + logC + K * logD
                   A = as before
                   B = as before
                logC = sum_k=1^K log(Gam(mk - sigma)) + log(Gam(n - mk + c + sigma)
                logD = log(Gam(1 + c)) - log(Gam(c + sigma)) - log(Gam(1 - sigma)) - log(Gam(n + c))

        See Eq. 10 and Teh and Gorur (2010), "Indian Buffet Processes with Power-Law Behavior,
            Advances in Neural Information Processing Systems 23. for details.

        Additionally, we can compute the probability of the number of cliques K as
            P(K) = Poisson(lamb) where lam can be ignored
        """
        alpha = alpha if alpha is not None else self.alpha
        sigma = sigma if sigma is not None else self.sigma
        c = c if c is not None else self.c

        # num nodes and num cliques
        N = self.num_nodes
        K = self.num_cliques()

        # compute A
        ivec = 1 + torch.arange(N, dtype=torch.float, device=self.device)
        A_terms = gammaln(ivec - 1 + c + sigma) - gammaln(ivec + c)
        A = A_terms.clamp(-20, 20).exp().sum()

        # compute B
        C = scipy.special.gammaln(1 + c) - scipy.special.gammaln(c + sigma)

        # compute first part of likelihood involving alpha
        ll = N * np.log(alpha) - alpha * A * C
        if alpha_only:
            return ll

        # compute logC
        mk = self.membership_counts()
        logC = gammaln(mk - sigma).sum() + gammaln(N - mk + c + sigma).sum()

        # compute logD
        logD = (
            scipy.special.gammaln(1 + c)
            - scipy.special.gammaln(c + sigma)
            - scipy.special.gammaln(1 - sigma)
            - scipy.special.gammaln(N + c)
        )

        # compute the rest of the likelihood
        ll = ll + logC + K * logD

        if include_K:
            ll = ll + torch.distributions.poisson.logpmf(K, self.lamb)

        return ll

    def sample_hypers(self, step_size=0.01):
        # same as full, but with sampling pie added in
        # mk = np.sum(self.Z,0)
        alpha_prop = self.alpha + step_size * np.random.randn()
        if alpha_prop > 0:
            lp_ratio = (self.alpha_params[0] - 1) * (
                np.log(alpha_prop) - np.log(self.alpha)
            ) + self.alpha_params[1] * (self.alpha - alpha_prop)

            ll_new = self.loglik(alpha=alpha_prop, alpha_only=True)
            ll_old = self.loglik(alpha_only=True)
            lratio = ll_new - ll_old + lp_ratio
            r = np.log(np.random.rand())
            if r < lratio:
                self.alpha = alpha_prop
        sigma_prop = self.sigma + step_size * np.random.randn()
        if sigma_prop > 0:
            if sigma_prop < 1:
                ll_new = self.loglik(sigma=sigma_prop)
                ll_old = self.loglik()

                lp_ratio = (self.sigma_params[0] - 1) * (
                    np.log(sigma_prop) - np.log(self.sigma)
                ) + (self.sigma_params[1] - 1) * (
                    np.log(1 - sigma_prop) - np.log(1 - self.sigma)
                )
                lratio = ll_new - ll_old + lp_ratio
                r = np.log(np.random.rand())

                if r < lratio:
                    self.sigma = sigma_prop

        pie_prop = self.pie + step_size * np.random.randn()
        if pie_prop > 0:
            if pie_prop < 1:
                ll_new = self.loglikZ(pie=pie_prop)
                ll_old = self.loglikZ()
                lp_ratio = (self.pie_params[0] - 1) * (
                    np.log(pie_prop) - np.log(self.pie)
                ) + (self.pie_params[1] - 1) * (
                    np.log(1 - pie_prop) - np.log(1 - self.pie)
                )
                lratio = ll_new - ll_old + lp_ratio
                r = np.log(np.random.rand())
                if r < lratio:
                    self.pie = pie_prop

    def gibbs(self):
        mk = self.membership_counts()
        K = self.num_cliques()

        for node in range(self.num_nodes):
            for clique in range(K):
                if self.Z[clique, node] == 1:
                    self.Z[clique, node] = 0
                    ll_0 = self.loglikZn(node)
                    self.Z[clique, node] = 1
                    if not np.isinf(ll_0):

                        ll_1 = self.loglikZn(node)
                        mk[node] -= 1
                        if mk[node] == 0:
                            continue
                            raise ValueError("empty clique")

                        prior0 = (K - mk[node]) / (K - self.sigma)
                        prior1 = 1 - prior0

                        lp0 = np.log(prior0) + ll_0
                        lp1 = np.log(prior1) + ll_1
                        lp0 = lp0 - torch.logsumexp(torch.tensor([lp0, lp1]), dim=0)
                        r = torch.log(torch.rand(1))
                        if r < lp0:
                            self.Z[clique, node] = 0
                        else:
                            mk[node] += 1
                else:
                    self.Z[clique, node] = 1
                    ll_1 = self.loglikZn(node)
                    self.Z[clique, node] = 0
                    if not np.isinf(ll_1):
                        ll_0 = self.loglikZn(node)

                        prior0 = (K - mk[node]) / (K - self.sigma)
                        prior1 = 1 - prior0

                        lp0 = np.log(prior0) + ll_0
                        lp1 = np.log(prior1) + ll_1
                        lp1 = lp1 - torch.logsumexp(torch.tensor([lp0, lp1]), dim=0)
                        r = torch.log(torch.rand(1))
                        if r < lp1:
                            self.Z[clique, node] = 1
                            mk[node] += 1

    def loglikZ(self, Z=None, pie=None):
        if Z is None:
            Z = self.Z
        if pie is None:
            pie = self.pie

        cic = Z.T @ Z
        cic = cic - torch.diag(torch.diag(cic))

        # check whether cic is ever zero, when network is 1
        zero_check = (1 - cic.clamp(max=1)) * self.adj
        if torch.sum(zero_check) == 0:
            p0 = (1 - pie) ** cic
            p1 = 1 - p0
            netmask = self.adj + 1
            netmask = torch.triu(netmask, 1) - 1
            lp = torch.where(netmask == 0, 0.01 + p0, 0.01 + p1).log().sum()
        else:
            lp = -float("inf")

        return lp

    def loglikZn(self, node, Z=None):
        if Z is None:
            Z = self.Z

        cic = Z.T[node] @ Z
        cic[node] = 0

        # check whether cic is ever zero, cicwhen network is 1
        zero_check = (1 - cic.clamp(max=1)) * self.adj[node, :]
        if torch.sum(zero_check) == 0:
            p0 = (1 - self.pie) ** cic
            p1 = 1 - p0
            lp = torch.where(self.adj[node, :] == 0, 0.01 + p0, 0.01 + p1).log().sum()
            # lp = torch.sum(torch.log(p0[adj[node, :] == 0])) + torch.sum(
            #     torch.log(p1[adj[node, :] == 1])
            # )
        else:
            lp = -float("inf")

        return lp

    def splitmerge(self):
        # pick an edge
        link_id = torch.randint(self.num_edges, (1,)).item()
        r = torch.rand(1).item()
        if r < 0.5:
            sender = self.edges[link_id][0]
            receiver = self.edges[link_id][1]
        else:
            sender = self.edges[link_id][1]
            receiver = self.edges[link_id][0]

        # pick the first clique
        K = self.num_cliques()
        valid_cliques = torch.nonzero(self.Z[:, sender])
        clique_i = valid_cliques[torch.randint(len(valid_cliques), (1,)).item()]
        valid_cliques = torch.nonzero(self.Z[:, receiver])
        clique_j = valid_cliques[torch.randint(len(valid_cliques), (1,)).item()]

        clique_i = int(clique_i)
        clique_j = int(clique_j)

        if clique_i == clique_j:
            # propose split
            Z_prop = self.Z.clone()
            Z_prop = torch.cat(
                (
                    Z_prop[:clique_i],
                    Z_prop[clique_i + 1 :],
                    torch.zeros(2, self.num_nodes),
                )
            )

            lqsplit = 0
            lpsplit = 0

            mk = self.membership_counts()
            for node in range(self.num_nodes):  # np.random.permutation(self.num_nodes):
                if self.Z[clique_i, node] == 1:
                    if node == sender:
                        # must be 11 or 10
                        Z_prop[K - 1, node] = 1

                        r = np.random.rand()
                        if r < 0.5:
                            Z_prop[K, node] = 1
                            # mk is one bigger, and K is one bigger, p1
                            lpsplit = (
                                lpsplit
                                + np.log(mk[node] + 1 - self.sigma)
                                - np.log(K + 1 - self.sigma)
                            )
                        else:
                            # mk is one bigger, and K is one bigger, p0
                            lpsplit = (
                                lpsplit
                                + np.log(K + 1 - mk[node] - 1)
                                - np.log(K + 1 - self.sigma)
                            )
                        lqsplit -= np.log(2)

                    elif node == receiver:
                        # must be 11 or 01
                        Z_prop[K, node] = 1
                        r = np.random.rand()
                        if r < 0.5:
                            Z_prop[K - 1, node] = 1
                            lpsplit = (
                                lpsplit
                                + np.log(mk[node] + 1 - self.sigma)
                                - np.log(K + 1 - self.sigma)
                            )
                        else:
                            lpsplit = (
                                lpsplit
                                + np.log(K - mk[node])
                                - np.log(K + 1 - self.sigma)
                            )
                        lqsplit -= np.log(2)
                    else:
                        r = np.random.rand()
                        if r < (1 / 3):
                            Z_prop[K - 1, node] = 1
                            # mk is one bigger, and K is one bigger, p0
                            lpsplit = (
                                lpsplit
                                + np.log(K - mk[node])
                                - np.log(K + 1 - self.sigma)
                            )
                        elif r < (2 / 3):
                            Z_prop[K, node] = 1
                            lpsplit = (
                                lpsplit
                                + np.log(K - mk[node])
                                - np.log(K + 1 - self.sigma)
                            )
                        else:
                            Z_prop[K - 1, node] = 1
                            Z_prop[K, node] = 1
                            # mk is one bigger, and K is one bigger, p1
                            lpsplit = (
                                lpsplit
                                + np.log(mk[node] + 1 - self.sigma)
                                - np.log(K + 1 - self.sigma)
                            )
                        lqsplit -= np.log(3)
                else:
                    # mk is the same and K is one bigger, p0
                    lpsplit = (
                        lpsplit + np.log(K + 1 - mk[node]) - np.log(K + 1 - self.sigma)
                    )

            # is the resulting proposal valid?

            ll_prop = self.loglikZ(Z_prop)
            if not np.isinf(ll_prop):
                ll_old = self.loglikZ()
                # then calculate the acceptance prob
                lqsplit = (
                    lqsplit
                    - np.log(np.sum(self.Z[:, sender]))
                    - np.log(np.sum(self.Z[:, receiver]))
                )
                # lqsplit =-np.log(np.sum(self.Z[:,sender]))-np.log(np.sum(self.Z[:,receiver]))
                lqmerge = -np.log(
                    np.sum(self.Z[:, sender])
                    - self.Z[clique_i, sender]
                    + np.sum(Z_prop[:, sender])
                ) - np.log(
                    np.sum(self.Z[:, receiver])
                    - self.Z[clique_i, receiver]
                    + np.sum(Z_prop[:, receiver])
                )

                lpsplit += np.log(self.lamb / (K + 1))
                laccept = lpsplit - lqsplit + lqmerge + ll_prop - ll_old
                r = np.log(np.random.rand())

                if r < laccept:
                    # pdb.set_trace()
                    # self.checksums
                    self.Z = Z_prop  # + 0
                    K += 1
                # self.checksums()

        else:
            # propose merge
            Z_sum = self.Z[clique_i, :] + self.Z[clique_j, :]
            Z_prop = self.Z.clone()
            Z_prop[clique_i] = np.minimum(Z_sum, 1)
            Z_prop = np.delete(Z_prop, clique_j, 0)
            ll_prop = self.loglikZ(Z_prop)
            if not np.isinf(ll_prop):
                # merge OK, proceed
                mk = self.membership_counts()
                # calculate the backward probability
                num_affected = Z_prop.sum()
                if num_affected < 2:
                    raise ValueError("num_affected<2")
                # lqsplit = -2*np.log(2) - (num_affected-2)*np.log(3)
                # OK now the merge probability
                lqmerge = -np.log(self.Z[:, sender].sum()) - np.log(
                    self.Z[:, receiver].sum()
                )

                # lqsplit = lqsplit -np.log(np.sum(self.Z[:,sender])-self.Z[clique_i,sender]-self.Z[clique_j,sender]+1) - np.log(np.sum(self.Z[:,receiver])-self.Z[clique_i,receiver]-self.Z[clique_j,receiver]+1)
                lqsplit = -np.log(
                    self.Z[:, sender].sum()
                    - self.Z[clique_i, sender]
                    - self.Z[clique_j, sender]
                    + 1
                ) - np.log(
                    self.Z[:, receiver].sum()
                    - self.Z[clique_i, receiver]
                    - self.Z[clique_j, receiver]
                    + 1
                )
                # lqsplit +=num_opt*np.log(0.5)

                lpsplit = 0
                for node in range(self.num_nodes):
                    if Z_sum[node] == 0:
                        # mk is the same, and K the same, p0
                        lpsplit = (
                            lpsplit + np.log(K - mk[node]) - np.log(K - self.sigma)
                        )
                    elif Z_sum[node] == 1:
                        # mk is plus one, and K the same, p0
                        lpsplit = (
                            lpsplit + np.log(K - mk[node] - 1) - np.log(K - self.sigma)
                        )
                    else:
                        # mk is plus one, and K the same, p2
                        lpsplit = (
                            lpsplit
                            + np.log(mk[node] + 1 - self.sigma)
                            - np.log(K - self.sigma)
                        )

                lpmerge = np.log(K / self.lamb)
                ll_old = self.loglikZ()

                laccept = lpmerge - lpsplit + lqsplit - lqmerge + ll_prop - ll_old
                r = np.log(np.random.rand())

                if r < laccept:
                    self.Z = Z_prop  # + 0
                    K -= 1


class LatentCliqueCoverLifting(SimplicialCliqueLifting):
    r"""Lifts graphs to cell complexes by identifying the cycles as 2-cells.

    Parameters
    ----------
    max_cell_length : int, optional
        The maximum length of the cycles to be lifted. Default is None.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, pie: float = 0.8, it=100, warmup_it=10, init="edges", **kwargs):
        super().__init__(**kwargs)
        self.pie = pie
        self.it = it
        self.warmup_it = warmup_it
        self.init = init

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Finds the cycles of a graph and lifts them to 2-cells.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        # net = torch_geometric.utils.to_dense_adj(data.edge_index)[0].numpy()
        # for i in range(net.shape[0]):
        #     net[i, i] = 1
        # edges = data.edge_index.numpy().T

        # mod = LatentCliqueCoverSampler(net, edges, pie=self.pie, init=self.init)
        mod = LatentCliqueCover(data, init=self.init)
        mod.sample(sample_hypers=False, num_iters=self.warmup_it, dot_every=1)
        mod.sample(sample_hypers=True, num_iters=self.it, dot_every=1)

        cic = mod.Z.T @ mod.Z
        cic = cic - torch.diag(torch.diag(cic))
        cic = cic.clamp(max=1).cpu().numpy()
        g = nx.from_numpy_matrix(adj)
        edges = torch.LongTensor(list(g.edges()), device=data.edge_index.device).T
        edges = torch.cat([edges, edges.flip(0)], dim=1)
        new_data = torch_geometric.data.Data(x=data.x, edge_index=edges)
        return super().lift_topology(new_data)


def poissonparams(K, alpha, sigma, c):
    # vectorised for speed
    ivec = np.arange(K, dtype=float)
    lpp = (
        np.log(alpha)
        + scipy.special.gammaln(1.0 + c)
        - scipy.special.gammaln(c + sigma)
        + scipy.special.gammaln(ivec + c + sigma)
        - scipy.special.gammaln(ivec + 1.0 + c)
    )

    pp = np.exp(lpp)

    return pp


def sample_from_ibp(K, alpha, sigma, c):
    """
    samples from the random clique cover model using the three parameter ibp
    params
        K: number of random cliques
        alpha, sigma, c: ibp parameters
    returns
        a sparse matrix, compressed by rows, representing the clique membership matrix
        recover the adjacency matrix with min(Z'Z, 1)
    """
    pp = poissonparams(K, alpha, sigma, c)
    # ivec = np.arange(K, dtype=float)
    # lpp = (
    #     np.log(alpha)
    #     + gammaln(1.0 + c)
    #     - gammaln(c + sigma)
    #     + gammaln(ivec + c + sigma)
    #     - gammaln(ivec + 1.0 + c)
    # )
    # pp = np.exp(lpp)
    new_nodes = np.random.poisson(pp)
    Ncols = new_nodes.sum()
    node_count = np.zeros(Ncols)

    # used to build sparse matrix, entries of each Zij=1
    colidx = []
    rowidx = []
    rightmost_node = 0

    # for each clique
    for n in range(K):
        # revisit each previously seen node
        for k in range(rightmost_node):
            prob_repeat = (node_count[k] - sigma) / (n + c)
            r = np.random.rand()
            if r < prob_repeat:
                rowidx.append(n)
                colidx.append(k)
                node_count[k] += 1

        for k in range(rightmost_node, rightmost_node + new_nodes[n]):
            rowidx.append(n)
            colidx.append(k)
            node_count[k] += 1

        rightmost_node += new_nodes[n]

    # build sparse matrix
    data = np.ones(len(rowidx), int)
    shape = (K, Ncols)
    Z = csr_matrix((data, (rowidx, colidx)), shape)

    return Z


if __name__ == "__main__":
    K, alpha, sigma, c = 10, 3, 0.7, 5
    Z = sample_from_ibp(K, alpha, sigma, c)

    adj = Z.transpose() @ Z
    g = nx.from_scipy_sparse_matrix(adj)
    for n in g.nodes():
        g.remove_edge(n, n)

    print("Number of edges:", g.number_of_edges())
    print("Number of nodes:", g.number_of_nodes())

    # Transform to a torch geometric data object
    data = torch_geometric.utils.from_networkx(g)
    data.x = torch.ones(data.num_nodes, 1)

    # Lift the topology to a cell complex
    lifting = LatentCliqueCoverLifting(piex=0.8, it=100, init="edges")
    complex = lifting.lift_topology(data)

    # Print the cell complex
    print(complex)
