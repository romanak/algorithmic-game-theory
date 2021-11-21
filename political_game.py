#!/usr/bin/python3

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 2021

Midterm project to implement the political game for Algorithmic Game Theory
class at Tamkang University. Based on Chuang-Chieh Lin et al. paper "How Good
Is a Two-Party Election Game?"

@author: Roman Akchurin
"""


class LinearLink(object):
    """Linear link model for computing the winning odds for a party."""
    @staticmethod
    def fn(A, B, social_bound):
        m = A.shape[0] # number of candidates in party A
        n = B.shape[0] # number of candidates in party B
        
        P = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                uA_i = A[i,:].sum()
                uB_j = B[j,:].sum()
                P[i,j] = 0.5 + (uA_i - uB_j)/(2*social_bound)
        return P

class BradleyTerry(object):
    """Bradley-Terry model for computing the winning odds for a party."""
    @staticmethod
    def fn(A, B, social_bound):
        """Bradley-Terry model formulates a candidate `i` winning odds
        against the competing candidate `j`."""
        m = A.shape[0] # number of candidates in party A
        n = B.shape[0] # number of candidates in party B
        
        P = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                uA_i = A[i,:].sum()
                uB_j = B[j,:].sum()
                P[i,j] = uA_i/(uA_i + uB_j)
        return P

class Softmax(object):
    """Softmax model for computing the winning odds for a party."""
    @staticmethod
    def fn(A, B, social_bound):
        m = A.shape[0] # number of candidates in party A
        n = B.shape[0] # number of candidates in party B
        
        P = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                uA_i = A[i,:].sum() / social_bound
                uB_j = B[j,:].sum() / social_bound
                P[i,j] = np.exp(uA_i)/(np.exp(uA_i) + np.exp(uB_j))
        return P

class PoliticalGame(object):
    """Political game object implementation."""
    def __init__(self, num_candidates=2, social_bound=100, \
        model=LinearLink, swing_voters=True, iterations=100, \
        force_egoism=False, seed=None):
        self.num_candidates = num_candidates
        self.social_bound = social_bound
        self.model = model
        self.swing_voters = swing_voters
        self.iterations = iterations
        self.force_egoism = force_egoism
        self.rng = np.random.default_rng(seed)
        self.history = list()

    def generate_party(self):
        """Generate a party with the following properties:
        `self.social_bound` is the bound of social utilities;
        `self.num_candidates` is the number of candidates in the party.
        
        Supporters for the candidate could be:
            (a) supporters for the candidate's party;
            (b) supporters for the opposing party;
            (c) swing voters

        Returns a party as a utility matrix."""
        
        if self.swing_voters == False:
            voter_types = 2 # default types of supporters
        else:
            voter_types = 3 # adding swing voters
        # create the first dummy candidate
        party = np.zeros((1,voter_types), dtype=np.int8)
        for i in range(self.num_candidates):
            while True:
                # create candidates one by one
                candidate = self.rng.integers(0, self.social_bound, \
                    size=(1,voter_types), dtype=np.int8, endpoint=True)
                if np.sum(candidate.sum(axis=1) <= self.social_bound) == 1:
                    break
            party = np.vstack((party,candidate))
        # remove the first dummy candidate
        party = party[1:,:]
        # sort based on the number of supporters for the candidates's party
        party = party[np.argsort(party[:, 0])][::-1]
        return party

    def confirm_egoism(self, A, B):
        """Check whether the selected pair of parties `A` and `B` are agoistic."""
        m = A.shape[0] # number of candidates in party A
        for i in range(m):
            if A[i,0] <= B[:,1].max() or B[i,0] <= A[:,1].max():
                return False
        return True

    def get_payoffs(self, A, B, P):
        """Compute the payoffs of two-party election game as expected utilities.
        
        Takes `A` the utility matrix for party A, `B` the utility matrix for
        party B, `P` the probabilities of winning according to the selected
        model.
        
        Reurns `a` the payoffs of party `A`, `b` the payoffs of party `B`"""
        a = np.zeros((self.num_candidates,self.num_candidates))
        b = np.zeros((self.num_candidates,self.num_candidates))
        
        for i in range(self.num_candidates):
            for j in range(self.num_candidates):
                a[i,j] = P[i,j]*A[i,0] + (1-P[i,j])*B[j,1]
                b[i,j] = (1-P[i,j])*B[j,0] + P[i,j]*A[i,1]
        return (a,b)

    def get_optimal_state(self, a, b, social_welfare):
        """Optimal state has the highest social welfare among all possible states.
        Takes payoff `a` of party A and payoff `b` of party B
        returns the best social welfare value."""
        max_idx = np.unravel_index(np.argmax(social_welfare, axis=None), \
            social_welfare.shape)
        return social_welfare[max_idx]

    def get_worst_PNE(self, a, b, social_welfare):
        """Pure Nash Equilibrium;
        Takes payoff `a` of party A and payoff `b` of party B
        returns position and worst pure nash equilibrium"""
        PNEs = list()

        for i in range(self.num_candidates):
            for j in range(self.num_candidates):
                if a[i,j] == a[:,j].max() and b[i,j] == b[i,:].max():
                    PNEs.append(((i,j), social_welfare[i,j]))
        # sort PNEs in ascending order
        PNEs.sort(key=lambda a: a[1])
        # return the worst (smallest in value) PNE
        if len(PNEs) > 0:
            return PNEs[0]
        else:
            return ((None, None), None)

    def get_PoA(self, optimal_state, PNE_val):
        """Calculate the price of anarchy."""
        if PNE_val == None or PNE_val == 0:
            return 0
        else:
            return optimal_state / PNE_val

    def run_election(self):
        """Run election once"""
        A = self.generate_party()
        B = self.generate_party()

        if self.force_egoism:
            while not self.confirm_egoism(A, B):
                A = self.generate_party()
                B = self.generate_party()

        P = self.model.fn(A, B, self.social_bound)
        a, b = self.get_payoffs(A, B, P)
        social_welfare = a + b
        optimal_state = self.get_optimal_state(a, b, social_welfare)
        (PNE_pos, PNE_val) = self.get_worst_PNE(a, b, social_welfare)
        PoA = self.get_PoA(optimal_state, PNE_val)
        return (A, a, B, b, PNE_pos, PNE_val, PoA)

    def run_iterations(self):
        """Run election `self.iterations` times."""
        for i in range(self.iterations):
            (A, a, B, b, PNE_pos, PNE_val, PoA) = self.run_election()
            self.history.append((A, a, B, b, PNE_pos, PNE_val, PoA))

if __name__ == "__main__":
    polgame = PoliticalGame(num_candidates=2, social_bound=100, \
        model=LinearLink, swing_voters=True, iterations=100, \
        force_egoism=False, seed=0)
    polgame.run_iterations()
    print(polgame.history)