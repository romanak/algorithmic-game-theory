# Algorithmic Game Theory

## Political game

The implementation of the political game is based on the paper Chuang-Chieh Lin et al. [How Good Is a Two-Party Election Game?](https://arxiv.org/abs/2001.05692), 2020.

### Assumptions

- There are two parties as strategic players according to the Duverger's law.
- Every party has the same number of candidates.
- Each party is represented as a matrix of utility values.
- Each row of the party matrix represents one candidate.
- The first column represents the number of candidate's own party supporters, the second column - the candidate's opposing party supporters, and the third column (optional) - the swing voters:

![A=\begin{bmatrix}u_A(A_1)&u_B(A_1)&u_S(A_1)\\\\u_A(A_2) & u_B(A_2)&u_S(A_2)\\\\\end{bmatrix}](https://latex.codecogs.com/svg.latex?A=\begin{bmatrix}u_A(A_1)&u_B(A_1)&u_S(A_1)\\\\u_A(A_2) & u_B(A_2)&u_S(A_2)\\\\\end{bmatrix})

![B=\begin{bmatrix}u_B(B_1)&u_A(B_1)&u_S(B_1)\\\\u_B(B_2)&u_A(B_2)&u_S(B_2)\\\\\end{bmatrix}](https://latex.codecogs.com/svg.latex?B=\begin{bmatrix}u_B(B_1)&u_A(B_1)&u_S(B_1)\\\\u_B(B_2)&u_A(B_2)&u_S(B_2)\\\\\end{bmatrix})

- There may be more than 2 candidates per party.
- The candidates are sorted in descending order according to the first column.
- The winning odds are calculated according to one of the three models: Linear Link, Bradley-Terry, and Softmax.
- Two parties are egoistic if candidates benefit (have the largest number) their own supporters more than those from the competing party.

### Requirements

- Python >= 3.8
- Numpy >= 1.21

### Usage

```python
# import the module
import PoliticalGame as pg

# initialize the object
# possible models: LinearLink, BradleyTerry, Softmax
# seed value is used for reproducible results
polgame = pg.PoliticalGame(num_candidates=2, social_bound=100, \
        model=pg.LinearLink, swing_voters=False, force_egoism=False, seed=0)

# run 10 elections
polgame.run_iterations(10)

# run 100 more elections
polgame.run_iterations(100)

# get the history record of the fist election as a tuple of:
# 0: utility values of party A
# 1: utility values of party B
# 2: probabilities of winning
# 3: payoffs of party A
# 4: payoffs of party B
# 5: position of worst Pure Nash equilibrium
# 6: social welfare value of worst Pure Nash equilibrium
# 7: Price of Anarchy
polgame.history[0]
```

### Additional comments

When voting, voters care about the following:

1. Likeability of the candidates, for example how friendly they seem to be, how honest they are, etc.

2. Policy that candidates introduce on the spectrum from liberal (Left) to conservative (Right) and how close the candidate is to the voter's position in the spectrum.