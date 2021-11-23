# Algorithmic Game Theory

## Political game

The implementation of the political game is based on the paper Chuang-Chieh Lin et al. [How Good Is a Two-Party Election Game?](https://arxiv.org/abs/2001.05692), 2020 ([more](https://romanakchurin.com/algorithmic-game-theory/political_game/)).

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
