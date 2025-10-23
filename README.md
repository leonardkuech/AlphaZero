This project implements the AlphaZero algorithm, a general-purpose reinforcement learning algorithm developed by DeepMind, for a custom-built, two-player, perfect-information board game called "Gliders". The algorithm combines a Monte Carlo Tree Search (MCTS) with a deep convolutional neural network (CNN) to learn and master the game from scratch.

The repository includes:
- The complete game logic for "Gliders", a hexagonal grid-based game.
- A `tkinter`-based UI for game visualization and human-vs-AI gameplay.
- The core AlphaZero implementation, including the MCTS and the CNN.
- Several other agents for comparison, such as MinMax, standard MCTS, and a Greedy agent.

## Structure

The project is organized into several key directories under `src/`:

*   `src/`
    *   `alpha_zero/`: Contains the core components of the AlphaZero agent.
        *   `CNN.py`: The PyTorch implementation of the convolutional neural network. It predicts move probabilities (policy) and game outcome (value) from a given game state.
        *   `MCTS.py`: The AlphaZero-specific Monte Carlo Tree Search, which uses the neural network to guide its search.
        *   `AZNode.py`: A specialized node class for the AlphaZero MCTS.
    *   `agents/`: Implements various AI agents to play the game.
        *   `Agent.py`: An abstract base class for all agents.
        *   `GreedyAgent.py`: A simple agent that always chooses the move leading to the highest-value tile.
        *   `PrunedMinMaxAgent.py`: A classic game-playing AI using the alpha-beta pruning algorithm.
        *   `MCTSAgent.py`: A standard MCTS agent with random playouts for comparison.
        *   `MCTSHardPlayout.py`: An MCTS agent that uses a MinMax agent for more intelligent (but slower) playouts.
    *   `TimedAgents/`: Contains agents that decide moves based on a time limit rather than a fixed number of simulations.
        *   `TimedMCTSAgent.py`: A time-based version of the standard MCTS agent.
    *   `game/`: Contains the core game logic.
        *   `GameState.py`: Defines the game state, rules, move generation, and win/loss conditions for "Gliders". It is heavily optimized with `numba` for performance.
    *   `ui/`: The graphical user interface for the game.
        *   `HexGridUI.py`: A `tkinter`-based GUI to render the hexagonal game board and handle user input.
    *   `Utils.py`: A collection of utility functions, including coordinate systems for the hex grid.
    *   `Node.py`: The node class used by the standard MCTS agents.

## The Game: Sugar Gliders

"Sugar Gliders" is a two-player strategy game played on a hexagonal grid.

*   **Objective**: To finish the game with a higher score than the opponent. A player's score is the sum of the values of tiles in their "reserve".
*   **Setup**: The board is populated with tiles of values 1 through 5. Each player places their "glider" on a starting tile.
*   **Gameplay**:
    *   Players take turns moving their glider.
    *   When a glider is on a numbered tile, it can move a distance exactly equal to that tile's number.
    *   When a glider is on an empty tile (or the center), it can spend a collected tile from its reserve to move a distance equal to the spent tile's value.
    *   When a glider lands on a numbered tile, that tile's value is added to the player's reserve, and the tile is removed from the board.
*   **End Game**: The game ends when both players pass their turn consecutively or when it's mathematically impossible for the trailing player to win.

## Author

Leonard Küch
