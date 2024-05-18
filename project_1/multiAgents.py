# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn (maximize)
                return max_value(state, depth)
            else:  # Ghosts' turn (minimize)
                return min_value(state, depth, agentIndex)

        def max_value(state, depth):
            v = float('-inf')
            legalActions = state.getLegalActions(0)
            if not legalActions:  # No more legal actions, return evaluation
                return self.evaluationFunction(state)
            
            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                value = minimax(successor, depth, 1)
                v = max(v, value)
            return v

        def min_value(state, depth, agentIndex):
            v = float('inf')
            next_agent = (agentIndex + 1) % state.getNumAgents()
            next_depth = depth + 1 if next_agent == 0 else depth

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:  # No more legal actions, return evaluation
                return self.evaluationFunction(state)
            
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = minimax(successor, next_depth, next_agent)
                v = min(v, value)
            return v

        # Find the best action for Pacman by starting the minimax process
        best_action = None
        best_value = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 0, 1)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def alpha_beta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn (maximize)
                return max_value(state, depth, alpha, beta)
            else:  # Ghosts' turn (minimize)
                return min_value(state, depth, agentIndex, alpha, beta)

        def max_value(state, depth, alpha, beta):
            v = float('-inf')
            legalActions = state.getLegalActions(0)
            if not legalActions:
                return self.evaluationFunction(state)

            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                value = alpha_beta(successor, depth, 1, alpha, beta)
                v = max(v, value)
                if v > beta:  # Pruning
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, depth, agentIndex, alpha, beta):
            v = float('inf')
            next_agent = (agentIndex + 1) % state.getNumAgents()
            next_depth = depth + 1 if next_agent == 0 else depth

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = alpha_beta(successor, next_depth, next_agent, alpha, beta)
                v = min(v, value)
                if v < alpha:  # Pruning
                    return v
                beta = min(beta, v)
            return v

        # Find the best action for Pacman by starting the alpha-beta process
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alpha_beta(successor, 0, 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn (maximize)
                return max_value(state, depth)
            else:  # Ghosts' turn (expectation)
                return exp_value(state, depth, agentIndex)

        def max_value(state, depth):
            v = float('-inf')
            legalActions = state.getLegalActions(0)
            if not legalActions:
                return self.evaluationFunction(state)

            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                value = expectimax(successor, depth, 1)
                v = max(v, value)
            return v

        def exp_value(state, depth, agentIndex):
            v = 0
            next_agent = (agentIndex + 1) % state.getNumAgents()
            next_depth = depth + 1 if next_agent == 0 else depth

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            num_actions = len(legalActions)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = expectimax(successor, next_depth, next_agent)
                v += value / num_actions
            return v

        # Find the best action for Pacman by starting the expectimax process
        best_action = None
        best_value = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 0, 1)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 3).

    DESCRIPTION: This function estimates the quality of a game state by taking into account
    the position of Pacman, the positions of ghosts, the remaining food, and other factors.

    """

    # Get the position of Pacman
    pacmanPos = currentGameState.getPacmanPosition()

    # Get the positions of the ghosts
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = [ghostState.getPosition() for ghostState in ghostStates]

    # Get the number of remaining food pellets
    foodList = currentGameState.getFood().asList()

    # Get the number of remaining food capsules
    capsuleList = currentGameState.getCapsules()

    # Initialize the evaluation score
    evaluation = currentGameState.getScore()

    # Calculate distances from food pellets
    foodDistances = [manhattanDistance(pacmanPos, food) for food in foodList]

    # Calculate distances from ghosts
    ghostDistances = [manhattanDistance(pacmanPos, ghostPos) for ghostPos in ghostPositions]

    # Update the evaluation score based on distances
    if foodDistances:
        evaluation -= min(foodDistances)
    if ghostDistances:
        evaluation += max(ghostDistances)

    # Update the evaluation score based on the number of food pellets and capsules
    evaluation -= len(foodList) + 2 * len(capsuleList)

    # Consider Power Pellets
    powerPellets = currentGameState.getCapsules()
    evaluation += len(powerPellets) * 10

    # Consider Ghost Vulnerability
    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            evaluation += 200 / (manhattanDistance(pacmanPos, ghostState.getPosition()) + 1)

    # Encourage Efficient Movement (optional)

    return evaluation

# Abbreviation
better = betterEvaluationFunction
