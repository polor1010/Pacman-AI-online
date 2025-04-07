# ghostAgents.py
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


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class SmarterRandomGhost(GhostAgent):
    """
    An enhanced version of RandomGhost that has a certain chance to move
    towards Pacman, making it more challenging than a purely random ghost.
    This ghost has randomness but with a bias towards hunting the player.
    It also includes team coordination to surround Pacman from different directions.
    """
    
    def __init__(self, index, hunting_prob=0.65, teamwork_enabled=True):
        """
        Initialize with an index and probability of hunting.
        hunting_prob: probability that the ghost will chase Pacman
        teamwork_enabled: whether ghosts coordinate their movements
        """
        self.index = index
        self.hunting_prob = hunting_prob
        self.teamwork_enabled = teamwork_enabled
        # 用於團隊協作的隨機偏移，使不同的鬼魂選擇不同的追逐路徑
        self.position_bias = (random.uniform(-2, 2), random.uniform(-2, 2))
    
    def getDistribution(self, state):
        # Start with a uniform distribution over legal actions
        dist = util.Counter()
        legalActions = state.getLegalActions(self.index)
        
        # Random component - give each action a base probability
        for a in legalActions:
            dist[a] = 1.0
            
        # Add the hunting component - evaluate which actions get closer to Pacman
        ghostPos = state.getGhostPosition(self.index)
        pacmanPos = state.getPacmanPosition()
        
        # 檢查鬼魂是否處於驚嚇狀態
        ghostState = state.getGhostState(self.index)
        isScared = ghostState.scaredTimer > 0
        
        # 獲取其他鬼魂的位置，用於團隊協作
        otherGhostPositions = []
        if self.teamwork_enabled:
            for i in range(1, state.getNumAgents()):
                if i != self.index:
                    otherGhostPositions.append(state.getGhostPosition(i))
        
        # 決定是否追逐 Pacman
        if random.random() < self.hunting_prob and not isScared:
            actionVectors = [Actions.directionToVector(a, 1.0) for a in legalActions]
            newPositions = [(ghostPos[0] + a[0], ghostPos[1] + a[1]) for a in actionVectors]
            
            # 計算到 Pacman 的距離
            targetPos = pacmanPos
            
            # 如果啟用團隊協作，則考慮其他鬼魂的位置，避免所有鬼魂選擇相同的路徑
            if self.teamwork_enabled and otherGhostPositions:
                # 為每個鬼魂添加一個小偏移，讓它們從不同方向包圍 Pacman
                biased_target = (pacmanPos[0] + self.position_bias[0], 
                                 pacmanPos[1] + self.position_bias[1])
                
                # 檢查是否有其他鬼魂已經很靠近 Pacman，如果有，則嘗試從不同方向接近
                close_ghosts = [pos for pos in otherGhostPositions 
                               if manhattanDistance(pos, pacmanPos) < 3]
                
                if close_ghosts:
                    # 使用預測的 Pacman 位置，假設 Pacman 會遠離已經靠近的鬼魂
                    # 這樣可以幫助形成包圍圈
                    avg_close_ghost_x = sum(pos[0] for pos in close_ghosts) / len(close_ghosts)
                    avg_close_ghost_y = sum(pos[1] for pos in close_ghosts) / len(close_ghosts)
                    
                    # 預測 Pacman 的逃跑方向（遠離平均鬼魂位置）
                    escape_dir_x = pacmanPos[0] - avg_close_ghost_x
                    escape_dir_y = pacmanPos[1] - avg_close_ghost_y
                    
                    # 預測 Pacman 的下一個位置
                    predicted_x = pacmanPos[0] + (0.5 * escape_dir_x)
                    predicted_y = pacmanPos[1] + (0.5 * escape_dir_y)
                    
                    # 使用預測位置作為目標
                    targetPos = (predicted_x, predicted_y)
                else:
                    # 如果沒有鬼魂靠近，使用帶偏移的目標位置
                    targetPos = biased_target
            
            # 計算到目標的距離
            distancesToTarget = [manhattanDistance(pos, targetPos) for pos in newPositions]
            
            # 找到最佳行動（最小化到目標的距離）
            bestScore = min(distancesToTarget)
            bestIndices = [i for i, d in enumerate(distancesToTarget) if d == bestScore]
            
            # 給接近 Pacman 的行動更高的權重
            for idx in bestIndices:
                # 更強的追逐獎勵
                dist[legalActions[idx]] += 4.0
                
            # 額外考慮：檢查是否有行動可以直接捕獲 Pacman
            for i, pos in enumerate(newPositions):
                if manhattanDistance(pos, pacmanPos) <= 1:  # 如果下一步可以捕獲 Pacman
                    dist[legalActions[i]] += 6.0  # 給予極高獎勵
                
        elif isScared:  # 如果鬼魂處於驚嚇狀態
            actionVectors = [Actions.directionToVector(a, 1.0) for a in legalActions]
            newPositions = [(ghostPos[0] + a[0], ghostPos[1] + a[1]) for a in actionVectors]
            
            # 計算到 Pacman 的距離
            distancesToPacman = [manhattanDistance(pos, pacmanPos) for pos in newPositions]
            
            # 找到最大化與 Pacman 距離的行動
            bestScore = max(distancesToPacman)
            bestIndices = [i for i, d in enumerate(distancesToPacman) if d == bestScore]
            
            # 給遠離 Pacman 的行動更高的權重
            for idx in bestIndices:
                dist[legalActions[idx]] += 5.0
                
            # 檢查是否有其他鬼魂靠近，如果有，則向它們的方向移動尋求保護
            if self.teamwork_enabled and otherGhostPositions:
                non_scared_ghosts = []
                for i in range(1, state.getNumAgents()):
                    if i != self.index and state.getGhostState(i).scaredTimer <= 0:
                        non_scared_ghosts.append(state.getGhostPosition(i))
                
                if non_scared_ghosts:
                    # 找到最近的非驚嚇鬼魂
                    closest_ghost_pos = min(non_scared_ghosts, 
                                         key=lambda pos: manhattanDistance(ghostPos, pos))
                    
                    # 計算哪個行動會使鬼魂更接近這個非驚嚇鬼魂
                    distancesToGhost = [manhattanDistance(pos, closest_ghost_pos) 
                                       for pos in newPositions]
                    bestScore = min(distancesToGhost)
                    bestIndices = [i for i, d in enumerate(distancesToGhost) if d == bestScore]
                    
                    # 給靠近非驚嚇鬼魂的行動更高的權重
                    for idx in bestIndices:
                        dist[legalActions[idx]] += 3.0
        
        # 常規移動行為：避免在複雜地圖中卡住
        # 檢查當前位置是否靠近牆角（即三個方向都是牆壁）
        wall_count = 0
        for a in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            if a not in legalActions:
                wall_count += 1
                
        # 如果當前位置很受限（有多個方向是牆壁），偏好向更開放的方向移動
        if wall_count >= 2:  # 如果至少有兩個方向是牆壁
            for a in legalActions:
                # 計算前進一步後的位置
                vec = Actions.directionToVector(a, 1.0)
                next_pos = (int(ghostPos[0] + vec[0]), int(ghostPos[1] + vec[1]))
                
                # 計算在下一個位置有多少合法的移動選項
                next_legal_actions = []
                for next_a in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                    next_vec = Actions.directionToVector(next_a, 1.0)
                    next_next_pos = (int(next_pos[0] + next_vec[0]), int(next_pos[1] + next_vec[1]))
                    # 簡單檢查：如果是當前位置則繼續
                    if next_next_pos == ghostPos:
                        continue
                    # 檢查邊界條件（這裡簡化處理）
                    if 0 <= next_next_pos[0] < state.data.layout.width and 0 <= next_next_pos[1] < state.data.layout.height:
                        if not state.hasWall(next_next_pos[0], next_next_pos[1]):
                            next_legal_actions.append(next_a)
                
                # 如果下一個位置有更多選項，增加權重
                if len(next_legal_actions) > len(legalActions) - wall_count:
                    dist[a] += 1.5
        
        # 標準化分佈
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared:
            speed = 0.5

        actionVectors = [Actions.directionToVector(
            a, speed) for a in legalActions]
        newPositions = [(pos[0]+a[0], pos[1]+a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(
            pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(
            legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1-bestProb) / len(legalActions)
        dist.normalize()
        return dist
