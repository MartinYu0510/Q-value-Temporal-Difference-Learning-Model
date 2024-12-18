"""
To run the code, please use the following command:  python3 p4.py
*If you wish to change the input file, please change the filePath variable in the main function.

Ideas: The model implemet the exploration function for forcing exporation. Random exploration is not considered in this model due to the measure of regret.

Given that 1000 iterations are done, in each iterations, the model will learn the optimal policy for 1000 epoches
Anaylsis:
1. Exploration constant K: 
    Larger K: Let's say K = 10, the number of optimal policies found is 10/1000 
    Smaller K: Let's say K = 0.5, the number of optimal policies found is 12/1000
    Observation: With a smaller K, the model will intend to explore less and exploit more. In this way, the model tends to choose a safe path and avoid moviing to a wrong direction.
    
2. Learning rate alpha:
    Larger alpha: Let's say alpha = 0.9, the number of optimal policies found is 10/1000
    Smaller alpha: Let's say alpha = 0.4, the number of optimal policies found is 16/1000
    Observation: With a smaller learning rate, the model will converge faster and achieve optimal faster. Particularly, the q-value in each state will highly depends on the q-value of the current state instead of the sample (i.e. the reward and the q-value of the next state). In this way, the model can reduce the effect of noise in the environment and find the correct optimal policy.
    
3. alpha decay rate:
    Larger alpha decay: Let's say alpha decay = 0.9, the number of optimal policies found is 7/1000
    Smaller alpha decay: Let's say alpha decay = 0.5, the number of optimal policies found is 10/1000
    Observation: With a smaller alpha decay, the model will reduce the learning rate slower so that the model can learn slower and capture more details in each epoch.
"""

import random
import math
minAlpha = 0.05

def checkVaildMove(currPos, direction, grid):
    if direction == "N":
        if currPos[0] - 1 < 0 :
            return False
        elif grid[currPos[0] - 1][currPos[1]] == "#":
            return False
    elif direction == "S":
        if currPos[0] + 1 >= len(grid):
            return False
        elif grid[currPos[0] + 1][currPos[1]] == "#":
            return False
    elif direction == "E":
        if currPos[1] + 1 >= len(grid[0]):
            return False
        elif grid[currPos[0]][currPos[1] + 1] == "#":
            return False
    elif direction == "W":
        if currPos[1] - 1 < 0:
            return False
        elif grid[currPos[0]][currPos[1] - 1] == "#":
            return False
    return True

def getExitState(grid):
    exit = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if(grid[i][j].isnumeric() or "-" in grid[i][j]):
                exit.append((i,j))
    return exit

def getStartState(grid):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if(grid[i][j]=="S"):
                return (i,j)
    return (2,0)

def posUpdate(currPos, action):
    if(action == "N"):
        return (currPos[0] - 1, currPos[1])
    elif(action == "S"):
        return (currPos[0] + 1, currPos[1])
    elif(action == "E"):
        return (currPos[0], currPos[1] + 1)
    elif(action == "W"):
        return (currPos[0], currPos[1] - 1)
    return currPos 

def stateUpdate( curPos,grid,actions,exit,noise,isCheck):
    direction = {'N':['N', 'E', 'W'], 'E':['E', 'S', 'N'], 'S':['S', 'W', 'E'], 'W':['W', 'N', 'S']}
    weight = [1-noise*2, noise, noise]
    if(not isCheck):
        action = random.choices(direction[actions], weight)[0]
    else:
        action = actions
    if(curPos in exit):
        return curPos,action
    if(checkVaildMove(curPos, action, grid)):
        nextPos = posUpdate(curPos, action)
    else:
        nextPos = curPos
    return nextPos,action

def checkOptimal(qVal,grid,exit,noise):
    curPos = getStartState(grid)
    for i in range(len(grid) * len(grid[0])):
        action = max(qVal[curPos], key=lambda action: qVal[curPos][action])
        # print(f"Position: {curPos}, Action: {action}")
        nextPos,dummy = stateUpdate(curPos, grid, action, exit,noise,True)
        if(nextPos in exit and float(grid[nextPos[0]][nextPos[1]]) > 0):
            return True
        elif(nextPos in exit):
            return False
        curPos = nextPos
    return False

def getPolicy(qVal,grid,exit):
    sol = {}
    for pos in qVal:
        actionMax = max(qVal[pos], key=lambda action: qVal[pos][action])
        sol[pos] = actionMax
    return sol
   
def qLearning(grid, discount, noise, rewards):
    exitState = getExitState(grid)
    direction = {'N':['N', 'E', 'W'], 'E':['E', 'S', 'N'], 'S':['S', 'W', 'E'], 'W':['W', 'N', 'S']}
    weight = [1-noise*2, noise, noise]
    qValGrid = {(i,j): {action : 0.00 for action in direction} for i in range(len(grid) ) for j in range(len(grid[0])) if grid[i][j] != "#"}
    gridVisited = {(i,j) : 0 for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j] != "#"}
    actionUsed = {(i,j) : {action : 0 for action in direction} for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j] != "#"}
    for pos in qValGrid:
        if pos in exitState:
            qValGrid[pos].clear()
            qValGrid[pos]["x"] = float(grid[pos[0]][pos[1]])
            actionUsed[pos].clear()
            actionUsed[pos]["x"] = 0
    explorK = 0.5
    alpha = 0.4
    a_decay = 0.4
    epoch = 1000
    curPos = getStartState(grid)
    optimalPolicy = {}
    action = ""
    for i in range(epoch):
        # print(f"Epoch: {i} epsilon: {epsilon} alpha: {alpha}")
        foundOptimal = False
        while True:
            if(curPos in exitState):
                break
            gridVisited[curPos] += 1
            action = max(qValGrid[curPos], key=lambda action: qValGrid[curPos][action] + explorK * math.sqrt((math.log(gridVisited[curPos]) + 1.00) / (actionUsed[curPos][action] + 1.00)))
            # print(f"Epoch: {i}, Position: {curPos}, Action: {action}")
            nextPos,realAction= stateUpdate(curPos, grid, action,exitState,noise,False)
            actionUsed[curPos][realAction] += 1
            actionMax = max(qValGrid[nextPos], key=lambda action: qValGrid[nextPos][action])
            sample = rewards + discount * qValGrid[nextPos][actionMax]
            if(nextPos in exitState):
                sample += float(grid[nextPos[0]][nextPos[1]])
            qValGrid[curPos][action] = (1-alpha) * qValGrid[curPos][action] + alpha * sample
            curPos = nextPos
        alpha *= a_decay   
        if(checkOptimal(qValGrid, grid, exitState,noise)):
            alpha *= a_decay ** 2
            foundOptimal = True
            optimalPolicy = getPolicy(qValGrid, grid, exitState)
        alpha = max(alpha, minAlpha)
    return foundOptimal, optimalPolicy

def policyPrinter(policy,grid):
    sol = ""
    for pos in policy:
        grid[pos[0]][pos[1]] = policy[pos]
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            sol += f"| {grid[i][j]} |"
        sol += "\n"
    return sol

def problemReading(filePath):
    with open (filePath) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    problem = {}
    problem["discount"] = float(lines[0].split()[1])
    problem["noise"] = float(lines[1].split()[1])
    problem["rewards"] = float(lines[2].split()[1])
    problem["iteration"] = int(lines[3].split()[1])
    problem["grid"] = []
    for i in range(4, len(lines)):
        if lines[i] == "policy:":
            break
        if lines[i] == "grid:":
            continue
        problem["grid"].append(lines[i].split())
    return problem

if __name__ == "__main__":
    filePath = "test_cases/p3/2.prob"
    problem = problemReading(filePath)
    trial = 1000
    random.seed(None)
    result = []
    countOpt = 0
    print("I am going to learn the optimal policy, please wait...\n")
    for i in range(trial):
        foundOpt, policy= qLearning(problem["grid"], problem["discount"], problem["noise"], problem["rewards"])
        if(foundOpt):
            result.append(policy)
            countOpt += 1
    sol = ""
    sol += f"Number of optimal policies: {countOpt}/{trial}\n"
    sol += f"Optimal policies:\n"
    grid = [x[:] for x in problem["grid"]]
    if(result):
        i = 1
        for policy in result:
            sol += f"Policy: {i}\n"
            sol += policyPrinter(policy, grid)
            sol += "\n"
            i += 1
    print(sol)
            
    