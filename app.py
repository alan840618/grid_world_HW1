from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

def evaluate_values(grid_size, start, goal, obstacles):
    V = np.zeros((grid_size, grid_size))
    gamma = 0.9  # 折扣因子
    reward_obstacle = -1.0  # 障礙物的懲罰值
    reward_default = -0.4  # 一般移動的獎勵
    reward_goal = 20.0  # 終點的獎勵
    theta = 1e-6  # 收斂條件
    delta = float('inf')  # 變化量初始化為無窮大
    
    while delta > theta:  # 使用 while 迴圈直到收斂
        delta = 0
        new_V = np.copy(V)
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) == goal:
                    continue  # 終點格本身不再更新
                elif (i, j) in obstacles:
                    new_V[i, j] = reward_obstacle  
                else:
                    values = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            if (ni, nj) == goal:
                                values.append(reward_goal)  # 若下一步是終點，獲得 +20
                            elif (ni, nj) in obstacles:
                                values.append(reward_obstacle)  # 若下一步是障礙物，獲得 -1
                            else:
                                values.append(gamma * V[ni, nj])
                        else:
                            values.append(reward_obstacle)  # 撞牆視為碰到障礙物
                    new_V[i, j] = reward_default + max(values)
                    delta = max(delta, abs(new_V[i, j] - V[i, j]))  # 計算最大變化量
        V = new_V
    return V.tolist()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_values', methods=['POST'])
def calculate_values():
    data = request.json
    grid_size = int(data['grid_size'])
    start = tuple(data['start'])
    goal = tuple(data['goal'])
    obstacles = [tuple(o) for o in data['obstacles']]
    values = evaluate_values(grid_size, start, goal, obstacles)
    return jsonify(values)

if __name__ == '__main__':
    app.run(debug=True)