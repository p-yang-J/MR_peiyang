from hyperopt import hp, fmin, tpe, Trials

# 定义搜索空间
space = {
    'x': hp.uniform('x', -10, 10),
    'y': hp.uniform('y', -10, 10),
    'z': hp.uniform('z', -10, 10),
}

def objective(params):
    # 这个函数应该根据当前的x, y, z值返回对应的轨迹偏差
    x = params['x']
    y = params['y']
    z = params['z']
    error = robot_error(x, y, z)  # 需要你自己定义
    return error
def robot_error(x,y,z):
    # get the error


    return robot_error()

# 运行优化
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print("Best parameters: ", best)

