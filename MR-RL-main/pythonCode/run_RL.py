import numpy as np
from sklearn.model_selection import train_test_split
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
import torch
action_scaler = MinMaxActionScaler(minimum=np.array([-3000,-3000,-4000,-3000]),maximum=np.array([3000,3000,4000,3000]))

dataset = d3rlpy.dataset.MDPDataset(
        observations=np.load('observations9.npy'),
        actions=np.load('actions9.npy'),
        rewards=np.load('rewards9.npy'),
        terminals=np.load('terminals9.npy'),
    )

dqn = d3rlpy.algos.BCQ(use_gpu=False,action_scaler=action_scaler,scaler="standard")
dqn.build_with_dataset(dataset)
# episode-wise split
train_episodes, test_episodes = train_test_split(dataset.episodes)
# setup metrics
metrics = {
    "soft_opc": d3rlpy.metrics.scorer.soft_opc_scorer(return_threshold=5),
    "initial_value": d3rlpy.metrics.scorer.initial_state_value_estimation_scorer,
}

# start training with episode-wise splits
dqn.fit(
    train_episodes,
    n_steps=30000,
    scorers=metrics,
    eval_episodes=test_episodes,
)
# save full parameters
dqn.save_model('bcq_test9.pt')
# dqn2 = d3rlpy.algos.CQL()
# dqn2.build_with_dataset(dataset)
# dqn2.load_model('dqn.pt')
# # 创建一个新的动作缩放器对象
# action_scaler = MinMaxActionScaler(minimum=[-3000, -3000, -4000], maximum=[3000, 3000, 4000])

# x_batch =  [ .08311828e+01, -1.73080331e+01, -6.52734832e+01 , 1.34699288e+04,
#    1.16981113e+04 , 9.57650246e+03]
# x_array = np.array(x_batch)
# x_array = x_array.reshape(1, -1)
# actions_new = dqn2.predict(x_array)[0]
# # 将其转换为 tensor
# actions_new_tensor = torch.tensor(actions_new, dtype=torch.float32)
# # 使用 `reverse_transform` 方法来恢复原始的动作值
# actions_original = action_scaler.reverse_transform(actions_new_tensor)
# # 如果需要，你可以将结果再转换回 numpy 数组
# actions_original = actions_original.numpy()
# print("训练得到的action：",actions_original)
