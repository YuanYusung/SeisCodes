import json
import numpy as np
from scipy.interpolate import interp1d
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def load_data(sta):
    # 加载数据的逻辑
    disp = np.load(f'./v4/disps/disps_{sta}.pkl', allow_pickle=True)
    if len(disp["period"]) == 0:
        print(f"Station {sta} are NAN.")
        return [np.nan], [np.nan], [np.nan]
    
    print(f"Processing sta {sta}.")
    with open(f'./v4/res/Vs_res_{sta}.json', 'r') as f:
        res = json.loads(f.read())
    misfits = np.array(res['misfits'])
    models = np.array(res['models'])
    best_model = models[np.argmin(misfits)]
    pers = disp["period"]
    misfit = disp["fit"] - disp["obs"]

    return best_model, pers, misfit


def grid_model(model, depth_grid):
    thickness = model[1:, 0]
    vp = model[1:, 1] 
    vs = model[1:, 2]
    rho = model[1:, 3]

    depths = np.cumsum(thickness)
    depths = np.insert(depths, 0, 0.025)  # 添加地表深度0.025
    
    model_depths = []
    model_vp = []
    model_vs = []
    model_rho = []
    
    for i in range(len(thickness)):
        model_depths.extend([depths[i], depths[i+1]])
        model_vp.extend([vp[i], vp[i]])
        model_vs.extend([vs[i], vs[i]])
        model_rho.extend([rho[i], rho[i]])

    vp_fn = interp1d(model_depths, model_vp, kind='previous', fill_value="extrapolate")
    vs_fn = interp1d(model_depths, model_vs, kind='previous', fill_value="extrapolate")
    rho_fn = interp1d(model_depths, model_rho, kind='previous', fill_value="extrapolate")

    return vp_fn(depth_grid), vs_fn(depth_grid), rho_fn(depth_grid)


def process_station(sta, depth_grid):
    best_model, _, misfit = load_data(sta)
    if np.all(np.isnan(best_model)):
        return None, None, None, None  # 返回台站和缺失数据

    grid_vp, grid_vs, grid_rho = grid_model(best_model, depth_grid)
    return grid_vp, grid_vs, grid_rho, misfit


def concat_data(depth_max=2., dz=0.005):
    sta1, sta2 = 250, 400
    depth_grid = np.arange(0.025, depth_max + dz, dz)

    n_stations = sta2 - sta1 + 1
    n_depths = len(depth_grid)
    pers = np.load(f'./v4/disps/disps_{275}.pkl', allow_pickle=True)["period"]
    # 初始化网格数据
    grid_vp = np.full((n_stations, n_depths), np.nan)
    grid_vs = np.full((n_stations, n_depths), np.nan)
    grid_rho = np.full((n_stations, n_depths), np.nan)
    grid_misfit = np.full((n_stations, len(pers)), np.nan)

    # 创建 ProcessPoolExecutor 用于并行化
    with ProcessPoolExecutor() as executor:
        # 使用 partial 函数绑定 depth_grid 参数，避免在每个任务中重复传递
        future_results = [executor.submit(partial(process_station, sta, depth_grid)) for sta in range(sta1, sta2 + 1)]

        # 收集处理结果
        for i, future in enumerate(future_results):
            grid_vp_data, grid_vs_data, grid_rho_data, misfit_data = future.result()

            if grid_vp_data is not None:
                # 填充网格数据
                grid_vp[i, :] = grid_vp_data
                grid_vs[i, :] = grid_vs_data
                grid_rho[i, :] = grid_rho_data
                grid_misfit[i, :] = misfit_data

    model = {
        'loc': np.array(range(sta1, sta2 + 1)) * 0.02,
        'depth': depth_grid,
        'vp': grid_vp,
        'vs': grid_vs,
        'rho': grid_rho,
    }

    misfit = {
        'loc': np.array(range(sta1, sta2 + 1)) * 0.02,
        'per': pers,
        'misfit': grid_misfit,
    }

    # 保存到文件
    with open(f'./model_v4.pkl', "wb") as f1:
        pickle.dump(model, f1)

    with open(f'./misfit_v4.pkl', "wb") as f2:
        pickle.dump(misfit, f2)

    return grid_vp, grid_misfit, depth_grid

concat_data()
