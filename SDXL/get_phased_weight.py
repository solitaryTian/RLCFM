import json
from collections import defaultdict
from scipy.signal import savgol_filter
# import matplotlib.pyplot as plt

import numpy as np
def moving_average(data, window_size=3):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

def savgol_smooth(data, window_size=5, poly_order=2):
    return savgol_filter(data, window_size, poly_order)


def calculate_mean(temp_loss_values):
    return np.mean(temp_loss_values)

def calculate_integral(temp_loss_values, global_steps):
    return np.trapz(temp_loss_values, global_steps)

# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def process_and_plot_data(data, output_file='test.json', plot_file='res_merge.png'):
    """
    读取 JSON 文件，处理数据，保存处理后的数据，填充缺失值并绘制图表。

    参数:
        output_file (str): 输出 JSON 文件路径。
        plot_file (str): 绘图保存路径，默认为 'res.png'。
    """


    # 处理数据
    def process_data(data):
        merged_data = defaultdict(lambda: defaultdict(list))

        # 第一步：合并相同 global_step 的 index 和 temp_loss
        for item in data:
            global_step = item["global_step"]
            indices = item["index"]
            temp_losses = item["temp_loss"]

            # 对 index 取 10 位数部分
            indices = [i // 10 * 10 for i in indices]

            # 将 index 和 temp_loss 按顺序拼接
            for idx, loss in zip(indices, temp_losses):
                merged_data[global_step][idx].append(loss)

        # 第二步：对相同 global_step 和 index 的 temp_loss 取平均
        result = []
        for global_step, index_dict in merged_data.items():
            for index, losses in index_dict.items():
                avg_loss = sum(losses) / len(losses)
                result.append({
                    "global_step": global_step,
                    "index": index,
                    "temp_loss": avg_loss
                })

        return result

    # 保存处理后的数据到 JSON 文件
    def save_json(data, file_path):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    # 填充缺失值
    def fill_missing_values(data, global_steps):
        index_data = {}

        # 初始化数据结构
        for item in data:
            index = item["index"]
            if index not in index_data:
                index_data[index] = {}

        # 填充已知值
        for item in data:
            global_step = item["global_step"]
            index = item["index"]
            temp_loss = item["temp_loss"]
            index_data[index][global_step] = temp_loss

        # 填充缺失值
        for index in index_data:
            for step in global_steps:
                if step not in index_data[index]:
                    # 向上追溯
                    current_step = step
                    while current_step >= 1:
                        if current_step - 2 in index_data[index]:
                            index_data[index][step] = index_data[index][current_step - 2]
                            break
                        current_step -= 2
                    else:
                        # 如果追溯到 global_step=1 仍不存在，则取 0
                        index_data[index][step] = 0

        return index_data

    # 绘制图表
    def plot_data(index_data, global_steps):
        last_100_mean_value_dict = {}

        # plt.figure(figsize=(10, 6))
        for index, temp_loss_data in index_data.items():
            # 获取每个 global_step 对应的 temp_loss
            temp_loss_values = [temp_loss_data.get(step, 0) for step in global_steps]

            # 对 temp_loss_values 进行平滑处理
            # smoothed_values = moving_average(temp_loss_values, window_size=3)
            # smoothed_steps = global_steps[:len(smoothed_values)]
            mean_value = calculate_mean(temp_loss_values)
            last_100_mean_value = calculate_mean(temp_loss_values[-100:])
            last_100_mean_value_dict[index] = last_100_mean_value
            # 绘制曲线
            # plt.plot(global_steps, temp_loss_values, label=f"Index {index} (origin Mean: {mean_value:.4f})")

        # plt.xlabel("Global Step")
        # plt.ylabel("Temp Loss")
        # plt.title("Temp Loss vs Global Step for Different Indices")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(plot_file)
        return last_100_mean_value_dict

    # 主逻辑
    # 读取原始数据

    # 处理数据
    processed_data = process_data(data)
    # print(f'processed_data={processed_data}')

    # 保存处理后的数据
    # save_json(processed_data, output_file)

    # 需要绘制的 global_steps
    # 找到最大的 global_step
    max_global_step = max(processed_data, key=lambda x: x["global_step"])["global_step"]

    # 找到最小的 global_step
    min_global_step = min(processed_data, key=lambda x: x["global_step"])["global_step"]
    
    
    
    global_steps = list(range(min_global_step, max_global_step, 2))
    # print(f'global_steps={global_steps}')
    # 填充缺失值
    index_data = fill_missing_values(processed_data, global_steps)

    # 绘制图表
    last_100_mean_value_dict = plot_data(index_data, global_steps)

    # print(last_100_mean_value_dict)
    # print(f"处理后的数据已保存到 {output_file}")
    # print(f"图表已保存到 {plot_file}")

    loss_list = []
    loss_list.append(last_100_mean_value_dict[0])
    loss_list.append(last_100_mean_value_dict[10])
    loss_list.append(last_100_mean_value_dict[20])
    loss_list.append(last_100_mean_value_dict[30])
    # print(f'loss_list={loss_list}')

    alpha = 1
    total = sum([la**alpha for la in loss_list])
    weights_list = [ (la**alpha)/total for la in loss_list]
    # print(f'weights_list={weights_list}')
    return  loss_list, weights_list






# 示例调用
if __name__ == "__main__":
    input_file = "/maindata/data/shared/public/songtao.tian/test_code/Phased-Consistency-Model-master/code/text_to_image_sdxl/outputs/base_model_PCM_sdxl_base_model_danbooru_dmd2_test_weight/iteration_data.json"
    data = read_json(input_file)
    data = [{'global_step': 1, 'index': [17, 29, 39, 8], 'temp_loss': [0.009973804466426373, 0.02695772796869278, 0.02782599627971649, 0.031140975654125214]}]
    output_file = "output_merge.json"
    process_and_plot_data(data, output_file)






# # 读取 JSON 文件
# def read_json(file_path):
#     with open(file_path, "r") as f:
#         return json.load(f)

# # 处理数据
# def process_data(data):
#     # 用于存储合并后的数据
#     merged_data = defaultdict(lambda: defaultdict(list))

#     # 第一步：合并相同 global_step 的 index 和 temp_loss
#     for item in data:
#         global_step = item["global_step"]
#         indices = item["index"]
#         temp_losses = item["temp_loss"]

#         # 对 index 取 10 位数部分
#         indices = [i // 10 * 10 for i in indices]

#         # 将 index 和 temp_loss 按顺序拼接
#         for idx, loss in zip(indices, temp_losses):
#             merged_data[global_step][idx].append(loss)

#     # 第二步：对相同 global_step 和 index 的 temp_loss 取平均
#     result = []
#     for global_step, index_dict in merged_data.items():
#         for index, losses in index_dict.items():
#             avg_loss = sum(losses) / len(losses)
#             result.append({
#                 "global_step": global_step,
#                 "index": index,
#                 "temp_loss": avg_loss
#             })

#     return result

# # 保存处理后的数据到 JSON 文件
# def save_json(data, file_path):
#     with open(file_path, "w") as f:
#         json.dump(data, f, indent=4)

# # 主函数
# def main1(output_file):
#     # 读取原始数据
#     input_file = "/maindata/data/shared/public/songtao.tian/test_code/Phased-Consistency-Model-master/code/text_to_image_sdxl/outputs/base_model_PCM_sdxl_base_model_danbooru_dmd2_test_weight/iteration_data.json"
#     data = read_json(input_file)

#     # 处理数据
#     processed_data = process_data(data)

#     # 保存处理后的数据
#     save_json(processed_data, output_file)

#     print(f"处理后的数据已保存到 {output_file}")






# import json
# import matplotlib.pyplot as plt



# # 填充缺失值
# def fill_missing_values(data, global_steps):
#     # 用于存储每个 index 对应的 temp_loss 数据
#     index_data = {}

#     # 初始化数据结构
#     for item in data:
#         index = item["index"]
#         if index not in index_data:
#             index_data[index] = {}

#     # 填充已知值
#     for item in data:
#         global_step = item["global_step"]
#         index = item["index"]
#         temp_loss = item["temp_loss"]
#         index_data[index][global_step] = temp_loss

#     # 填充缺失值
#     for index in index_data:
#         for step in global_steps:
#             if step not in index_data[index]:
#                 # 向上追溯
#                 current_step = step
#                 while current_step >= 1:
#                     if current_step - 2 in index_data[index]:
#                         index_data[index][step] = index_data[index][current_step - 2]
#                         break
#                     current_step -= 2
#                 else:
#                     # 如果追溯到 global_step=1 仍不存在，则取 0
#                     index_data[index][step] = 0

#     return index_data

# # 绘制图表
# def plot_data(index_data, global_steps):
#     plt.figure(figsize=(10, 6))
#     for index, temp_loss_data in index_data.items():
#         # 获取每个 global_step 对应的 temp_loss
#         temp_loss_values = [temp_loss_data.get(step, 0) for step in global_steps]

#         # 对 temp_loss_values 进行平滑处理
#         smoothed_values = moving_average(temp_loss_values, window_size=3)  # 或使用 savgol_smooth
#         smoothed_steps = global_steps[:len(smoothed_values)]
#         mean_value = calculate_mean(temp_loss_values)


#         # 绘制曲线
#         # plt.plot(global_steps, temp_loss_values, label=f"Index {index}")
#         plt.plot(smoothed_steps, smoothed_values, label=f"Index {index} (origin Mean: {mean_value:.4f})")


#     plt.xlabel("Global Step")
#     plt.ylabel("Temp Loss")
#     plt.title("Temp Loss vs Global Step for Different Indices")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('res.png')



# # 主函数
# def main2(input_file):
#     # 读取 JSON 文件

#     data = read_json(input_file)

#     # 需要绘制的 global_steps
#     global_steps = list(range(1,6395,2))

#     # 填充缺失值
#     index_data = fill_missing_values(data, global_steps)

#     # 绘制图表
#     plot_data(index_data, global_steps)




# # 运行主函数
# if __name__ == "__main__":
#     file = 'output.json'
#     main1(output_file=file)
#     main2(input_file=file)
