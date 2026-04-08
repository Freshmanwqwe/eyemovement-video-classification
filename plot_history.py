import json
import os
import argparse
import matplotlib.pyplot as plt

def plot_json_history(json_path):
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"Error: 文件 {json_path} 不存在。")
        return

    # 读取 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {json_path} 不是一个有效的 JSON 文件。")
            return

    # 获取所有键名（排除空列表）
    keys = [k for k in data.keys() if isinstance(data[k], list) and len(data[k]) > 0]
    
    if not keys:
        print("JSON 文件中没有找到可以绘制的列表数据。")
        return

    base_name = os.path.splitext(json_path)[0]
    
    for key in keys:
        values = data[key]
        
        # 为每个键创建独立的 Figure
        plt.figure(figsize=(10, 4))
        
        # 绘制线图
        plt.plot(range(1, len(values) + 1), values, marker='.', label=key)
        
        plt.title(f'Plot of {key}')
        plt.xlabel('Epochs / Steps')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # 为每个图分别保存名称
        output_image_path = f"{base_name}_{key}.png"
        plt.savefig(output_image_path, dpi=300)
        print(f"图像 {key} 生成成功！已保存至：{output_image_path}")

    # 显示所有图像
    plt.show()

if __name__ == "__main__":
    # 默认路径使用你当前打开的文件路径
    default_path = r"_output\output4reg\training_history.json"
    
    parser = argparse.ArgumentParser(description="Plot history from a JSON file.")
    parser.add_argument('--file', type=str, default=default_path, help='Path to the JSON file')
    
    args = parser.parse_args()
    plot_json_history(args.file)