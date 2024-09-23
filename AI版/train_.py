from ultralytics import YOLO
import os
import shutil

# 使用当前工作目录
base_path = os.path.join(os.getcwd(), 'moonbix_dataset')
yaml_path = os.path.join(os.getcwd(), 'moonbix_data.yaml')

# 确保 weights 文件夹存在
weights_dir = os.path.join(os.getcwd(), 'weights')
os.makedirs(weights_dir, exist_ok=True)

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练模型
results = model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    name='moonbix_model',
    project='runs/detect',
    save=True,
    save_period=10,  # 每10个epoch保存一次
    pretrained=True,
    optimizer='Adam',
    lr0=0.001,
    weight_decay=0.0005,
    augment=True
)

# 复制最终模型到 weights 文件夹
final_model_path = os.path.join(os.getcwd(), 'runs', 'detect', 'moonbix_model', 'weights', 'best.pt')
if os.path.exists(final_model_path):
    shutil.copy(final_model_path, os.path.join(weights_dir, 'final_model.pt'))
    print(f"最终模型已复制到: {os.path.join(weights_dir, 'final_model.pt')}")
else:
    print("未找到最终模型文件，请检查训练输出路径")

# 验证模型
val_results = model.val()
print(f"验证结果: {val_results}")

# 打印训练摘要
print("\n训练摘要:")
if isinstance(results, dict):
    print(f"最佳验证mAP: {results.get('metrics/mAP50-95(B)', 'N/A')}")
else:
    print("无法获取详细的训练结果。")

print(f"模型文件应该在以下路径: {os.path.join(os.getcwd(), 'runs', 'detect', 'moonbix_model', 'weights')}")
print("同时，最终模型已被复制到 weights 文件夹。")

# 列出 weights 文件夹中的所有文件
print("\nweights 文件夹内容:")
for file in os.listdir(weights_dir):
    print(file)

print("\n训练和保存过程已完成。请检查 weights 文件夹和 runs/detect/moonbix_model/weights 文件夹中的模型文件。")