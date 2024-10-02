import torch
import cv2
import numpy as np

# 加载预训练的 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 读取图像
image = cv2.imread("car_image.jpg")

# 转换图像为 RGB 格式（YOLOv5 需要 RGB 图像）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 使用模型进行推理（YOLOv5 会自动处理预处理、前向传播、后处理等）
results = model(image_rgb)

# 解析检测结果
results_df = results.pandas().xyxy[0]  # 获取检测框的 pandas DataFrame 格式
print(results_df)  # 打印结果，包含类别、坐标等信息

# 遍历检测结果并标注汽车
for index, row in results_df.iterrows():
    # 检测的是汽车
    if row['name'] == 'car':
        # 获取边框坐标
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # 画框和标签
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示图像
cv2.imshow("YOLOv5 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
