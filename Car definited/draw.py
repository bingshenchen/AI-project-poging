import torch
import cv2

# 加载预训练的 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 读取图像
image = cv2.imread("car_image.jpg")

# 转换图像为 RGB 格式（YOLOv5 需要 RGB 图像）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 使用模型进行推理
results = model(image_rgb)

# 解析检测结果
results_df = results.pandas().xyxy[0]  # 获取检测框的 pandas DataFrame 格式

# 遍历检测结果并在图像上绘制边框和标签
for index, row in results_df.iterrows():
    # 获取边框坐标
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

    # 获取类别名称和置信度
    label = f"{row['name']} {row['confidence']:.2f}"

    # 绘制边框
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 绘制标签
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示图像
cv2.imshow("YOLOv5 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果你想将图片保存为文件，请使用以下代码：
cv2.imwrite("car_image_with_detections.jpg", image)
