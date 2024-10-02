import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# 加载 MoveNet Multi-Person Pose Estimation 模型
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']


# 绘制关键点
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)


# 绘制连线
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


# 连接点的定义
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}


# 遍历每个人并绘制关键点和连线
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


# 读取图像
image_path = "move.jpg"  # 替换为你的图像路径
frame = cv2.imread(image_path)

if frame is None:
    raise ValueError("未能读取图像，请检查文件路径！")

# 调整图像大小并准备输入数据
img = cv2.resize(frame, (640, 384))
img = np.expand_dims(img, axis=0)
img = np.array(img, dtype=np.int32)

# 进行姿态估计
outputs = movenet(tf.convert_to_tensor(img))
keypoints_with_scores = outputs['output_0'].numpy()[:,:,:51].reshape((6, 17, 3))

# 绘制关键点和连接
loop_through_people(frame, keypoints_with_scores, EDGES, 0.3)

# 保存处理后的图片
cv2.imwrite("pose_estimation_result.jpg", frame)
print("图像已保存为 pose_estimation_result.jpg")
