import os.path as osp

import cv2


data_dir = "/home/deepixel/GITHUB/data"
result_dir = "/home/deepixel/GITHUB/synergize-motion-appearance/result_videos"
result_video_name = "long_hair4_608434_Business_Meeting_1920x1080_crop2"

driving_video_path = osp.join(data_dir, "driving_video", "608434_Business_Meeting_1920x1080_crop2.mp4")
result_video_path = osp.join(result_dir, result_video_name+'.mp4')

cap1 = cv2.VideoCapture(driving_video_path)
cap2 = cv2.VideoCapture(result_video_path)

if (cap1.isOpened() == False) or (cap2.isOpened() == False):
    print("Error opening video stream or file")

fps = int(cap1.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out_video = cv2.VideoWriter(osp.join(result_dir, f'{result_video_name}_comb.mp4'), fourcc, fps, (256+256, 256))
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2: break

    combined_frame = cv2.hconcat([frame1, frame2])

    out_video.write(combined_frame)

cap1.release()
cap2.release()
out_video.release()
cv2.destroyAllWindows()
print('done')