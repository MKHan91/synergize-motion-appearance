import os
import os.path as osp
import cv2
import argparse
import imageio
import ffmpeg
import torch
import yaml
import numpy as np
import sys
sys.path.insert(0, ".")

from copy import deepcopy
from tqdm.auto import tqdm
from shutil import copyfileobj
from scipy.spatial import ConvexHull
from tempfile import NamedTemporaryFile
from torchvision.transforms.functional import normalize

from basicsr.archs import build_network
from basicsr.utils.options import ordered_yaml
from basicsr.utils import img2tensor, tensor2img
import matplotlib.pyplot as plt


def visualize_motion(flow):
    import numpy as np
    import cv2
    
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    
    if flow.ndim == 5:   # [B, K, H, W, 2]
        flow = flow[0, 0]  
    elif flow.ndim == 4: # [B, H, W, 2]
        flow = flow[0]

    # flow: [H, W, 2] numpy array
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]

    # mag: 움직임이 큰 곳은 밝게, 작은 곳은 어둡게 표현
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 1] = 255
    # hsv[..., 0] = 0
    hsv[..., 1] = 0
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)




# region - DRAW KPTS
def draw_keypoints_on_face(image, keypoints, save_path=None, show_numbers=False):
    """
    68개의 keypoints를 얼굴 이미지에 그리는 함수
    
    Args:
        image: 얼굴 이미지 (numpy array)
        keypoints: (68, 2) 형태의 keypoint 좌표
        save_path: 저장할 경로 (optional)
        show_numbers: keypoint 번호를 표시할지 여부
    """
    # 이미지 복사본 생성
    img_copy = image.copy()
    
    # 68개 keypoint의 각 영역별 색상 정의
    colors = {
        'jaw': (0, 255, 0),           # 초록 (0-16)
        'right_eyebrow': (255, 0, 0), # 빨강 (17-21)
        'left_eyebrow': (255, 0, 0),  # 빨강 (22-26)
        'nose': (0, 0, 255),          # 파랑 (27-35)
        'right_eye': (255, 255, 0),   # 노랑 (36-41)
        'left_eye': (255, 255, 0),    # 노랑 (42-47)
        'mouth': (255, 0, 255)        # 마젠타 (48-67)
    }
    
    # 각 영역별 keypoint 범위
    regions = {
        'jaw': range(0, 17),
        'right_eyebrow': range(17, 22),
        'left_eyebrow': range(22, 27),
        'nose': range(27, 36),
        'right_eye': range(36, 42),
        'left_eye': range(42, 48),
        'mouth': range(48, 68)
    }
    
    # keypoints 그리기
    for region_name, point_range in regions.items():
        color = colors[region_name]
        for i in point_range:
            if i < len(keypoints):
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                cv2.circle(img_copy, (x, y), 2, color, -1)
                
                # 번호 표시 (선택사항)
                if show_numbers:
                    cv2.putText(img_copy, str(i), (x+3, y-3), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # 연결선 그리기 (윤곽선)
    # draw_connections(img_copy, keypoints, colors)
    
    # 결과 표시
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title('68 Facial Keypoints')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()
    
    return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)


# region - relative motion
def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False, adjust_shape_movement=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def load_network(net, load_path, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    print(f'Loading {net.__class__.__name__} model from {load_path}.')
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)

    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
            print('Loading: params_ema does not exist, use params.')
        load_net = load_net[param_key]

    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)
    return net


# region - BEST FRAME
def find_best_frame(source, driving, cpu=False):
    import face_alignment  # type: ignore (local file)
    from scipy.spatial import ConvexHull

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    
    # draw_keypoints_on_face(driving[0], kp_source, save_path='driving_keypoints.png', show_numbers=False)
    
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    
    kpts_images = []
    for i, image in tqdm(enumerate(driving)):
        try:
            kp_driving = fa.get_landmarks(255 * image)[0]
            # # --------------- added ---------------
            # kpts_img = draw_keypoints_on_face(image, kp_driving, save_path=False, show_numbers=False)
            # comb_kpts_img = np.hstack((cv2.cvtColor(image, cv2.COLOR_BGR2RGB), kpts_img))
            # kpts_images.append(comb_kpts_img)
            # # ---------------------------------------------
            
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except Exception as e:
            print(e)
    
    # # --------------- added ---------------
    # kpts_video = np.stack(kpts_images, axis=0)
    # imageio.mimwrite('kpts_video.mp4', kpts_video, fps=29)
    # # ---------------------------------------------
    
    return frame_num



# region - ani
def make_animation(source_image, driving_video, net_g, motion_estimator, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        driving_imgs = []
        flows = []
        
        source_img = source_image.unsqueeze(0)
        driving_video = [frame.unsqueeze(0) for frame in driving_video]
        if not cpu:
            source_img = source_img.cuda()
            driving_video[0] = driving_video[0].cuda()
        
        kp_source = motion_estimator.estimate_kp(source_img)
        kp_driving_initial = motion_estimator.estimate_kp(driving_video[0])

        for frame_idx in tqdm(range(len(driving_video))):
            driving_frame = driving_video[frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = motion_estimator.estimate_kp(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale
                                )
            
            dense_motion = motion_estimator.estimate_motion_w_kp(kp_source=kp_source, kp_driving=kp_norm, source_image=source_img)

            # # --------------------------- added ---------------------------
            # deformation = dense_motion['deformation']
            # flow = visualize_motion(deformation)
            # cv2.imwrite(f"motion_flow{frame_idx}.png", flow)
            # # ------------------------------------------------------
            
            out_dict = net_g(source_img, dense_motion, w=1, inference=True)
            
            # for idx, deformation in enumerate(out_dict['res_deform_list']):
            #     flow = visualize_motion(deformation)
            #     cv2.imwrite(f"motion_flow{idx}.png", flow)
            deformation = out_dict['res_deform_list'][-1][:, :, :, :2]
            # deformation2 = out_dict['deformation_list'][-2][:, :, :, :2]
            # deformation = deformation1 + deformation2
            # flow = tensor2img(deformation.detach().cpu(), rgb2bgr=False, min_max=(-1,1))
            flow = visualize_motion(deformation)
            # cv2.imwrite(f"motion_flow.png", flow)
            flows.append(flow)
            
            predictions.append(tensor2img([out_dict['out'].detach().cpu()], rgb2bgr=False, min_max=(-1, 1)))
            driving_imgs.append(tensor2img([driving_video[frame_idx].detach().cpu()], rgb2bgr=False, min_max=(-1, 1)))

    return predictions, driving_imgs, flows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='/home/dev/synergize-motion-appearance/options/test.yml', help="path to config")

    parser.add_argument("--source_image", default='/home/dev/synergize-motion-appearance/data/source/long_hair4.jpg', help="path to source image")
    parser.add_argument("--driving_video", default='/home/dev/synergize-motion-appearance/data/driving_video/upper_body_005_crop.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='/home/dev/synergize-motion-appearance/synergize-motion-appearance/result_videos', help="path to output")
    parser.add_argument("--visual_video", default=None, help="path to visual output")

    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")
    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None, help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.add_argument("--audio", dest="audio", action="store_true", help="copy audio to output from the driving video" )

    parser.set_defaults(relative=True)
    parser.set_defaults(adapt_scale=True)
    parser.set_defaults(find_best_frame=True)
    parser.set_defaults(adapt_scale=False)
    parser.set_defaults(audio_on=False)

    opt = parser.parse_args()
    with open(opt.config, mode='r') as f:
        Loader, _ = ordered_yaml()
        config = yaml.load(f, Loader=Loader)

    source_image = cv2.imread(opt.source_image, cv2.IMREAD_COLOR)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    # prepare data
    source_image = cv2.resize(source_image, (256, 256), interpolation=cv2.INTER_LINEAR) 
    driving_video = [cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR) for frame in driving_video]

    source = img2tensor(source_image.astype(np.float32) / 255., bgr2rgb=True, float32=True)
    driving = [img2tensor(frame.astype(np.float32) / 255., bgr2rgb=False, float32=True) for frame in driving_video]

    normalize(source, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    for i in range(len(driving)):
        normalize(driving[i], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)

    # load model
    net_g = build_network(config['network_g'])
    load_path = config['path'].get('pretrain_network_g', None)
    if load_path is not None:
        param_key = config['path'].get('param_key_g', 'params')
        net_g = load_network(net_g, load_path, config['path'].get('strict_load_g', True), param_key)
    net_g.eval()
    
    motion_estimator = build_network(config['network_motion_estimator'])
    load_path = config['path'].get('pretrain_network_motion_estimator', None)
    if load_path is not None:
        param_key = config['path'].get('param_key_m', 'params')
        motion_estimator = load_network(motion_estimator, load_path, config['path'].get('strict_load_motion_estimator', True), param_key)
    motion_estimator.eval()

    if not opt.cpu:
        net_g.cuda()
        motion_estimator.cuda()

    # animate
    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB), [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in driving_video], cpu=opt.cpu)
        print("Best frame: " + str(i))
        driving_forward = driving[i:]
        driving_backward = driving[:(i+1)][::-1]

        # 앞으로 진행하는 프레임
        predictions_forward, driving_forward_list, flow_forward_list = make_animation(source, driving_forward, net_g, motion_estimator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        # 역으로 진행하는 프레임 (모션 보정을 위해)
        predictions_backward, driving_backward_list, flow_backward_list = make_animation(source, driving_backward, net_g, motion_estimator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)

        predictions = predictions_backward[::-1] + predictions_forward[1:]
        drivings = driving_backward_list[::-1] + driving_forward_list[1:]
        flows = flow_forward_list[::-1] + flow_backward_list[1:]
    else:
        predictions, drivings = make_animation(source, driving, net_g, motion_estimator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    
    # added
    # 최종 warping된 motion flow
    imageio.mimsave("./motionflow.gif", flows, fps=10)
    
    
    # save video
    source_name = osp.split(opt.source_image)[-1][:-4]
    video_name = osp.split(opt.driving_video)[-1][:-4]
    opt.result_video = osp.join(opt.result_video, f"{source_name}_{video_name}.mp4")
    imageio.mimwrite(opt.result_video, predictions, fps=fps)

    if opt.visual_video is not None:
        source = tensor2img([source.detach().cpu()], rgb2bgr=False, min_max=(-1, 1))
        visual = []
        for i in range(len(predictions)):
            vis = np.concatenate((source, drivings[i], predictions[i]), axis=1)
            visual.append(vis)
        imageio.mimwrite(opt.visual_video, visual, fps=fps)

    # copy audio
    if opt.audio:
        try:
            with NamedTemporaryFile(suffix=os.path.splitext(opt.result_video)[1]) as output:
                ffmpeg.output(ffmpeg.input(opt.result_video).video, ffmpeg.input(opt.driving_video).audio, output.name, c='copy').run()
                with open(opt.result_video, 'wb') as result:
                    copyfileobj(output, result)
        except ffmpeg.Error:
            print("Failed to copy audio: the driving video may have no audio track or the audio format is invalid.")
        
        if opt.visual_video is not None:
            try:
                with NamedTemporaryFile(suffix=os.path.splitext(opt.visual_video)[1]) as output:
                    ffmpeg.output(ffmpeg.input(opt.visual_video).video, ffmpeg.input(opt.driving_video).audio, output.name, c='copy').run()
                    with open(opt.visual_video, 'wb') as result:
                        copyfileobj(output, result)
            except ffmpeg.Error:
                print("Failed to copy audio: the driving video may have no audio track or the audio format is invalid.")
