import os
import sys
import numpy as np
import argparse
from PIL import Image
import time
import scipy.io as scio
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup
from UR_Robot import UR_Robot
import socket
import cv2 as cv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
import pyrealsense2 as rs
import json
import png

from grounding_dino_demo import load_model, parse_opt, load_image, compute_xy, get_grounding_output, plot_boxes_to_image

import pyaudio
import wave
import whisper

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/')
parser.add_argument('--checkpoint_path', default='/home/llin/Grasp/logs/log_realsense/minkuresunet_epoch10.tar')
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='/home/llin/Grasp/logs/log_realsense/dump_epoch10/')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=100000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=-1,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--grasp', action='store_true', default=False)
parser.add_argument('--use_camera', type=bool, default=False, help='Whether use camera')
parser.add_argument('--speech', type=bool, default=False, help='Whether use microphone to speech')
parser.add_argument('--detect', type=bool, default=True, help='Whether use Grounding DINO to detect')
parser.add_argument('--scene', type=str, default='0188')
parser.add_argument('--index', type=str, default='0000')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


def data_process():
    root = cfgs.dataset_root
    camera_type = cfgs.camera

    depth = np.array(Image.open('./pics/depth1.jpg'))
    rgb = np.array(Image.open('./pics/rgb1.jpg'), dtype=np.float32) / 255.0

    camera = CameraInfo(1280.0, 720.0, 912.494, 912.934, 656.01, 371.285, 1000.0)
    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    depth_mask = (depth > 0)
    mask = depth_mask
    cloud_masked = cloud[mask]
    rgb = rgb[mask]

    # sample points random
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    rgb = rgb[idxs]


    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'rgb':rgb.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled).astype(np.float32),
                }
    return ret_dict


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference(data_input):
    batch_data = minkowski_collate_fn([data_input])
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

    net.eval()
    tic = time.time()

    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    # Forward pass
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)

    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)

    # collision detection
    if cfgs.collision_thresh > 0:
        cloud = data_input['point_clouds']
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]

    # save grasps
    save_dir = os.path.join(cfgs.dump_dir, scene_id, cfgs.camera)
    save_path = os.path.join(save_dir, cfgs.index + '.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gg.save_npy(save_path)

    toc = time.time()
    print('inference time: %fs' % (toc - tic))


CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors',
    'teddy bear', 'hair drier', 'toothbrush']

def speech():
    # 实例化一个PyAudio对象
    pa = pyaudio.PyAudio()
    # 打开声卡，设置 采样深度为16位、声道数为2、采样率为16、输入、采样点缓存数量为2048
    stream = pa.open(format=pyaudio.paInt16, channels=2, rate=16000, input=True, frames_per_buffer=2048)
    # 新建一个列表，用来存储采样到的数据
    record_buf = [] 
    count = 0
    while count < 8*5:
        audio_data = stream.read(2048)      # 读出声卡缓冲区的音频数据
        record_buf.append(audio_data)       # 将读出的音频数据追加到record_buf列表
        count += 1
        print('*')
    wf = wave.open('01.wav', 'wb')          # 创建一个音频文件，名字为“01.wav"
    wf.setnchannels(2)                      # 设置声道数为2
    wf.setsampwidth(2)                      # 设置采样深度为
    wf.setframerate(16000)                  # 设置采样率为16000
    # 将数据写入创建的音频文件
    wf.writeframes("".encode().join(record_buf))
    # 写完后将文件关闭
    wf.close()
    # 停止声卡
    stream.stop_stream()
    # 关闭声卡
    stream.close()
    # 终止pyaudio
    pa.terminate()

 
def speech_recognition(speech_file, model):
    # whisper
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(speech_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    speech_language = max(probs, key=probs.get)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    speech_text = result.text
    return speech_text, speech_language

# 该函数作用是获取对齐后的图像（rgb和深度图对齐）
def get_aligned_images():
	# 获取一帧图像
    frames = pipeline.wait_for_frames()
    # 将图像对齐
    aligned_frames = align.process(frames)
    # 获取对齐后的深度图
    aligned_depth_frame = aligned_frames.get_depth_frame()
    # 深度相关参数
    # depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    color_frame = aligned_frames.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }
    with open('./intrinsics.json', 'w') as fp:
        json.dump(camera_parameters, fp)
    # 获取16位深度图
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # 转为8位深度图
    # 2^8为256，2^16为65536， 2^8/2^16约等于0.004，alpha=0.004，cv2.convertScaleAbs作用是所有像素点的值乘以alpha
    depth_image_8bit = cv.convertScaleAbs(depth_image, alpha=0.004)
    pos=np.where(depth_image_8bit==0)
    depth_image_8bit[pos]=255
    # 获取彩色图
    color_image = np.asanyarray(color_frame.get_data())
    # 获取到图像中心的距离
    d = aligned_depth_frame.get_distance(320, 240)
    return color_image, depth_image,

if __name__ == '__main__':

    if cfgs.use_camera:
        # 开启摄像头通信管道
        pipeline = rs.pipeline()
        # 获取配置
        config = rs.config()
        # 设置深度图为16位，fps为30，尺寸640x480
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        # 设置RGB图为bgr格式（每个通道8位共24位），fps为30，尺寸640x480
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # 开启通道
        profile = pipeline.start(config)
        # 获取颜色对齐
        align_to = rs.stream.color
        align = rs.align(align_to)
        depth_intrin = None

        while 1:
            n=0
            # 获取rgb和深度图
            rgb, depth = get_aligned_images()
            cv.imshow('RGB image',rgb)
            key = cv.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                pipeline.stop()
                break
            elif key==ord('s'):
                n=n+1
                # 保存rgb图
                cv.imwrite('/home/llin/graspness_implementation/pics/rgb' + str(n)+'.jpg',rgb)
                # 保存16位深度图
                with open('/home/llin/graspness_implementation/pics/depth' + str(n) + ".jpg", 'wb') as f:
                    writer = png.Writer(width=depth.shape[1], height=depth.shape[0],
                                        bitdepth=16, greyscale=True)
                    zgray2list = depth.tolist()
                    writer.write(f, zgray2list)
        cv.destroyAllWindows()
    
    # GroundingDINO
    if cfgs.detect:
        if cfgs.speech:
            speech()
        recognition_model = whisper.load_model("base")
        speech_text, speech_language = speech_recognition("./01.wav",recognition_model)
        print("speech_text",speech_text,"speech_laguage",speech_language)
        object_prompt = speech_text.casefold().strip().split(' ')[-1]
        detect_args = parse_opt()
        detect_args.text_prompt = "mouse"
        #use speech
        # detect_args.text_prompt = object_prompt
        config_file = detect_args.config  # change the path of the model config file
        grounded_checkpoint = detect_args.grounded_checkpoint  # change the path of the model
        image_path = detect_args.input_image
        text_prompt = detect_args.text_prompt
        output_dir = detect_args.output_dir
        box_threshold = detect_args.box_threshold
        text_threshold = detect_args.box_threshold
        device = detect_args.device

        os.makedirs(output_dir, exist_ok=True)
        # load image
        image_pil, image = load_image(image_path)
        model = load_model(config_file, grounded_checkpoint, device=device)
        boxes_filt, pred_phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device=device)
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],
            "labels": pred_phrases,
        }
        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        image_with_box.save(os.path.join(output_dir, "grounding_dino_output.jpg"))
        box = compute_xy(image_pil, pred_dict)
        camera_ifo = CameraInfo(1280.0, 720.0, 912.494, 912.934, 656.01, 371.285, 1000.0)
        points_xmin = (box[0] - camera_ifo.cx) / camera_ifo.fx
        points_xmax = (box[2] - camera_ifo.cx) / camera_ifo.fx
        points_ymin = (box[1] - camera_ifo.cy) / camera_ifo.fy
        points_ymax = (box[3] - camera_ifo.cy) / camera_ifo.fy
    

    scene_id = 'scene_' + cfgs.scene
    index = cfgs.index
    data_dict = data_process()

    if cfgs.infer:
        inference(data_dict)
    if cfgs.vis:
        import copy
        pc = data_dict['point_clouds']
        color = data_dict['rgb']
        gg = np.load(os.path.join(cfgs.dump_dir, scene_id, cfgs.camera, cfgs.index + '.npy'))
        gg = GraspGroup(gg)
        gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 30:
            gg = gg[:30]
        if cfgs.detect:
            for i in range(30):
                camera_position = gg[i].translation.reshape(3,1)
                print("camera_position",camera_position)
                camera_x = camera_position[0]
                camera_y = camera_position[1]
                camera_z = camera_position[2]

                print(points_xmin,camera_x /camera_z)
                if camera_x /camera_z < points_xmin or camera_x /camera_z > points_xmax:
                    continue
                if camera_y /camera_z < points_ymin or camera_y /camera_z > points_ymax:
                    continue
                gg = gg[i:i+1]
                break

        grippers = gg.to_open3d_geometry_list()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, *grippers])

    if cfgs.grasp:
        pc = data_dict['point_clouds']
        color = data_dict['rgb']
        gg = np.load(os.path.join(cfgs.dump_dir, scene_id, cfgs.camera, cfgs.index + '.npy'))
        gg = GraspGroup(gg)
        gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 30:
            gg = gg[:30]

        ur_robot = UR_Robot(tcp_host_ip="192.168.1.101", tcp_port=30003, workspace_limits=None)

        end2camera = np.array([[1,0,0,-0.03],[0,1,0,0.9],[0,0,1,0]])

        ur_robot.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ur_robot.tcp_socket.connect((ur_robot.tcp_host_ip, ur_robot.tcp_port))
        state_data = ur_robot.tcp_socket.recv(1500)
        actual_tool_positions = ur_robot.parse_tcp_state_data(state_data, 'cartesian_info')
        print("actual_tool_positions",actual_tool_positions)


        if cfgs.detect:
            grasp_i = 0
            for i in range(30):
                camera_position = gg[i].translation.reshape(3,1)
                if camera_position[0] /camera_position[2] < points_xmin or camera_position[0] /camera_position[2] > points_xmax:
                    continue
                if camera_position[1] /camera_position[2] < points_ymin or camera_position[1] /camera_position[2] > points_ymax:
                    continue
                else:
                    grasp_i = i
                    break
            
            camera_position[0] = camera_position[0]-0.03
            camera_position[1] = camera_position[1]+0.09
            camera_position[2] = camera_position[2]-0.12
            camera_rotation = gg[grasp_i].rotation_matrix

            pre_position = np.array([0,0,-0.1]).reshape(3,1)
            pre_position = np.dot(camera_rotation,pre_position) + camera_position
  
            camera2robot_posiontion = actual_tool_positions[:3]
            camera2robot_rota_vec = actual_tool_positions[3:]

            camera2robot_R = ur_robot.rotating_vector2R(camera2robot_rota_vec)
            # print("R",camera2robot_R)
            camera2robot = np.zeros((4,4))
            camera2robot[0:3,0:3] = camera2robot_R
            camera2robot[0:3,3:] = camera2robot_posiontion.reshape(3,1)
            camera2robot[3,3] = 1

            robot_position = np.dot(camera2robot[0:3,0:3],camera_position) + camera2robot[0:3,3:]
            pre_position = np.dot(camera2robot[0:3,0:3],pre_position) + camera2robot[0:3,3:]
            robot_position = list(pre_position.reshape(3)) + list(robot_position.reshape(3))
            print("robot_position",robot_position)
            robot_rotation = np.dot(camera2robot[0:3,0:3],camera_rotation)

            rpy = list(ur_robot.R2rpy(robot_rotation))
    

            ur_robot.grasp(position = robot_position,rpy = rpy)


        else:
            camera_position = gg[0].translation.reshape(3,1)
            camera_position[0] = camera_position[0]-0.03
            camera_position[1] = camera_position[1]+0.09
            camera_position[2] = camera_position[2]-0.12
            camera_rotation = gg[0].rotation_matrix
            
            pre_position = np.array([0,0,-0.1]).reshape(3,1)
            pre_position = np.dot(camera_rotation,pre_position) + camera_position

            camera2robot_posiontion = actual_tool_positions[:3]
            camera2robot_rota_vec = actual_tool_positions[3:]

            camera2robot_R = ur_robot.rotating_vector2R(camera2robot_rota_vec)
            # print("R",camera2robot_R)
            camera2robot = np.zeros((4,4))
            camera2robot[0:3,0:3] = camera2robot_R
            camera2robot[0:3,3:] = camera2robot_posiontion.reshape(3,1)
            camera2robot[3,3] = 1

            robot_position = np.dot(camera2robot[0:3,0:3],camera_position) + camera2robot[0:3,3:]
            pre_position = np.dot(camera2robot[0:3,0:3],pre_position) + camera2robot[0:3,3:]
            robot_position = list(pre_position.reshape(3)) + list(robot_position.reshape(3))
            print("robot_position",robot_position)
            robot_rotation = np.dot(camera2robot[0:3,0:3],camera_rotation)

            rpy = list(ur_robot.R2rpy(robot_rotation))
    
            ur_robot.grasp(position = robot_position,rpy = rpy)

