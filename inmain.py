import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.chdir('./')
CODE_DIR = 'VToonify'
device = 'cuda'

# os.chdir(f'./{CODE_DIR}')
MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), CODE_DIR, 'checkpoint')
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), CODE_DIR, 'data')
OUT_DIR = os.path.join(os.path.dirname(os.getcwd()), CODE_DIR, 'output')

# import sys
# sys.path.append(".")
# sys.path.append("..")

# import argparse
import numpy as np
import cv2
import dlib
import torch
from torchvision import transforms
# import torchvision
import torch.nn.functional as F
# from tqdm import tqdm
# import matplotlib.pyplot as plt
from model.vtoonify import VToonify
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import save_image, load_image, visualize, load_psp_standalone, get_video_crop_parameter, tensor2cv2
import imageio
import mediapipemodel as med
import threading

def get_download_model_command(file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../checkpoint/. """
    current_directory = os.getcwd()
    save_path = MODEL_DIR
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url

MODEL_PATHS = {
    "encoder": {"id": "1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej", "name": "encoder.pt"},
    "faceparsing": {"id": "1jY0mTjVB8njDh6e0LP_2UxuRK3MnjoIR", "name": "faceparsing.pth"},
    "arcane_exstyle": {"id": "1TC67wRJkdmNRZTqYMUEFkrhWRKKZW40c", "name": "exstyle_code.npy"},
    "caricature_exstyle": {"id": "1xr9sx_WmRYJ4qHGTtdVQCSxSo4HP3-ip", "name": "exstyle_code.npy"},
    "cartoon_exstyle": {"id": "1BuCeLk3ASZcoHlbfT28qNru4r5f-hErr", "name": "exstyle_code.npy"},
    "pixar_exstyle": {"id": "1yTaKuSrL7I0i0RYEEK5XD6GI-y5iNUbj", "name": "exstyle_code.npy"},
    "arcane000": {"id": "1pF4fJ8acmawMsjjXo4HXRIOXeZR8jLVh", "name": "generator.pt"},
    "arcane077": {"id": "16rLTF2oC0ZeurnM6hjrfrc8BxtW8P8Qf", "name": "generator.pt"},
    "caricature039": {"id": "1C1E4WEoDWzl0nAxR9okKffFmlMOENbeF", "name": "generator.pt"},
    "caricature068": {"id": "1B1ko1x8fX2aJ4BYCL12AnknVAi3qQc8W", "name": "generator.pt"},
    "cartoon026": {"id": "1YJYODh_vEyUrL0q02okjcicpJhdYY8An", "name": "generator.pt"},
    "cartoon299": {"id": "101qMUMfcI2qDxEbfCBt5mOg2aSqdTaIt", "name": "generator.pt"},
    "pixar052": {"id": "16j_l1x0DD0PjwO8YdplAk69sh3-v95rr", "name": "generator.pt"},
    "cartoon": {"id": "11s0hwhZWTLacMAzZH4OU-o3Qkp54h30J", "name": "generator.pt"},
}

style_type = "cartoon026" #@param ["cartoon026", "cartoon299", "arcane000", "arcane077", "pixar052", "caricature039", "caricature068"]

"""
cartoon026:      balanced 
cartoon299:      big eyes 
arcane000:       for female 
arcane077:       for male 
pixar052:                  
caricature039:   big mouth 
caricature068:   balanced  
"""

path1 = MODEL_PATHS["encoder"]
download_command1 = get_download_model_command(file_id=path1["id"], file_name=path1["name"])
# print('download_command1',download_command1)
path = MODEL_PATHS["faceparsing"]
download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])


# download vtoonify
path = MODEL_PATHS[style_type]
download_command2 = get_download_model_command(file_id=path["id"], file_name = style_type + '_' + path["name"])

# download extrinsic style code
path = MODEL_PATHS[style_type[:-3]+'_exstyle']
download_command3 = get_download_model_command(file_id=path["id"], file_name = style_type[:-3] + '_' + path["name"])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

vtoonify = VToonify(backbone='dualstylegan')
vtoonify.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, style_type + '_generator.pt'), map_location=lambda storage, loc: storage)[
        'g_ema'])
vtoonify.to(device)

parsingpredictor = BiSeNet(n_classes=19)
parsingpredictor.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, 'faceparsing.pth'), map_location=lambda storage, loc: storage))
parsingpredictor.to(device).eval()

modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
if not os.path.exists(modelname):
    import wget, bz2
    wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname + '.bz2')
    zipfile = bz2.BZ2File(modelname + '.bz2')
    data = zipfile.read()
    open(modelname, 'wb').write(data)
landmarkpredictor = dlib.shape_predictor(modelname)

pspencoder = load_psp_standalone(os.path.join(MODEL_DIR, 'encoder.pt'), device)

exstyles = np.load(os.path.join(MODEL_DIR, style_type[:-3] + '_exstyle_code.npy'), allow_pickle='TRUE').item()
stylename = list(exstyles.keys())[int(style_type[-3:])]
exstyle = torch.tensor(exstyles[stylename]).to(device)
with torch.no_grad():
    exstyle = vtoonify.zplus2wplus(exstyle)
    exstyle7=exstyle[:, :7]
print('Model successfully loaded!')
kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])
handbod = med.handDetctor()
global inputs
import base64
from flask import request,Flask
from flask_cors import *
app = Flask(__name__)
CORS(app, supports_credentials=True)
import json
print('All successfully loaded!')
def thread_body(frame):
    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
    x = transform(frame).unsqueeze(dim=0).to(device)
    x_p = F.interpolate(
        parsingpredictor(2 * (F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0],
        scale_factor=0.5, recompute_scale_factor=False).detach()
    global inputs
    inputs = torch.cat((x, x_p / 16.), dim=1)
# 换动漫脸主要算法
def process(frame):
    thread = threading.Thread(target=thread_body, kwargs={'frame':frame})
    thread.start()
    height,width = frame.shape[0],frame.shape[1]
    lm_eye_left, lm_eye_right, mouth_left, mouth_right = handbod.findPositionface(frame,height,width)
    with torch.no_grad():
        try:
            s_w = vtoonify.zplus2wplus(pspencoder(transform(align_face(frame, mouth_left, mouth_right, lm_eye_left, lm_eye_right)).unsqueeze(dim=0).to(device)))
            s_w[:, :7] = exstyle7
            y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s=0.5)
            y_tilde = torch.clamp(y_tilde, -1, 1)
            tmp = ((y_tilde[0].detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
            tmp = cv2.resize(tmp, (640, 480))
            return tmp
        except:
            return frame

@app.route('/cartoon', methods=['POST'])
def image2image():
    try:
        file = request.files.get("image")
        if file is not None:
            image = imageio.imread(file)
        else:
            return '上传文件失败'
    except:
        return 'detect error', 400
    frame = image
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    imgresult = process(frame)
    cv2.imwrite('reuslt.jpg', imgresult)
    return str(imgresult)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8070, debug=True)



