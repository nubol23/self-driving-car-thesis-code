import pickle
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import torch.nn as nn
import glob
import os
import sys

from scipy import stats

try:
    sys.path.append(glob.glob('/home/nubol23/Desktop/Installers/CARLA_0.9.9.4/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
try:
    sys.path.append('/home/nubol23/Desktop/Installers/CARLA_0.9.9.4/PythonAPI/carla')
except IndexError:
    pass
try:
    sys.path.append('/home/nubol23/Desktop/Codes/Tesis/NNTrain/Drive')
    sys.path.append('/home/nubol23/Desktop/Codes/Tesis/Train/Depth')
    sys.path.append('/home/nubol23/Desktop/Codes/Tesis/NNTrain/Segmentation')
except IndexError:
    pass
    
from utils.sync_mode import CarlaSyncMode
from custom_mobilenet import CustomMobileNet, CustomMobileNetExt
from models import CustomMobilenetDepth
from semseg_model import CustomMobilenetSemseg
from utils.constants import junction_data, contours, directions, weather_idxs

import carla

from collections import Counter
import pygame
import numpy as np
import re
import math
import weakref
import collections
import cv2
from PIL import Image
from priority_queue import PriorityQueue

from numba import njit

def should_quit():
    # función incluida con la API de Carla
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def find_weather_presets():
    # función incluida con la API de Carla
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def img_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array


def show_window(surface, array, pos=(0,0)):
    if len(array.shape) > 2:
        array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    surface.blit(image_surface, pos)


def create_camera(cam_type, vehicle, pos, h, w, lib, world):
    cam = lib.find(f'sensor.camera.{cam_type}')
    cam.set_attribute('image_size_x', str(w))
    cam.set_attribute('image_size_y', str(h))
    camera = world.spawn_actor(
        cam,
        pos,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid)
    return camera

import matplotlib.pyplot as plt

def set_random_weather(world):
    weather_presets = find_weather_presets()
    weather_index = np.random.choice(weather_idxs)
    preset = weather_presets[weather_index]
    world.set_weather(preset[0])
    
def spawn_player(world):
    world_map = world.get_map()
    player = None
    while player is None:
        spawn_points = world_map.get_spawn_points()
        spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
        
        player = world.try_spawn_actor(world.get_blueprint_library().filter('model3')[0], 
                                       spawn_point)
        
    return player
      
def load_network(model_class, path, shape=None, pretrained=None):
    if isinstance(shape, type(None)):
        model = model_class()
    else:
        model = model_class((180, 240), pretrained=False)
    epoch, arch, state_dict = torch.load(path).values()
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    
    return model
    
def take_action(world, junction_data, road_id, lane_id, action, position):
    wp = world.get_map().get_waypoint(position)
    if wp.is_junction:
        junc = wp.get_junction()
        k = (road_id, lane_id, junc.id)
        
        if not k in junction_data.keys():  # Si llega a una intersección por lado desconocido
            action = [0, 0, 0, 1] # sin acción.
        else:
            choice = np.random.choice(junction_data[k])  # Seleccionar una dirección
            if action == [0, 0, 0, 1]:
                if choice == 'l':
                    action = [1, 0, 0, 0]
                elif choice == 'r':
                    action = [0, 1, 0, 0]
                elif choice == 'f':
                    action = [0, 0, 1, 0]
    else:
        road_id, lane_id = wp.road_id, wp.lane_id
        action = [0, 0, 0, 1]
        
    return road_id, lane_id, action
    
def steer_correction(action, mutable_params, threshold):
    if action == [1, 0, 0, 0]:
        action_str = 'left'
        # Si no está en un rango de giro mínimo
        if not mutable_params['s'] < -threshold:
            action_str = 'force right'
            # forzar giro                  
            mutable_params['s'] = -threshold
            # incrementar el angulo conforme pasa el tiempo si no se corrige                        
            mutable_params['s'] -= mutable_params['ac']
            # clip del valor             
            mutable_params['s'] = max(mutable_params['s'], -0.8)
            # acumular el incremento 0.02                      
            mutable_params['ac'] += 0.02          
        else:
            # Si está en rango se reinicia
            mutable_params['ac'] = 0              
        ema_str = ''
    elif action == [0, 1, 0, 0]:
        action_str = 'right'
        # similar al caso de arriba pero al otro lado
        if not mutable_params['s'] > threshold: 
            action_str = 'force right'
            mutable_params['s'] = threshold
            mutable_params['s'] += mutable_params['ac']
            mutable_params['s'] = min(mutable_params['s'], 0.8)
            mutable_params['ac'] += 0.02
        else:
            mutable_params['ac'] = 0
        ema_str = ''
    elif action == [0, 0, 1, 0]:
        action_str = 'forward'
        # Si debe ir recto pero gira a la fuerza
        if not abs(mutable_params['s']) <= threshold:
            # Dar volantazo para corregir la posición               
            mutable_params['s'] = -mutable_params['s']
        # Restaurar acumulador de giro                                
        mutable_params['ac'] = 0                  
        ema_str = ''
    # Está fuera de una intersección
    elif action == [0, 0, 0, 1]:
        # Si ema está en un tiempo t=0                  
        if isinstance(mutable_params['ema'], type(None)):
            # inicializar           
            mutable_params['ema'] = mutable_params['s']             
        else:
            # Dar un paso de EMA
            mutable_params['ema'] = mutable_params['alpha']*mutable_params['s'] + (1-mutable_params['alpha'])*mutable_params['ema']
        # Si tiene poco giro (esta en una recta)    
        if abs(mutable_params['s']) <= threshold:  
            ema_str = 'ema'
            # Aplicar ema
            s = mutable_params['ema']              
        else:
            # Si está en una curva del trayecto que no es intersección
            # No hacer nada aunque se siga actualizando.
            ema_str = ''
        action_str = 'no action'
        # Reiniciar acumulador
        mutable_params['ac'] = 0
        
    return action_str, ema_str # dbg
    
def get_class_semseg(segmentation, class_id):
    seg = ((segmentation == class_id)*255).astype(np.uint8)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((5, 5)))
    seg = cv2.dilate(seg, np.ones((5, 5)), iterations = 2)
    seg = cv2.erode(seg, np.ones((5, 5)), iterations = 1)
    
    return seg
    
def extract_objects_depth(depth_map, contours, veh, poles, ped):
    objects_depth = np.round(depth_map).astype(np.uint8)
    depth_vals_veh = cv2.bitwise_and(objects_depth, veh)
    depth_vals_pol = cv2.bitwise_and(objects_depth, poles)
    objects_depth = cv2.bitwise_or(depth_vals_veh, depth_vals_pol)
    cv2.fillPoly(objects_depth, pts=[contours], color=0)

    return objects_depth

def bounding_box(min_area, signals, directions, x_min_thresh=0):
    pq = PriorityQueue()
    if signals.sum() > 0:  # Comprobar si contiene semáforos
        bbox = find_contours(signals, directions, x_min_thresh)
        for x1, y1, x2, y2 in bbox:
            area = abs(x1 - x2)*abs(y1 - y2)
            if area > min_area and x1 > x_min_thresh:
                pq.push((-area, x1, y1, x2, y2))
    
    valid = True if not pq.empty() and abs(pq.top()[0]) > min_area else False
    
    if valid:
        _, x1, y1, x2, y2 = pq.top()
        x, y = round(x1), round(y1)
        w, h = round(abs(x1 - x2)), round(abs(y1 - y2))
        
        return x, y, w, h
    return None

def cluster(crop, color_code):
    cod_val = 'na'
    inp = np.float32(crop.reshape((-1,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    _, label, center = cv2.kmeans(inp, K, None, criteria,10, cv2.KMEANS_RANDOM_CENTERS)
    res = np.uint8(center[label.flatten()]).reshape((crop.shape))
    res = res * (res > 200)
    rgb_sig = [res[:, :, 0].sum(), res[:, :, 1].sum(), res[:, :, 2].sum()]
    rc, gc, bc = rgb_sig
    if rc > 0 and gc > 0:
        if bc > 0:
            cod_val = 'g'
        else:
            cod_val = color_code[np.argmax(rgb_sig)]
    elif rc > 0 or gc > 0 or bc > 0:
        cod_val = color_code[np.argmax(rgb_sig)]
        
    return cod_val, res, rc, gc, bc
    
@njit
def ff(mat, i, j, x1, y1, x2, y2, directions):
    if (0 <= i < mat.shape[0]) and (0 <= j < mat.shape[1]) and mat[i, j] != 0:
        mat[i, j] = 0
        x1 = min(x1, j)
        y1 = min(y1, i)
        x2 = max(x2, j)
        y2 = max(y2, i)

        for dx, dy in directions:
            x1, y1, x2, y2 = ff(mat, i+dy, j+dx, x1, y1, x2, y2, directions)
    
    return x1, y1, x2, y2

@njit
def find_contours(img, directions, x_min_thresh=0, pad=1, copy=True):
    if copy:
        local_img = img.copy()
    else:
        local_img = img
    boxes = []

    for i in range(0, local_img.shape[0]):
        for j in range(x_min_thresh, local_img.shape[1]):
            if local_img[i, j] != 0:
                x1, y1, x2, y2 = ff(local_img, i, j, np.inf, np.inf, -np.inf, -np.inf, directions)
                boxes.append((int(max(x1-pad, 0)),
                              int(max(y1-pad, 0)), 
                              int(min(x2+pad, local_img.shape[1])),
                              int(min(y2+pad, local_img.shape[0]))))

    # retorna en formato x1, y1, x2, y2
    return boxes

def main():
    actor_list = []
    pygame.init()

    w, h = 240, 180

    display = pygame.display.set_mode((2*w, 2*h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    try:
        client = carla.Client('localhost', 2000)
        
        client.set_timeout(2.0)
        world = client.get_world()
        
        player = spawn_player(world)
        actor_list.append(player)
        set_random_weather(world)

        blueprint_library = world.get_blueprint_library()
        
        font = cv2.FONT_HERSHEY_SIMPLEX

        cam_pos = carla.Transform(carla.Location(x=1.6, z=1.7))
        camera_rgb = create_camera(cam_type='rgb',
                                   vehicle=player,
                                   pos=cam_pos,
                                   h=h, w=w,
                                   lib=blueprint_library,
                                   world=world)
        actor_list.append(camera_rgb)
    
        model_depth = load_network(CustomMobilenetDepth, 
                                   '/home/nubol23/Desktop/Codes/Tesis/Train/Depth/weights/c_mob_20.pth.tar',
                                   (180, 240), False)
                                   
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        depth_preprocess = transforms.Compose([
            transforms.ToTensor(),
            lambda T: T[:3],
            normalize
        ])
        
        model_drive = load_network(CustomMobileNet, 
                                   '/home/nubol23/Desktop/Codes/Tesis/NNTrain/Drive/weights/mob_drive_24.pth.tar')
        
        drive_preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
        
        model_semseg = load_network(CustomMobilenetSemseg,
                                    '/home/nubol23/Desktop/Codes/Tesis/NNTrain/Segmentation/weights/s_mob_7.pth.tar',
                                    (180, 240), False)
        
        with CarlaSyncMode(world, camera_rgb, fps=30) as sync_mode:
            road_id, lane_id = 0, 0
            action = [0, 0, 0, 1]
            
            # ema inicializado en None
            # ac: acumulador del incremento forzado de giro si no se tiene reacción inmediata
            mutable_params = {'ema': None, 'alpha': 0.75, 'ac': 0, 's': None, 't': None, 'b': 0}
            
            # límite del rango para aplicar EMA
            threshold = 0.2
            
            min_area = 95
            x_min_thresh = 120 # Detecciones válidas por la derecha
            color_code = ['r', 'g', 'b']
            
            # Iterando por la simulación
            while True:
                if should_quit():
                    return
                clock.tick()

                # Avanzar un tick de la simulación
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # Convertir a un arreglo BGR
                rgb_arr = img_to_array(image_rgb)

                # Convertir a RGB
                X_img = Image.fromarray(cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB))
                X_drive = drive_preprocess(X_img).view(1, 3, 224, 224).cuda()
                X_depth = depth_preprocess(X_img).view(1, 3, 180, 240).cuda()
                
                position = player.get_location()
                road_id, lane_id, action = take_action(world, junction_data, road_id, lane_id, action, position)
                
                action_tensor = torch.tensor(action).view(1, -1).cuda()
                
                with torch.no_grad():
                    pred_drive = model_drive(X_drive, action_tensor).cpu()
                    depth_map = model_depth(X_depth)[0, 0].cpu().numpy()
                    segmentation = model_semseg(X_drive).cpu().numpy()[0].argmax(axis=0).astype(np.uint8)
                
                # throttle, steer prediction
                mutable_params['t'], mutable_params['s'] = round(float(pred_drive[0, 0]), 3), round(float(pred_drive[0, 1]), 3)
                # Clip de la aceleración para evitar carrera
                mutable_params['t'] = min(mutable_params['t'], 0.5)
                
                # aplicando ema.
                action_str, ema_str = steer_correction(action, mutable_params, threshold)
                
                walkers = get_class_semseg(segmentation, 4)
                vehicles = get_class_semseg(segmentation, 10)
                poles = get_class_semseg(segmentation, 5)
                
                v_boxes = find_contours(vehicles, directions, pad=1, copy=True)
                w_boxes = find_contours(walkers, directions, pad=1, copy=True)
                p_boxes = find_contours(poles, directions, pad=1, copy=True)
                
                objects_depth = extract_objects_depth(depth_map, contours, vehicles, poles, walkers)
                
                # calcular la moda de las profundidades
                if np.sum(objects_depth) != 0:
                    c = sorted(Counter(list(objects_depth.ravel())).items(), key=lambda x: -x[1])
                    if c[0][0] == 0:
                        moda = c[1][0]
                        count = c[1][1]
                    else:
                        moda = c[0][0]
                        count = c[0][1]
                else:
                    moda = np.inf
                    count = np.inf
                
                if moda <= 4 and 40 <= count < np.inf:
                    mutable_params['s'], mutable_params['t'] = 0, 0
                    mutable_params['b'] += 0.10
                    mutable_params['b'] = round(min(max(0, mutable_params['b']), 1), 3)
                else:
                    mutable_params['b'] = 0
                
                signals = get_class_semseg(segmentation, 12)
                
                # caja delimitadora
                box = bounding_box(min_area, signals, directions, x_min_thresh)
                
                code_val = 'na'
                signal_img = np.ones((180, 60, 3))
                rc, gc, bc = 0, 0, 0

                k_img = None
                if isinstance(box, tuple):
                    x, y, w, h = box
                    crop = cv2.resize(cv2.cvtColor(rgb_arr[y:y+h, x:x+w, :], cv2.COLOR_BGR2RGB), None, fx=3, fy=3)
                    
                    # K-MEANS
                    code_val, k_img, rc, gc, bc = cluster(crop, color_code) # -> 60x180
                    
                if code_val == 'r':
                    mutable_params['b'] = 1
                    mutable_params['s'] = 0
                    mutable_params['t'] = 0
                
                player.apply_control(carla.VehicleControl(throttle= mutable_params['t'], 
                                                          steer= mutable_params['s'], 
                                                          brake= mutable_params['b']))
                
                line1 = f"t: {round(mutable_params['t'], 2)} b: {round(mutable_params['b'], 2)} s: {round(mutable_params['s'], 2)} color: {code_val}"
                line2 = f"moda: {moda}, r: {int(rc>0)} g: {int(gc>0)} b: {int(bc>0)}"
                line3 = f"act: {action_str} {action}"                
                rgb_arr = cv2.resize(rgb_arr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                cv2.putText(rgb_arr, line1, (10, 30), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(rgb_arr, line1, (10, 30), font, 0.6, (156, 237, 58), 1, cv2.LINE_AA)
                cv2.putText(rgb_arr, line2, (10, 60), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(rgb_arr, line2, (10, 60), font, 0.6, (156, 237, 58), 1, cv2.LINE_AA)
                cv2.putText(rgb_arr, line3, (10, 90), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(rgb_arr, line3, (10, 90), font, 0.6, (156, 237, 58), 1, cv2.LINE_AA)
                
                if not isinstance(box, type(None)):
                    color = (58, 58, 237) if code_val == 'r' else (58, 237, 67)
                    cv2.rectangle(rgb_arr, (box[0]*2, box[1]*2), ((box[0]+box[2])*2, (box[1]+box[3])*2), color, 2)
                    
                for x1, y1, x2, y2 in v_boxes:
                    # bgr color
                    cv2.rectangle(rgb_arr, (x1*2, y1*2), (x2*2, y2*2), (237, 152, 58), 2)
                for x1, y1, x2, y2 in p_boxes:
                    cv2.rectangle(rgb_arr, (x1*2, y1*2), (x2*2, y2*2), (115, 58, 237), 2)
                for x1, y1, x2, y2 in w_boxes:
                    cv2.rectangle(rgb_arr, (x1*2, y1*2), (x2*2, y2*2), (58, 161, 237), 2)
                    
                show_window(display, rgb_arr)
                # graficar el semáforo segmentado
                if not isinstance(k_img, type(None)):
                    k_img = cv2.cvtColor(k_img, cv2.COLOR_RGB2BGR)
                    k_img = cv2.resize(k_img, (31, 55), interpolation=cv2.INTER_CUBIC)
                    show_window(display, 
                        k_img,
                        (rgb_arr.shape[1]-k_img.shape[1], 0))
                pygame.display.flip()
    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
