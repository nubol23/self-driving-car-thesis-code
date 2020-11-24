import glob
import os
import sys

try:
    sys.path.append(glob.glob('/home/nubol23/Desktop/Installers/CARLA_0.9.9.4/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print('No se pudo cargar libcarla')
try:
    sys.path.append('/home/nubol23/Desktop/Installers/CARLA_0.9.9.4/PythonAPI/carla')
except IndexError:
    print('No se pudo cargar Carla')

import carla
import random
import pygame
import numpy as np
import queue
import re
import math
import weakref
import collections
import cv2
import pandas as pd


class CarlaSyncMode(object):
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class World(object):
    """ Class representing the surrounding environment """
    def __init__(self, carla_world):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.recording_enabled = False
        self.recording_start = 0

        """Restart the world"""
        blueprint = self.world.get_blueprint_library().filter('model3')[0]
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        print("Spawning the player")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player)
        self.gnss_sensor = GnssSensor(self.player)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.player.get_world().set_weather(preset[0])

    def random_weather(self):
        self._weather_index = random.randint(0, len(self._weather_presets)-1)
        print(len(self._weather_presets), 'WEATHER:', self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.player.get_world().set_weather(preset[0])

    def set_weather(self, index: int):
        preset = self._weather_presets[index]
        self.player.get_world().set_weather(preset[0])

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


class CollisionSensor(object):
    """ Class for collision sensors"""
    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]


def draw_image(surface, image, pos=(0, 0), blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, pos)


def img_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array


def show_window(surface, array):
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))


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


def main():
    actor_list = []
    pygame.init()
    world = None

    base_path = '/home/nubol23/Documents/DriveDatasetStableExtra/Train'
    # n_sim = len(os.listdir(f'{base_path}/Images/')) + 20
    n_sim = 36
    os.makedirs(f'{base_path}/Images/{n_sim}/rgb/')
    # os.makedirs(f'{base_path}/Images/{n_sim}/depth/')
    os.makedirs(f'{base_path}/Images/{n_sim}/mask/')

    w, h = 240, 180

    display = pygame.display.set_mode((w, h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    # display = pygame.display.set_mode((w*2, h), pygame.HWSURFACE | pygame.DOUBLEBUF)    
    clock = pygame.time.Clock()

    data = {'throttle': [], 'brake': [], 'steer': [], 'junction': []}
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = World(client.get_world())

        # world.random_weather()
        world.set_weather(8)

        vehicle = world.player
        blueprint_library = world.world.get_blueprint_library()

        cam_pos = carla.Transform(carla.Location(x=1.6, z=1.7))

        camera_rgb = create_camera(cam_type='rgb',
                                   vehicle=vehicle,
                                   pos=cam_pos,
                                   h=h, w=w,
                                   lib=blueprint_library,
                                   world=world.world)
        actor_list.append(camera_rgb)

        camera_semseg = create_camera(cam_type='semantic_segmentation', vehicle=vehicle, pos=cam_pos, h=h, w=w,
                                      lib=blueprint_library, world=world.world)
        actor_list.append(camera_semseg)

        camera_depth = create_camera(cam_type='depth', vehicle=vehicle, pos=cam_pos, h=h, w=w,
                                     lib=blueprint_library, world=world.world)
        actor_list.append(camera_depth)

        world.player.set_autopilot(True)

        # with CarlaSyncMode(world.world, camera_rgb, fps=20) as sync_mode:
        with CarlaSyncMode(world.world, camera_rgb, camera_semseg, camera_depth, fps=20) as sync_mode:
            frame = 0
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg, image_depth = sync_mode.tick(timeout=2.0)
                # snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # Draw the display.
                rgb_arr = img_to_array(image_rgb)
                # show_window(display, rgb_arr)                
                semseg_arr = img_to_array(image_semseg)[:, :, 2]
                semseg_arr = ((semseg_arr == 12)*255).astype(np.uint8)
                semseg_arr = cv2.merge([semseg_arr, semseg_arr, semseg_arr])
                arr = np.hstack([rgb_arr, semseg_arr])
                show_window(display, arr)
                
                pygame.display.flip()                
                c = vehicle.get_control()
                # vehicle.apply_control(carla.VehicleControl(throttle=min(c.throttle, 0.4), steer=c.steer, brake=c.brake))

                if c.throttle != 0:
                    loc = vehicle.get_location()
                    # print(round(c.throttle, 3), round(c.steer, 3), round(c.brake, 3), loc.x, loc.y)
                    wp = world.map.get_waypoint(loc)

                    # if wp.is_junction:

                    depth_arr = img_to_array(image_depth)
                    mask_arr = img_to_array(image_semseg)
                    # cv2.imwrite(f'{base_path}/{n_sim}/rgb/{frame}_{c.throttle}_{c.brake}_{c.steer}.png', rgb_arr)                                       
                    # cv2.imwrite(f'{base_path}/Images/{n_sim}/depth/{frame}.png', depth_arr)

                    # En uso
                    cv2.imwrite(f'{base_path}/Images/{n_sim}/rgb/{frame}.png', rgb_arr)
                    cv2.imwrite(f'{base_path}/Images/{n_sim}/mask/{frame}.png', mask_arr)

                    data['throttle'].append(min(c.throttle, 0.4))
                    data['brake'].append(c.brake)
                    data['steer'].append(c.steer)
                    data['junction'].append(wp.is_junction)

                    frame += 1

                    if frame == 8000:
                        return
                    if frame % 1000 == 0:
                        print(f'Frame: {frame}')
    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        world.destroy()
        pygame.quit()
        print('saving csv...')
        df = pd.DataFrame.from_dict(data)
        df.to_csv(f'{base_path}/Dfs/{n_sim}.csv', index=False)
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
