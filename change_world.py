import sys
import os
import glob

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

import carla

client = carla.Client('localhost', 2000)
client.load_world('Town04')
