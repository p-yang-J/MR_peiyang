# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:30:01 2022

@author: hp


stage = ump.get_device(1)
stage.calibrate_zero_position()


manipulator = ump.get_device(1)
manipulator.goto_pos((0.0, 0.0, 0.0, 0.0), speed=20)



manipulator = ump.get_device(1)
manipulator.goto_pos((2500.0412, 6810.0003, 15830.1419), speed=200)

pressure = ump.get_device(30)
pressure.set_pressure(1, 2.3e-4)

"""

from sensapex import UMP
from sensapex import SensapexDevice, UMP
ump = UMP.get_ump()
dev_ids = ump.list_devices()


devs = {i: SensapexDevice(i) for i in dev_ids}

print("SDK version:", ump .sdk_version())
print("Found device IDs:", dev_ids)

stage = ump.get_device(1)
stage.calibrate_zero_position()

def print_pos(timeout=None):
    line = ""
    for i in dev_ids:
        dev = devs[i]
        try:
            pos = str(dev.get_pos(timeout=timeout))
        except Exception as err:
            pos = str(err.args[0])
        pos = pos + " " * (30 - len(pos))
        line += f"{i:d}:  {pos}"
    print(line)

print_pos()
dev = devs[1]
print(dev.get_pos())


#dev.goto_pos([9999.837890625, 9999.900390625, 9999.837890625, 9999.7841796875], 2000)
dev.goto_pos([0, 900.900390625, 900.837890625, 900.7841796875], 5000)
'''
pos1 = [[], [], [],[]]
start_pos = [2999.837890625, 6999.900390625, 10999.837890625, 9999.7841796875]
#(2500.0412, 6810.0003, 15830.1419)
import time
for i in range(20):
    start_pos = [19999.837890625 - i*1000, 16999.900390625- i*100, 50999.837890625- i*100, 9999.7841796875]
    print(start_pos)
    time.sleep(0.1)
    dev.goto_pos(start_pos, 200)
'''

#tool D [ 9999 middle.... 12999....0.2]
#X [ 9999 middle.... 15999....3999 [ reasonable 5000]]
#Y [ 9999 middle.... 15999....3999]
#Z [ 9999 middle.... 20075....3999]