import random
from sensapex import UMP
from sensapex import SensapexDevice, UMP
ump = UMP.get_ump()
dev_ids = ump.list_devices()


devs = {i: SensapexDevice(i) for i in dev_ids}

# print("SDK version:", ump .sdk_version())
# print("Found device IDs:", dev_ids)

# stage = ump.get_device(1)
# stage.calibrate_zero_position()

# def print_pos(timeout=None):
#     line = ""
#     for i in dev_ids:
#         dev = devs[i]
#         try:
#             pos = str(dev.get_pos(timeout=timeout))
#         except Exception as err:
#             pos = str(err.args[0])
#         pos = pos + " " * (30 - len(pos))
#         line += f"{i:d}:  {pos}"
#     print(line)

# print_pos()
dev = devs[1]
a = 9999
#print(dev.get_pos())
#[0,19999]
#dev.goto_pos([0, 0, 0 ,9999], 5000)
dev.goto_pos([9999, 9999, 9999, 9999], 5000)
#dev.goto_pos([6000, 14000, 15000, 14000], 5000) #point 图像右
#dev.goto_pos([7000, 17000, 15000, 14000], 5000) #point 图像左
#dev.goto_pos([9999 - addy, 9999 - addx, 9999 - addz, 9999+position], 10000)
#dev.goto_pos([21552.300785898264,7716.887111118403,-11219.106442604763,19999],5000)
#dev.goto_pos([14000, 17000, 15300, 19999], 5000)
a = a+1
#dev.goto_pos([000.837890625, 900.900390625, 900.837890625, 900.7841796875], 5000)

#dev.goto_pos([19999.837890625, 9999.900390625, 9999.837890625, 9999.7841796875], 3000)
































