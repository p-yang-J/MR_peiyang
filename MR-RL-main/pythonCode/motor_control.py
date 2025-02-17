import serial
import time

ser = serial.Serial('COM4', 9600, timeout=1)
print("Connection established")

currentRead = 1
cmd = ''
currentThreshold = 250
motorstep = 115000
cycle = 0
data = [0] * 10000
velocityMax = 2000

ser.write(b'ANSW0\n')
time.sleep(0.05)

cmd = 'SP' + str(velocityMax) + '\n'
ser.write(cmd.encode())
time.sleep(0.05)

while True:
    # send command, bend the joint
    motorstep = motorstep * -1
    cmd = 'LA' + str(motorstep) + '\n'
    ser.write(cmd.encode())
    cmd = "M\n"
    ser.write(cmd.encode())
    time.sleep(2.6)

    # read the current values
    ser.flushInput()
    time.sleep(0.05)
    cmd = 'GRC\n'
    ser.write(cmd.encode())

    mesg = ser.readline().decode()
    currentRead = float(mesg)
    data[cycle] = currentRead
    if currentRead < currentThreshold:
        break

    cycle += 1
    if cycle > 10000:
        break

print("The final cycle is ", cycle)
ser.close()
print("Port disconnected")

