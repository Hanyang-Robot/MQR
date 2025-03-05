import binascii
import time

import numpy as np
import serial
import serial.tools
import serial.tools.list_ports


class Robotiq2F85():
    CLOSE_LENGTH = 0.163
    FINGER_WIDTH = 0.0272

    def __init__(self, config):
        self.ser = serial.Serial(
            port=config['gripper']['port'],
            baudrate=config['gripper']['baudrate'],
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0,
        )

        self.clear_rAct_request = b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30'
        self.set_rAct_request = b'\x09\x10\x03\xE8\x00\x03\x06\x01\x00\x00\x00\x00\x00\x72\xE1'
        self.close_request = b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29'
        self.open_request = b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19'
        self.read_status_request = b'\x09\x04\x07\xD0\x00\x03\xB1\xCE'
        
    def activate(self):
        self.ser.write(self.clear_rAct_request)
        time.sleep(0.1)
        self.ser.write(self.set_rAct_request)
        time.sleep(4)

    def close(self):
        while True:
            self.ser.write(self.close_request)
            raw_response = self.ser.readline()
            response = binascii.hexlify(raw_response).decode('utf-8')
            if response == '091003e800030130':
                break
            else:
                time.sleep(2)
            

    def open(self):
        while True:
            self.ser.write(self.open_request)
            raw_response = self.ser.readline()
            response = binascii.hexlify(raw_response).decode('utf-8')
            if response == '091003e800030130':
                break
            else:
                time.sleep(2)
    
    def check_grasp(self):
        grasp_response = []
        while len(grasp_response) == 0:
            self.ser.write(self.read_status_request)
            time.sleep(0.1)

            # Get response
            raw_response = self.ser.readline()
            raw_response = raw_response.split(b"\t")

            # Decoding response
            response = []
            for i in range(len(raw_response)):
                decoded_response = binascii.hexlify(raw_response[i]).decode('utf-8')
                response = np.append(response, decoded_response)

            # Remove duplicate values
            for data in response:
                if data not in grasp_response:
                    grasp_response.append(data)

        # Check grasp
        if '0406b900' in grasp_response:
            return True
        else:
            return False

    def disconnect(self):
        self.ser.close()
        
if __name__ == "__main__":
    config = {
        'gripper': {
            'port': "/dev/ttyUSB0",
            'baudrate': 115200
        }
    }
    
    gripper = Robotiq2F85(config)
    gripper.activate()

    gripper.close()
    gripper.open()        
    gripper.disconnect()