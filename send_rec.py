import numpy as np
import serial
import threading
import time
port_sender= "COM32"
port_receiver="COM33"

BYTE_RESET_PROBABLITY=0.005



def send_data(ser, data):

    START = 0xAA
    END = 0x55

    payload = data.astype(np.uint8).tolist()
    length = len(payload)
    checksum = sum(payload) % 256

    data_to_send = []
    data_to_send.append(START)
    data_to_send.append(length)
    data_to_send.extend(payload)
    data_to_send.append(checksum)
    data_to_send.append(END)
    
    for b in data_to_send:
        ser.write(bytes([b]))

def receive_data(ser):

    START = 0xAA
    END = 0x55

    while True:
      b = ser.read(1)
      if not b:
        return [], False
      if b[0] == START:
        break

    length_b = ser.read(1)
    if not length_b:
        return [], False

    length = length_b[0]


    payload = []
    for _ in range(length):
        d = ser.read(1)
        if not d:
            return [], False
        payload.append(d[0])

    checksum_b = ser.read(1)
    end_b = ser.read(1)

    if not checksum_b or not end_b:
        return payload, False

    if end_b[0] != END:
        return payload, False

    calc_checksum = sum(payload) % 256

    if calc_checksum == checksum_b[0]:
        return np.array(payload), True
    else:
        return np.array(payload), False
    
def generate_pwm():
    return np.random.randint(0, 255, size=(100,))


def receive_thread_task(received_data, no_of_success):
    no_of_tries = 0
    try:
        with serial.Serial(port_receiver, 9600, timeout=0.2) as ser:
            while len(received_data) < 100 and no_of_tries < 150:
                received_arr, acknowledgement = receive_data(ser)

                if np.any(received_arr) or acknowledgement:
                    received_data.append(received_arr)
                    if acknowledgement:
                        no_of_success[0] += 1
                        print(f"[RECEIVER] PASSED")
                    else:
                        print(f"[RECEIVER] FAILED")

                no_of_tries += 1
                time.sleep(0.01)
    except Exception as e:
        print("Receiver error:", e)



def send_thread_task(all_data):
    try:
        with serial.Serial(port_sender, 9600) as ser:
            for i, data in enumerate(all_data):
                send_data(ser, data)
                print(f"[SENDER] [{i+1}] Packet Sent")
                time.sleep(0.15)
    except Exception as e:
        print("Sender error:", e)


def main():
    pwm_data = [generate_pwm() for i in range(100)]
    received_data = []
    no_of_success = [0]

    receiver_thread = threading.Thread(target=receive_thread_task, args=(received_data, no_of_success))
    receiver_thread.daemon = True
    receiver_thread.start()

    time.sleep(1)

    sender_thread = threading.Thread(target=send_thread_task, args=(pwm_data,))
    sender_thread.daemon = True
    sender_thread.start()

    sender_thread.join(timeout=30)
    receiver_thread.join(timeout=30)

    print(f"Total Successful: {no_of_success[0]}/100")


if __name__ == "__main__":
    main()