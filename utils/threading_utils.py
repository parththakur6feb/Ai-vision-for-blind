from queue import Queue
import threading

commands = Queue()

def listen_thread(mic):
    while True:
        cmd = mic.listen()
        if cmd:
            commands.put(cmd)

# Start thread
threading.Thread(target=listen_thread, args=(mic,), daemon=True).start()

while True:
    frame = cam.get_frame()
    if not commands.empty():
        command = commands.get()
        if "object" in command:
            objects = detector.detect(frame)
            # speak..

