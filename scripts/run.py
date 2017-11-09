######################################
# born script
######################################
from persistqueue import Queue
from threading import Thread
import subprocess
import json
import time

queues = list()
to_add = Queue("add_queue")
done = Queue("done_queue")

#
# migration functions
#
def create_image_param_file(image_name):
    # TODO: change to correct parameters
    params = dict()
    params["d1"] = 1
    params["d2"] = 1
    params["d3"] = 1
    params["n1"] = 1
    params["n2"] = 1
    params["n3"] = 1
    params["i1"] = 1
    params["o2"] = 1
    params["o3"] = 1
    params["label1"] = "Undefined"
    params["label2"] = "Undefined"
    params["label3"] = "Undefined"
    params["filename"] = "image.%s.json.dat" % image_name

    # TODO: cp dat
    with open("image." + str(image_name) + ".json", "w") as image_param_file:
        json.dump(params, image_param_file, indent=4)
        image_param_file.write("\n")

def create_migration_file(image_name, worker_id):
    # TODO: change to correct parameters
    params = dict()
    params["data"] = "data.big.%s.json" % image_name
    params["image"] = "image.%s.json" % image_name
    params["wavelet"] = "wavelet.json"
    params["velocity"] = "velocity.json"

    with open("mig." + str(worker_id) + ".P", "w") as migration_file:
        json.dump(params, migration_file, indent=4)
        migration_file.write("\n")

def run_rtm(worker_id):
    gpus = list()
    for i in range(worker_id * 2, (worker_id + 1) * 2):
        gpus.append(str(i))

    cmd = "RTM3D json=mig.%s.P" % str(worker_id)
    cmd = "CUDA_VISIBLE_DEVICES=" + ",".join(gpus) + " " + cmd
    print(cmd)

    # TODO: uncomment the next line
    #return subprocess.call(cmd.split())

#
# add functions
#
def create_add_file(image_name):
    params = dict()
    params["out"] = "final.image.json"
    params["in1"] = "image.%s.json" % image_name
    params["in2"] = "final.image.json"

    with open("add.P", "w") as add_file:
        json.dump(params, add_file, indent=4)
        add_file.write("\n")

def run_add():
    cmd = "Add json=add.P"

    # TODO: uncomment the next line
    #return subprocess.call(cmd.split())

#
# main worker functions
#
def migration_worker(worker_id):
    while True:
        item = queues[worker_id].get()
        print("Migrating " + str(item))
        create_image_param_file(item)
        create_migration_file(item, worker_id)
        run_rtm(worker_id)
        to_add.put(item)
        queues[worker_id].task_done()

def add_worker():
    while True:
        item = to_add.get()
        print("Adding " + str(item))
        create_add_file(item)
        run_add()
        done.put(item)
        to_add.task_done()

if __name__ == "__main__":
    # create migration queues
    num_migration_worker = 4
    for i in range(num_migration_worker):
        queues.append(Queue("migration_queue_worker_" + str(i)))

    # start migration workers
    for i in range(num_migration_worker):
        t = Thread(target=migration_worker, args=(i,))
        t.daemon = True
        t.start()

    # start add worker
    num_add_worker = 1
    for i in range(num_add_worker):
        t = Thread(target=add_worker)
        t.daemon = True
        t.start()

    # wait for them to finish
    for i in range(num_migration_worker):
        queues[i].join()
    to_add.join()
