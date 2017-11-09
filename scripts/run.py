######################################
# born script
######################################
from persistqueue import Queue
from threading import Thread
import time

queues = list()
to_add = Queue("add_queue")
done = Queue("done_queue")

def migration_worker(worker_id):
    while True:
        item = queues[worker_id].get()
        print("Migrating " + str(item))
        time.sleep(5)
        to_add.put(item)
        queues[worker_id].task_done()

def add_worker():
    while True:
        item = to_add.get()
        print("Adding " + str(item))
        time.sleep(1)
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

#
# functions
#
# - load_to_migrate_queue()
# - load_migrating_queue()
# - load_to_add_queue()
# - load_done_queue()
#
# - migrate()
#   - create_image_param_file()
#   - create_migration_file()
#   - run_rtm()
#
# - add()
#   - create_add_file()
#   - run_add()
