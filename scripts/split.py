#
# split work to multiple queues
#
from persistqueue import Queue

def load_all_data():
    shots = [x for x in range(100)]
    return shots

def split_data(num_workers, data):
    # create queues
    queues = list()
    for i in range(num_workers):
        queues.append(Queue("migration_queue_worker_" + str(i)))

    # split data into queues
    cur = 0
    for i in data:
        queues[cur].put(i)
        cur = (cur + 1) % num_workers

    print("Split " + str(len(data)) + " data into " + str(num_workers) + " workers")

if __name__ == "__main__":
    num_workers = 4
    data = load_all_data()
    split_data(num_workers, data)
