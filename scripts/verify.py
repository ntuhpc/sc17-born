from persistqueue import Queue

if __name__ == "__main__":
    # load all queues
    #to_add = Queue("add_queue")
    done = Queue("done_queue")
    #num_migration_worker = 4
    #queues = list()
    #for i in range(num_migration_worker):
    #    queues.append(Queue("migration_queue_worker_" + str(i)))

    # check content in queue
    print("Done queue contains " + str(done.qsize()) + " items")
