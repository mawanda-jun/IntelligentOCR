import glob
import os
from io import StringIO
from threading import Thread
import logging
from logger import TimeHandler
from costants import THREADS, INFERENCE_GRAPH
from pipeline import pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.addHandler(TimeHandler().handler)


class MyThread(Thread):
    def __init__(self, name, file_path):
        Thread.__init__(self)
        self.name = name
        self.path = file_path

    def run(self):
        for file_path in self.path:
            file_path = os.path.join(file_path)
            fp = StringIO()
            pipeline(
                pdf_path=file_path,
                inference_graph_path=INFERENCE_GRAPH,
                thread_name=self.name
            )
            logger.info(fp.getvalue())
            fp.close()


if __name__ == '__main__':
    path_list = []
    for path in glob.iglob("..\\Polizze\\" + '/**/*.pdf', recursive=True):
        path_list.append(path)

    el_per_list = int(len(path_list) / THREADS)
    thread_list = []

    i = 0
    path_list_per_thread = []
    if len(path_list) == 1:
        new_thread = MyThread('Thread_{}'.format(0), path_list)
        new_thread.start()
        new_thread.join()
    else:
        for i in range(0, THREADS):
            if i < THREADS - 2:
                path_list_per_thread = path_list[el_per_list * i:el_per_list * (i + 1) - 1]
            else:
                path_list_per_thread = path_list[
                                       el_per_list * i:len(path_list) - 1]  # lista vuota se c'e' un solo elemento

            new_thread = MyThread('Thread_{}'.format(i), path_list_per_thread)
            new_thread.start()
            thread_list.append(new_thread)

        for new_thread in thread_list:
            new_thread.join()
