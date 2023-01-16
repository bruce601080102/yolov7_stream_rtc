#! /usr/bin/env python
# coding=utf8
import time
from concurrent import futures
import grpc
import ndarray_pb2_grpc, ndarray_pb2
import numpy as np
import cv2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class TestService(ndarray_pb2_grpc.GrpcServiceServicer):
    '''
    繼承GrpcServiceServicer,實現hello方法
    '''
    def __init__(self):
        pass

    def ndarray(self, request, context):
        '''
        具體實現hello的方法，並按照pb的返回物件構造HelloResponse返回
        :param request:
        :param context:
        :return:
        '''
        print("----")
        img = request.ndarray
        w = request.width
        h = request.height
        img = np.array(np.frombuffer(img, dtype=np.uint8)).astype(np.uint8)
        img = np.reshape(img, (w, h, 3))
        
        cv2.putText(img, "text", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        
        # return ndarray_pb2.NDArray(ndarray=img.tobytes())
        return ndarray_pb2.Coordinate(coordinate=1)

def run():
    '''
    模擬服務啟動
    :return:
    '''
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    ndarray_pb2_grpc.add_GrpcServiceServicer_to_server(TestService(),server)
    server.add_insecure_port('[::]:50052')
    server.start()
    print("start service...")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    run()