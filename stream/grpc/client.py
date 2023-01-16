
# coding=utf8
import grpc
import ndarray_pb2_grpc, ndarray_pb2
import numpy as np
import cv2


def run():
    conn = grpc.insecure_channel('localhost:50052')
    client = ndarray_pb2_grpc.GrpcServiceStub(channel=conn)
    nda = np.arange(10).astype(np.uint8)
    nda = np.random.randint(-10,10,size=(3,3)).astype(np.uint8)
    wh = nda.shape
    nda = cv2.imread("./00011.jpg")
    # print(nda,wh)

    arg1 = nda.tobytes()
    arg2 = wh[0]
    arg3 = wh[1]
    print(arg3)

    request = ndarray_pb2.Input(ndarray=arg1, width=arg2, height=arg3)
    # request = ndarray_pb2.NDArray(ndarray=arg1)

    response = client.ndarray(request)
    result1 = np.array(np.frombuffer(response.ndarray, dtype=np.uint8)).astype(np.uint8)
    result1 = np.reshape(result1, (wh[0], wh[1]))
    print("received:", result1)

if __name__ == '__main__':
    run()