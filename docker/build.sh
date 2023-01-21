docker build -t mydeepstream:v2 . --no-cache

docker run --gpus all -it -p 30:22 -p 8890:8888  -p 9997:9997  -p 9998:9998  -v /home/bruce/桌面/jupyter/deepstream/:/opt/nvidia/deepstream/deepstream-6.1/commonFolder -e DISPLAY=:0 mydeepstream:v2

