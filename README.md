### step1:開始
```sh
mkdir Github

cd Github

git clone https://github.com/bruce601080102/yolov7_stream_rtc.git

cd yolov7_stream_rtc/docker

./build.sh
```

### step2:進入容器內
```sh
docker exec -it mydeepstream:v2 bash

cd /opt/nvidia/deepstream/deepstream-6.1/commonFolder/Github/yolov7_stream_rtc

pip3 install -r requirement.txt

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
### step2-1:內置zsh
```sh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/bruce601080102/yolov7_stream_rtc/main/docker/build_zsh.sh)" 
```


### step3:pt下載
```sh
cd /opt/nvidia/deepstream/deepstream-6.1/commonFolder/Github/yolov7_stream_rtc

mkdir stream/yolo/weight/default

cd stream/yolo/weight/default

wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```


#### 串流執行
```sh
python3 -m streamlit run navigation.py --server.port 9998 --server.enableCORS=false
```
本地網址:http://localhost:9998


#### ssl憑證測試
- 由於攝像頭啟動基於安全需要憑證或是localhost域名才能順利執行,因此可以使用proxyssl來測試是否能正常使用
```sh
cd ssl
./ssl-proxy-linux-amd64 -from 192.168.10.112:9998 -to 192.168.10.112:8502
```

#### ssh綁網域設定測試
```sh
export STREAMLIT_RUN_FILE_OR_URL=navigation.py
```

## step4:Tensor-RT生成
- 執行以下命令集會生成出engine文件,一個環境只能生出對應的engine,因此更換環境需要重新生成
- `zsh環境下無法使用,需切回bash`
```sh

cd stream/output_rt

chmod u+x *.sh

mkdir engine

./build.sh
```

### step5:supervisorctl啟動
```sh
supervisorctl start all
```

## DEMO


<table border="3" align="center">
    <tr>
        <th>Default Demo</th>
        <th>PlateLicense Demo</th>
    </tr>
    <tr>
        <td><img src="./images/animation.gif" alt="" height=400 width=200 /></td>
        <td><img src="./images/PlateLicense.gif" alt="" height=400 width=200 /></td>
    </tr>
</table>