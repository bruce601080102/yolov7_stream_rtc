# !/bin/expect

output_rt_path=$(pwd)
echo $output_rt_path
string=$output_rt_path
array=(${string///yolov7_stream_rtc/ })
parent_path=${array[0]}

cd ${parent_path}

step1_is_download_yolv7(){
    is_yolov7_folder="yes"
    check_folder=$(ls)
    array1=(${check_folder// / })

    for val in "${array1[@]}";
    do
        if [[ "$val" == "yolov7" ]]
        then
            is_yolov7_folder="yes"
        fi
    done

    if [[ "$is_yolov7_folder" == "no" ]]
        then
            git clone https://github.com/WongKinYiu/yolov7
            cd ${output_rt_path}
            cp EfficientNMS.py export_onnx.py ${parent_path}/yolov7
            cd ${parent_path}/yolov7
            echo $(pwd)
    else
            cd ${output_rt_path}
            cd ${parent_path}/yolov7
            echo $(pwd)
            echo "yolov7 folder already exists in this directory"
    fi
}

step2_create_crop_onnx(){
    # create onnx
    project_path=${parent_path}"/yolov7_stream_rtc/stream"
    pt_path=${project_path}"/yolo/weight"
    # ================================crop=============================
    echo "=====crop onnx start====="

    str_crop=$(python3 export_onnx.py --weights $pt_path/crop/best.pt --img-size 416 416)
    pass_string="Export complete"
    result=$(echo $str_crop | grep "${pass_string}")
    if [[ "$result" != "" ]]
    then
        return 1
    else
        return 0
    fi
}

step3_create_seg_onnx(){
    # ================================seg=============================
    echo "=====seg onnx start====="
    str_seg=$(python3 export_onnx.py --weights $pt_path/seg/best.pt --img-size 416 416)
    pass_string="Export complete"
    result=$(echo $str_seg | grep "${pass_string}")
    if [[ "$result" != "" ]]
    then
        return 1
    else
        return 0
    fi
}

step4_onnx_build_status(){

    if [ "$onnx_crop_bool" = "1" ] ; 
    then
        bool=1
        echo '[=====build crop onnx success=====]'
    else
        bool=0
        echo "[=====build crop onnx failed=====]"
    fi

    if [ "$onnx_seg_bool" = "1" ] ; 
    then
        bool=1
        echo "[=====build seg onnx success=====]"
    else
        bool=0
        echo "[=====build seg onnx failed=====]"
    fi
    return $bool
}

step5_create_crop_rt(){
    echo "=====crop tensort start====="
    
    tr_crop=$($rt_output_path/trtexec --onnx=$pt_path/crop/best.onnx --saveEngine=$rt_output_path/best1.engine --fp16 --workspace=200)
    pass_string="&&&& PASSED TensorRT.trtexec"
    result=$(echo $tr_crop | grep "${pass_string}")
    if [[ "$result" != "" ]];
    then
        return 1
    else
        return 0
    fi
}

step6_create_seg_rt(){
    echo "=====seg tensort start====="
    
    tr_seg=$($rt_output_path/trtexec --onnx=$pt_path/seg/best.onnx --saveEngine=$rt_output_path/best2.engine --fp16 --workspace=200)
    pass_string="&&&& PASSED TensorRT.trtexec"
    result=$(echo $tr_seg | grep "${pass_string}")
    if [[ "$result" != "" ]];
    then
        return 1
    else
        return 0
    fi
}

step1_is_download_yolv7

step2_create_crop_onnx
onnx_crop_bool=$?

step3_create_seg_onnx
onnx_seg_bool=$?

step4_onnx_build_status $onnx_crop_bool $onnx_seg_bool
is_start_step5=$?

rt_output_path=$parent_path/yolov7_stream_rtc/stream/output_rt
pt_path=$parent_path/yolov7_stream_rtc/stream/yolo/weight

if [ "$is_start_step5" = "1" ] ; then
    step5_create_crop_rt
    tr_crop_bool=$?

    step6_create_seg_rt
    tr_seg_bool=$?
    echo "tr_seg_bool:"$tr_seg_bool
else
    echo "create failed"
    tr_crop_bool="0"
    tr_seg_bool="0"
fi

if [[ "$tr_crop_bool" != "1" ]] ; then
    echo "[crop tensorrt build failed]"
else
    echo "[crop tensorrt build success]"
fi

if [[ "$tr_seg_bool" != "1" ]] ; then
    echo "[seg tensorrt build failed]"
else
    echo "[seg tensorrt build success]"
fi
