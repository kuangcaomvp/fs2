该工程用于将florence2转onnx tensorrt用于模型推理

1 

   https://huggingface.co/microsoft/Florence-2-base 下载florence2模型 放在根目录下



     --fs2
     
         --Florence-2-base
         
           -- **.json
           
           -- **.py
           
           -- **.bin
     
2 

      conda 创建python8 或者 10的环境 安装pytorch transformers pyyaml onnx onnxruntime-gpu  onnxsim等依赖库 具体安装方法略

3 

      根目录下创建config文件夹，运行 python yaid_lm_config.py


4    

     python florence2onnx_enc.py 
     
     python florence2onnx_dec.py
   
     python inference_onnx.py 测试  注意修改代码内的图片为自己的测试图片路径


5    

     python export_enc_trt.py
   
     python export_dec_trt.py
   
     python inference_cuda.py   测试  注意修改代码内的图片为自己的测试图片路径

  
  
