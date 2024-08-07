log文件中保存的best_model为训练后的模型
enhance中保存的是inference后的降噪音频
wind_noise分为训练验证测试集分别与干净鸟叫声混合


command：
训练： python train.py -C config/train/cicadanet_train.json5    
推理： python inference.py -C config/inference/cicadanet_inference.json5 -cp E:\HAN\CRNnet\database\log\cicadanet_train\checkpoints\best_model.tar -dist E:\HAN\CRNnet\database\enhance