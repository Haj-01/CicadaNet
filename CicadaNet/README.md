

## ToDo
- [x] Real-time version
- [x] Update trainer
- [x] Visualization of the spectrogram and the metrics (PESQ, STOI, SI-SDR) in the training
- [ ] More docs

## Usage


‪D:\vs2019\VisualStudioSetup.exe --noweb --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended‪D:\vscommunity2019\vs_setup.exe --noweb --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended

Training:
```
python train.py -C config/train/baseline_model.json5
```
CUDA_VISIBLE_DEVICES=1 python train.py -C config/train/baseline_model.json5
CUDA_VISIBLE_DEVICES=1 python train.py -C config/train/crn_b.json5
# The configuration file used to train the model is "config/train/train.json"
# Use GPU No.1 and 2 for training

tensorboard --logdir=E:\JNT\CRN_Net\A-CRN-Network\logs-lr=0.0004\baseline_model\logs
tensorboard --logdir=E:\JNT\CRN_Net\logs\GroupRelPos_Conformer_gcn32_dim=256\baseline_model\logs


Inference:

```
python inference.py \
    -C config/inference/basic.json5 \
    -cp ~/Experiments/CRN/baseline_model/checkpoints/latest_model.tar \
    -dist ./enhanced
```
python inference.py  -C config/inference/basic.json5 -cp G:\JNT\code\crnn\job\cicadanet_mag_map_magloss_11tscb\baseline_model\checkpoints\best_model.tar -dist ./enhanced









