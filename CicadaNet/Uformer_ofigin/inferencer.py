import librosa
import torch
from tqdm import tqdm
import soundfile
from Uformer_ofigin.base_inferencer import BaseInferencer
from Uformer_ofigin.interencer_full_band1 import full_band_no_truncation
import time
import xlwt

@torch.no_grad()
def inference_wrapper(
        dataloader,
        model,
        device,
        inference_args,
        enhanced_dir
):
    excelName = r"E:\JNT\Uformer\bird_testdata\uformer_cpu_runtime.xls"

    f = xlwt.Workbook(encoding='utf-8', style_compression=0)  # 新建一个excel
    sheet = f.add_sheet('sheet1')  # 新建一个sheet
    i = 0
    for noisy_ds, name in tqdm(dataloader, desc="Inference"):
        assert len(name) == 1, "The batch size of inference stage must 1."
        name = name[0]
        import time
        start = time.time()
        # noisy = noisy.numpy()
        # clean = clean.numpy().reshape(-1)
        # print(f"noisy:{noisy_ds.shape}")

        if inference_args["inference_type"] == "full_band_no_truncation":
            output = full_band_no_truncation(model, device, inference_args, noisy_ds)
        else:
            raise NotImplementedError(f"Not implemented Inferencer type: {inference_args['inference_type']}")

        # librosa.output.write_wav(enhanced_dir / f"{name}.wav", enhanced, sr=16000)

        sr = 32000
        soundfile.write(enhanced_dir / f"{name}.wav", output, sr)
        # end = time.time()
        # time = end - start
        # sheet.write(i, 1, time)  # 参数i,0,s分别代表行，列，写入值
        # i = i + 1
        # f.save(excelName)
        # soundfile.write(enhanced_dir / f"{name}.wav", output, sr)
        # librosa.output.write_wav(enhanced_dir / f"{name}.wav", output, sr=16000)
        # end = time.time()
        # print(f" time: {end - start} ")


class Inferencer(BaseInferencer):
    def __init__(self, config, checkpoint_path, output_dir):
        super(Inferencer, self).__init__(config, checkpoint_path, output_dir)

    @torch.no_grad()
    def inference(self):
        inference_wrapper(
            dataloader=self.dataloader,
            model=self.model,
            device=self.device,
            inference_args=self.inference_config,
            enhanced_dir=self.enhanced_dir
        )
