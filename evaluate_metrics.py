from metrics import create_metric, MetricType


def load_metrics_models():
    print("Loading metric models...")
    metrics_models = {
        "psnr": create_metric(MetricType.PSNR),
        "ssim": create_metric(MetricType.SSIM),
        "fsim": create_metric(MetricType.FSIM),
        "masked": create_metric(MetricType.MASKED, lpips_net="alex"),
        "editing_score": create_metric(MetricType.QWEN),
    }
    print("Metric models loaded.")
    return metrics_models


if __name__ == "__main__":

    metrics_models = load_metrics_models()

    roots = [
        "./output/SD_Inpainting/full_dataset/VAE_MSE_FT_2_STAGE",
        "./output/SD_Inpainting/full_dataset/DiffVax",
        "./output/SD_Img2Img/full_dataset/VAE_MSE_2_STAGE_16",
        "./output/SD_Img2Img/full_dataset/VAE_MSE",
        "./output/SD_Img2Img/full_dataset/VAE_MSE_FT",
        "./output/SD_Img2Img/full_dataset/VAE_MSE_FT_2_STAGE",
        "./output/SD_Img2Img/full_dataset/DiffVax",
        "./output/InstructionPix2Pix/full_dataset/VAE_MSE_2_STAGE_16",
        "./output/InstructionPix2Pix/full_dataset/VAE_MSE_FT_2_STAGE",
        "./output/InstructionPix2Pix/full_dataset/VAE_MSE_FT",
        "./output/InstructionPix2Pix/full_dataset/VAE_MSE",
        "./output/InstructionPix2Pix/full_dataset/DiffVax",
    ]

    for root in roots:
        #metrics_models["psnr"].evaluate_folder(root)
        #metrics_models["ssim"].evaluate_folder(root)
        #metrics_models["fsim"].evaluate_folder(root)
        #metrics_models["masked"].evaluate_folder(root)
        metrics_models["editing_score"].evaluate_folder(root)
        print(f"Done {root}")

    print("Done all evaluations.")