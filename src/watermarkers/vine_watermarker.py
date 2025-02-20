import numpy as np
import torch
from src import config
from src.types import ImageType
from src.utils import (
    grayscale_image_to_watermark,
    measure_time,
    watermark_to_grayscale_image,
)
from src.watermarker import DecodingResults, EncodingResults, Watermarker
from PIL import Image
from src.watermarkers.vine.src.vine_turbo import VINE_Turbo, VAE_decode, VAE_encode
from torchvision import transforms
from src.watermarkers.vine.src.stega_encoder_decoder import CustomConvNeXt

pretrained_model_name = "Shilin-LU/VINE-R-Enc"
watermark_encoder = VINE_Turbo.from_pretrained(pretrained_model_name)
watermark_encoder.to(config.device)
# parser.add_argument(
#     "--pretrained_model_name",
#     type=str,
#     default="Shilin-LU/VINE-R-Enc",
#     help="pretrained_model_name",
# )


decoder = CustomConvNeXt.from_pretrained(pretrained_model_name)
decoder.to(config.device)


def crop_to_square(image):
    width, height = image.size

    min_side = min(width, height)
    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


class VineWatermarker(Watermarker):
    def __init__(self) -> None:
        # def __init__(self, amount: float, scale: int, watermark_length: int) -> None:
        super().__init__()
        # self.amount = amount
        # self.scale = scale
        # self.watermark_length = watermark_length

    def __encode(self, image: ImageType, watermark: str) -> EncodingResults:
        # watermark_bytes = watermark.encode("utf-8")
        # greyscale_watermark = watermark_to_grayscale_image(
        #     watermark_bytes, image.shape, self.scale
        # )
        # watermarked_image = image.astype(int) + (
        #     greyscale_watermark[:, :, None].astype(int) * self.amount
        # )

        encoded_image_256 = watermark_encoder(image, watermark)

        # watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
        return encoded_image_256

    def encode(self, image: ImageType, watermark: str) -> EncodingResults:
        ### ============= load image =============
        # input_image_pil = Image.open(args.input_path).convert("RGB")  # 512x512
        input_image_pil = Image.fromarray(image).convert("RGB")  # 512x512
        # input_image_pil = image.convert("RGB")  # 512x512
        if input_image_pil.size[0] != input_image_pil.size[1]:
            input_image_pil = crop_to_square(input_image_pil)

        size = input_image_pil.size
        t_val_256 = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )
        t_val_512 = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
            ]
        )
        resized_img = t_val_256(input_image_pil)  # 256x256
        resized_img = 2.0 * resized_img - 1.0
        input_image = (
            transforms.ToTensor()(input_image_pil).unsqueeze(0).to(config.device)
        )  # 512x512
        input_image = 2.0 * input_image - 1.0
        image = resized_img.unsqueeze(0).to(config.device)
        ### ============= load message =============
        # if args.load_text: # text to bits
        if len(watermark) > 12:
            print("Error: Can only encode 100 bits (12 characters)")
            raise SystemExit
        data = bytearray(watermark + " " * (12 - len(watermark)), "utf-8")
        packet_binary = "".join(format(x, "08b") for x in data)
        watermark = [int(x) for x in packet_binary]
        watermark.extend([0, 0, 0, 0])
        watermark = torch.tensor(watermark, dtype=torch.float).unsqueeze(0)
        watermark = watermark.to(config.device)
        # else: # random bits
        #     data = torch.randint(0, 2, (100,))
        #     watermark = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        #     watermark = watermark.to(device)

        encoded_image_256, time_taken = measure_time(self.__encode)(image, watermark)
        print(encoded_image_256.shape, "\n\nencoded shape 0")

        ### ============= resolution scaling to original size =============
        residual_256 = encoded_image_256 - resized_img  # 256x256
        residual_512 = t_val_512(residual_256)  # 512x512 or original size
        encoded_image = residual_512 + input_image  # 512x512 or original size
        encoded_image = encoded_image * 0.5 + 0.5
        encoded_image = torch.clamp(encoded_image, min=0.0, max=1.0)
        print(encoded_image.shape, "\n\nencoded shape 1")
        return encoded_image, time_taken

    def __decode(
        self, image: ImageType, source: ImageType
    ) -> str:  # TODO: implement decode
        # source_avg = source.mean(axis=(0, 1))
        # image_avg = image.mean(axis=(0, 1))
        # avg_diff = source_avg - image_avg
        # image_avg = np.clip(
        #     image_avg.astype(int) + avg_diff.astype(int), 0, 255
        # ).astype(np.uint8)
        # diff = image.astype(int) - source.astype(int)
        # diff_avg = diff.mean(axis=2)
        # diff_avg = diff_avg.astype(float) / self.amount
        # diff_avg = (diff_avg - diff_avg.min()) / (diff_avg.max() - diff_avg.min()) * 255
        # diff_avg = np.clip(diff_avg, 0, 255).astype(np.uint8)
        # watermark_bytes = grayscale_image_to_watermark(
        #     diff_avg, self.scale, self.watermark_length
        # )
        # try:
        #     watermark = watermark_bytes.decode("utf-8")
        # except UnicodeDecodeError:
        #     watermark = None
        ### ============= watermark decoding & detection =============
        pred_watermark = decoder(image)

        pred_watermark = np.array(pred_watermark[0].cpu().detach())
        pred_watermark = np.round(pred_watermark)
        decoded = pred_watermark.astype(int)

        return decoded

    def decode(
        self, image: ImageType, *, source: ImageType | None = None
    ) -> DecodingResults:
        # if len(args.groundtruth_message) > 12:
        # print('Error: Can only encode 100 bits (12 characters)')
        # raise SystemExit
        # args.groundtruth_message == text # TODO
        # data = bytearray(args.groundtruth_message + ' ' * (12 - len(args.groundtruth_message)), 'utf-8')
        # packet_binary = ''.join(format(x, '08b') for x in data)
        # watermark = [int(x) for x in packet_binary]
        # watermark.extend([0, 0, 0, 0])
        # watermark = torch.tensor(watermark, dtype=torch.float).unsqueeze(0)
        # watermark = watermark.to(device)

        ### ============= load image =============
        t_val_256 = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )
        # image = Image.open(args.input_path).convert("RGB")
        image = image.convert("RGB")
        image = t_val_256(image).unsqueeze(0).to(config.device)

        decoded, time_taken = measure_time(self.__decode)(image)

        # pred_watermark_list = pred_watermark.tolist()
        # groundtruth_watermark_list = watermark[0].cpu().detach().numpy().astype(int).tolist()

        # same_elements_count = sum(x == y for x, y in zip(groundtruth_watermark_list, pred_watermark_list))
        # acc = same_elements_count / 100
        # print('Decoding Bit Accuracy:', acc)
        # decoded, time_taken = measure_time(self.DECODERRRRR)(image, source)
        print(decoded.shape, "\n\decoded shape")

        return decoded, time_taken

    def get_name(self) -> str:
        return f"{super().get_name()}"
