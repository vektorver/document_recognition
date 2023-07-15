import torchvision
import torch
import logging
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import torch.nn as nn
import cv2
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import warnings
from pycocotools.coco import COCO
from dotenv import dotenv_values
import easyocr
import detectron2.data.transforms as T
from scipy.spatial import ConvexHull
import math
from shapely.geometry import Polygon

dotenv_config = dotenv_values(".env")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
logger = logging.getLogger('detectron2')
logger.setLevel(logging.CRITICAL)

print('GPU: ' + str(torch.cuda.is_available()))


class CustomPredictor(DefaultPredictor):

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            # image = T.ResizeTransform(image.shape[0], image.shape[1], 2339, 1654).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            # print(image.shape, inputs["height"], inputs["width"])
            predictions = self.model([inputs])[0]
            return predictions



class PageByPageSEGMpredictor:
    def __init__(self, model_path):

        cfg = get_cfg()
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) 
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 75
        cfg.INPUT.MIN_SIZE_TEST= 800
        cfg.MODEL.DEVICE = "cpu"
        cfg.INPUT.MAX_SIZE_TEST = 800
        cfg.INPUT.FORMAT = 'BGR'
        cfg.TEST.DETECTIONS_PER_IMAGE = 70
        self.predictor = CustomPredictor(cfg)


    
    def __call__(self, img):
        outputs = self.predictor(img)
        return outputs


class Tokenizer:

    def __init__(self, alphabet):
        self.OOV_TOKEN = '<OOV>'
        self.CTC_BLANK = '<BLANK>'
        self.char_map = self.get_char_map(alphabet)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    def get_char_map(self, alphabet):
        """Make from string alphabet character2int dict.
        Add BLANK char fro CTC loss and OOV char for out of vocabulary symbols."""
        char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
        char_map[self.CTC_BLANK] = 0
        char_map[self.OOV_TOKEN] = 1
        return char_map

    """Class for encoding and decoding string word to sequence of int
    (and vice versa) using alphabet."""

    def encode(self, word_list):
        """Returns a list of encoded words (int)."""
        enc_words = []
        for word in word_list:
            enc_words.append(
                [self.char_map[char] if char in self.char_map
                 else self.char_map[self.OOV_TOKEN]
                 for char in word]
            )
        return enc_words
    
    def get_num_chars(self):
        return len(self.char_map)

    def decode(self, enc_word_list):
        """Returns a list of words (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary token."""
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                # skip if blank symbol, oov token or repeated characters
                if (
                    char_enc != self.char_map[self.OOV_TOKEN]
                    and char_enc != self.char_map[self.CTC_BLANK]
                    # idx > 0 to avoid selecting [-1] item
                    and not (idx > 0 and char_enc == word[idx - 1])
                ):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class MoveChannels:
    """Move the channel axis to the zero position as required in pytorch."""

    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        else:
            return np.moveaxis(image, 0, -1)


class ImageResize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        image = cv2.resize(image, (self.width, self.height),
                           interpolation=cv2.INTER_LINEAR)
        return image


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class CRNN(nn.Module):

    def get_resnet34_backbone(self, pretrained=True):
        m = torchvision.models.resnet34(pretrained=True)
        input_conv = nn.Conv2d(3, 64, 7, 1, 3)
        blocks = [input_conv, m.bn1, m.relu,
                  m.maxpool, m.layer1, m.layer2, m.layer3]
        return nn.Sequential(*blocks)

    def __init__(
        self, number_class_symbols, time_feature_count=256, lstm_hidden=256,
        lstm_len=2,
    ):
        super().__init__()
        self.feature_extractor = self.get_resnet34_backbone(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x


class InferenceTransform:

    def get_val_transforms(self, height, width):
        transforms = torchvision.transforms.Compose([
            ImageResize(height, width),
            MoveChannels(to_channels_first=True),
            Normalize(),
            ToTensor()
        ])
        return transforms

    def __init__(self, height, width):
        self.transforms = self.get_val_transforms(height, width)

    def __call__(self, images):
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        transformed_tensor = torch.stack(transformed_images, 0)
        return transformed_tensor

# 'cuda'


class OcrPredictor:

    def predict(self, images, model, tokenizer, device):
        model.eval()
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
        pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
        text_preds = tokenizer.decode(pred)
        return text_preds

    def __init__(self, model_path, config, device='cpu'):
        self.tokenizer = Tokenizer(config['alphabet'])
        self.device = torch.device(device)
        # load model
        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=config['image']['height'],
            width=config['image']['width'],
        )

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(f"Input must contain np.ndarray, "
                            f"tuple or list, found {type(images)}.")

        images = self.transforms(images)
        pred = self.predict(images, self.model, self.tokenizer, self.device)

        if one_image:
            return pred[0]
        else:
            return pred


class PiepleinePredictor:

    def __init__(self, predictor, segm_model_path, ocr_model_path, ocr_config):
        torch.cuda.empty_cache()
        self.segm_predictor = predictor(model_path=segm_model_path)
        # self.ocr_predictor = OcrPredictor(
        #     model_path=ocr_model_path,
        #     config=config_json
        # )
        print('Model created')
        self.ocr_predictor = easyocr.Reader(['ru'])
        # self.ocr_predictor = pytesseract.image_to_string
        
        
    def getClasses(self):
        
        classes = dotenv_config['CLASSES_WITH_ORDER'].split(',')
        return {i+1: classes[i] for i in range(len(classes))}


    def crop_img_by_polygon(self, img, polygon):
        # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
        pts = np.array(polygon)
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        croped = img[y:y+h, x:x+w].copy()
        pts = pts - pts.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        return dst

    def get_contours_from_mask(self, mask, min_area=5):
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)
        contour_list = []
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                contour_list.append(contour)
        return contour_list

    def get_larger_contour(self, contours):
        larger_area = 0
        larger_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > larger_area:
                larger_contour = contour
                larger_area = area
        return larger_contour

    
    def get_image_visualization(self, img, pred_data, fontpath, font_koef=50):
        h, w = img.shape[:2]
        font = ImageFont.truetype(fontpath, int(h/font_koef))
        empty_img = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(empty_img)

        for prediction in pred_data['predictions']:
            polygon = prediction['polygon']
            pred_text = prediction['text']
            
            if prediction['class'] not in ['passport', 'photo']:
                cv2.drawContours(img, np.array([polygon]), -1, (0, 255, 255), 1)
                x, y, w, h = cv2.boundingRect(np.array([polygon]))
                draw.text((x, y), pred_text, fill=0, font=font)

        vis_img = np.array(empty_img)
        vis = np.concatenate((img, vis_img), axis=1)
        return vis
    
    def textOutput(self, output):
        
        mainClasses = dotenv_config['MAIN_CLASSES'].split(',')
        textData = {}
        
        
        for i in mainClasses:
            textData[i] = {}
            
        
        for pred in output['predictions']:
            for i in mainClasses:
                if i in pred['class']:
                    textData[i][pred['class']] = pred['text']
                  
        for key, value in textData.items():
            textData[key] = list(zip(list(textData[key].keys()), list(textData[key].values())))
            textData[key] = sorted(textData[key], key = lambda word: word[0])
            textData[key] = ' '.join([i[1] for i in textData[key]])
                    
        return textData

    def contourToRectangle(self, points):
        points = np.array(points).reshape(-1, 2)
        hull = ConvexHull(np.array(points).reshape(-1, 2))
        pp = []
        for simplex in hull.simplices:
            pp.append([points[simplex, 0][0], points[simplex, 1][0]])
            pp.append([points[simplex, 0][1], points[simplex, 1][1]])

        cent = (sum([p[0] for p in pp])/len(pp),
                sum([p[1] for p in pp])/len(pp))
        pp.sort(key=lambda p: math.atan2(p[1]-cent[1], p[0]-cent[0]))

        newpol = np.array(pp, dtype='float32')

        rect = cv2.minAreaRect(np.array(newpol))
        box = cv2.boxPoints(rect)
        newpol = np.int0(box)
        return newpol.reshape(-1, 1, 2)

    def iou_polygon(self, polygon1, polygon2):
        polygon1 = Polygon(polygon1)
        polygon2 = Polygon(polygon2)
        if polygon1.is_valid and polygon2.is_valid:
            intersect = polygon1.intersection(polygon2).area
            union = polygon1.union(polygon2).area
            iou = intersect / union
            return iou
        else:
            return 0
            
    def __call__(self, img):
        output = {'predictions': []}
        outputs = self.segm_predictor(img)
        prediction = outputs['instances'].pred_masks.cpu().numpy()
        classes = outputs['instances'].pred_classes
        namesOfClasses = self.getClasses()
        contours = []

        for i, pred in enumerate(prediction):
            contour_list = self.get_contours_from_mask(pred)

            
            if 'passport' in namesOfClasses[int(classes[i])+1]:
                contours.append(self.get_larger_contour(contour_list))
                # print('self.get_larger_contour(contour_list)', self.get_larger_contour(contour_list).shape)
            else:
                rectangled_contour_list = []
                for contour in contour_list:
                    rectangled_contour_list.append(self.contourToRectangle(contour))

                contours.append(self.get_larger_contour(rectangled_contour_list))


        for i, contour in enumerate(contours):
            if contour is not None:
                crop = self.crop_img_by_polygon(img, contour)
                
                if 'series' in namesOfClasses[int(classes[i])+1]:
                    result = self.ocr_predictor.readtext(np.rot90(crop))
                    pred_text = ' '.join([i[1].upper() for i in result])

                elif "passport" in namesOfClasses[int(classes[i])+1]:
                    pred_text = ''
                else:
                    result = self.ocr_predictor.readtext(crop)
                    pred_text = ' '.join([i[1].upper() for i in result])

                output['predictions'].append(
                    {
                        'polygon': [[int(i[0][0]), int(i[0][1])] for i in contour],
                        'mask': prediction[i],
                        'text': pred_text, 
                        'class': namesOfClasses[int(classes[i])+1]

                    }
                )

        vis = self.get_image_visualization(
            img, output, dotenv_config['FONT_PATH'])
        
        return vis, self.textOutput(output), output


config_json = {
    "alphabet": " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`{|}~«»ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№",
    "image": {
        "width": 256,
        "height": 32
    }
}

SEGM_MODEL_PATH = os.getcwd() + '/' + dotenv_config['SEGM_MODEL_PATH']
OCR_MODEL_PATH = os.getcwd() + '/' + dotenv_config['OCR_MODEL_PATH']

print(SEGM_MODEL_PATH)
print(OCR_MODEL_PATH)

pipeline_predictor = PiepleinePredictor(
    predictor=PageByPageSEGMpredictor,
    segm_model_path=SEGM_MODEL_PATH,
    ocr_model_path=OCR_MODEL_PATH,
    ocr_config=config_json
)
