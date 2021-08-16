# Yolov5Net
YOLOv5 object detection with ML.NET, ONNX

![example](https://github.com/mentalstack/yolov5-net/blob/master/img/result.jpg?raw=true)

## Installation

Run this line from Package Manager Console:

```
Install-Package Yolov5Net -Version 1.0.1
```

## Usage

Yolov5Net contains two COCO pre-trained models: YoloCocoP5Model, YoloCocoP6Model. If you have custom trained model, then inherit from YoloModel and override all required properties and methods, see YoloCocoP5Model or YoloCocoP6Model implementation to get know how to wrap your own model. 

```
using var stream = new FileStream("assets/test.jpg", FileMode.Open);
{
    var image = Image.FromStream(stream);
    var scorer = new YoloScorer(new YoloCocoP5Model("assets/weights/yolov5s.onnx"));
    List<YoloPrediction> predictions = scorer.Predict(image);
}
```

