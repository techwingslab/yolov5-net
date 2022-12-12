# Yolov5Net
YOLOv5 object detection with ML.NET, ONNX

![example](https://github.com/mentalstack/yolov5-net/blob/master/img/result.jpg?raw=true)

## Installation

Run this line from Package Manager Console:

```
Install-Package Yolov5Net -Version 1.0.9
```

For CPU usage run this line from Package Manager Console:

```
Install-Package Microsoft.ML.OnnxRuntime -Version 1.13.1
```

For GPU usage run this line from Package Manager Console:

```
Install-Package Microsoft.ML.OnnxRuntime.Gpu -Version 1.13.1
```

CPU and GPU packages can't be installed together.

## Usage

Yolov5Net contains two COCO pre-defined models: YoloCocoP5Model, YoloCocoP6Model. 

If you have custom trained model, then inherit from YoloModel and override all the required properties and methods. See YoloCocoP5Model or YoloCocoP6Model implementation to get know how to wrap your own model. 

```cs
using var image = await Image.LoadAsync<Rgba32>("Assets/test.jpg");

using var scorer = new YoloScorer<YoloCocoP5Model>("Assets/Weights/yolov5n.onnx");

var predictions = scorer.Predict(image);

var font = new Font(new FontCollection().Add(@"C:\Windows\Fonts\consola.ttf"), 16);
foreach (var prediction in predictions) // iterate predictions to draw results
{
    var score = Math.Round(prediction.Score, 2);

    var (x, y) = (prediction.Rectangle.Left - 3, prediction.Rectangle.Top - 23);

    image.Mutate(a => a.DrawPolygon(new Pen(prediction.Label.Color, 1),
        new PointF(prediction.Rectangle.Left, prediction.Rectangle.Top),
        new PointF(prediction.Rectangle.Right, prediction.Rectangle.Top),
        new PointF(prediction.Rectangle.Right, prediction.Rectangle.Bottom),
        new PointF(prediction.Rectangle.Left, prediction.Rectangle.Bottom)
    ).DrawText($"{prediction.Label.Name} ({score})",
        font,
        prediction.Label.Color,
        new PointF(x, y)));
}

await image.SaveAsync("Assets/result.jpg");
```
