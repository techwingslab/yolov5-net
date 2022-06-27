using System;
using System.Collections.Generic;
using System.Drawing;
using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models;
using Yolov5Net.Scorer.Models.Abstract;
using OpenCvSharp;

namespace Yolov5Net.App
{
    class webcam_detection
    {
        VideoCapture video_capture;

        public void Init()
        {
            //Initialise the video capture module
            video_capture = new VideoCapture(0);
            video_capture.Set(3, 640); //Set the frame width
            video_capture.Set(4, 480); //Set the frame height

        }

        private Mat GrabFrame()
        {
            Mat image = new Mat();
            //Capture frame by frame
            video_capture.Read(image);
            return image;
        }

        public void DetectFeatures()
        {
            Mat image;
            while (true)
            {
                //Grab the current frame
                image = GrabFrame();

                // Load model.onnx
                using var scorer = new YoloScorer<YoloCocoP5Model>("yolov5s.onnx");

                Bitmap image_for_det;

                image_for_det = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(image);

                // Detection
                List<YoloPrediction> predictions = scorer.Predict(image_for_det);

                foreach (var prediction in predictions) // iterate predictions to draw results
                {
                    double score = Math.Round(prediction.Score, 2);
                    Console.WriteLine(prediction.Label.Name);
                    Console.WriteLine(score);

                    var x1 = prediction.Rectangle.X;
                    var y1 = prediction.Rectangle.Y;
                    var x2 = prediction.Rectangle.X + prediction.Rectangle.Width;
                    var y2 = prediction.Rectangle.Y + prediction.Rectangle.Height;

                    Cv2.Rectangle(image, new OpenCvSharp.Point(x1, y1), new OpenCvSharp.Point(x2, y2) , color: new Scalar(0,255,0), thickness: 1);

                    Cv2.PutText(image, $"{prediction.Label.Name} ({score})", new OpenCvSharp.Point(x1-3, y1-23), color: new Scalar(0, 255, 0), fontFace: HersheyFonts.HersheySimplex, fontScale: 2);


                }

                Cv2.ImShow("frame", image);
                if (Cv2.WaitKey(1) == (int)ConsoleKey.Enter) // Press Enter to exit
                    break;
            }

        }

        public void Release()
        {
            video_capture.Release();
            Cv2.DestroyAllWindows();
        }

        static void Main(string[] args)
        {
            webcam_detection web_det = new webcam_detection();
            web_det.Init();
            web_det.DetectFeatures();
            web_det.Release();

        }
    }

}
