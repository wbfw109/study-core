#pragma once
#include <opencv2/opencv.hpp>
typedef cv::Scalar_<uchar> Scalar8u;

class Canvas {
 private:
  cv::Mat _canvas;
  uchar *_pCanvas = nullptr;

 public:
  Canvas();
  Canvas(int rows, int cols, int channels);

  cv::Mat &get_canvas();
  void set_canvas(cv::Mat &canvas);
  uchar *get_pCanvas();
  void set_pCanvas(uchar *pCanvas);

  void draw_line(cv::Point pt1, cv::Point pt2, const Scalar8u &color);
  void draw_rectangle(cv::Point pt_topLeft, cv::Point pt_btmRight,
                      const Scalar8u &color);
  void draw_circle(cv::Point pt, int radius, const Scalar8u &color);
};
