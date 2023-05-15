#include "assignment/Canvas.hpp"

Canvas::Canvas() {
  _canvas = cv::Mat::zeros(10, 10, CV_8UC1);
  _pCanvas = _canvas.data;
}

Canvas::Canvas(int rows, int cols, int channels = 1) {
  if (channels == 1)
    _canvas = cv::Mat::zeros(rows, cols, CV_8UC1);
  else if (channels == 3)
    // TODO: for 3 channels.. What is pData's offset value for 3 channel
    _canvas = cv::Mat::zeros(rows, cols, CV_8UC3);

  _pCanvas = _canvas.data;
}

cv::Mat &Canvas::get_canvas() { return _canvas; }
void Canvas::set_canvas(cv::Mat &canvas) { _canvas = canvas; }
uchar *Canvas::get_pCanvas() { return _pCanvas; }
void Canvas::set_pCanvas(uchar *pCanvas) { _pCanvas = pCanvas; }

void Canvas::draw_line(cv::Point pt1, cv::Point pt2,
                       const Scalar8u &color = Scalar8u(255)) {
  int channels = _canvas.channels();
  cv::Size size = _canvas.size();
  int width = size.width;
  int height = size.height;

  // draw horizontal. //
  if (pt1.x != pt2.x && pt1.y == pt2.y) {
    const int y_offset_value = pt1.y * width;
    if (pt1.x < pt2.x) { // draw horizontal - left to right
      for (int x = pt1.x; x <= pt2.x; x++) {
        _pCanvas[y_offset_value + x] = color[0]; // 	_pCanvas = _canvas.data;
      }
    } else { // draw horizontal - right to left
      for (int x = pt1.x; x >= pt2.x; x--) {
        _pCanvas[y_offset_value + x] = color[0];
      }
    }
  }
  // draw vertical.
  else if (pt1.x == pt2.x && pt1.y != pt2.y) {
    const int x_offset_value = pt1.x;
    if (pt1.y < pt2.y) { // draw vertical - top to bottom
      for (int y = pt1.y; y <= pt2.y; y++) {
        _pCanvas[y * width + x_offset_value] = color[0];
      }
    } else { // draw vertical - bottom to top
      for (int y = pt1.y; y >= pt2.y; y--) {
        _pCanvas[y * width + x_offset_value] = color[0];
      }
    }
  } else {
    std::cout << "Invalid to draw";
  }

  std::cout << "pt1=" << pt1.x << "," << pt1.y << " ----> "
            << "pt2=" << pt2.x << "," << pt2.y << std::endl;
}

void Canvas::draw_rectangle(cv::Point pt_topLeft, cv::Point pt_btmRight,
                            const Scalar8u &color = Scalar8u(255)) {
  cv::Size size = _canvas.size();
  int width = size.width;
  int height = size.height;

  cv::Point pt[4] = {cv::Point(pt_topLeft.x, pt_topLeft.y),
                     cv::Point(pt_btmRight.x, pt_topLeft.y),
                     cv::Point(pt_btmRight.x, pt_btmRight.y),
                     cv::Point(pt_topLeft.x, pt_btmRight.y)};
  size_t pt_size = std::size(pt);
  for (size_t i = 0; i < pt_size; i++) {
    Canvas::draw_line(pt[i], pt[(i + 1) % pt_size], color);
  }
}
void Canvas::draw_circle(cv::Point pt, int radius,
                         const Scalar8u &color = Scalar8u(255)) {
  cv::Size size = _canvas.size();
  int width = size.width;
  int height = size.height;
  cv::circle(_canvas, cv::Point(width / 2, height / 2), radius, color);
}
