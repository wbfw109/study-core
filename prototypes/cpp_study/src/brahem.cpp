#include "brahem.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

class DrawCanvas {
 private:
  cv::Mat canvas;
  uchar *pCanvas = nullptr;

 public:
  DrawCanvas() {};
  DrawCanvas(int rows, int cols, int channels = 1) {
    if (channels == 1)
      canvas = cv::Mat::zeros(rows, cols, CV_8UC1);
    else if (channels == 3)
      canvas = cv::Mat::zeros(rows, cols, CV_8UC3);
    pCanvas = canvas.data;
  }
  ~DrawCanvas() {}

  const cv::Mat &GetCanvas() { return canvas; }
  void Draw_line(cv::Point pt1, cv::Point pt2, cv::Scalar color = 255) {
    int x0 = pt1.x;
    int y0 = pt1.y;
    int x1 = pt2.x;
    int y1 = pt2.y;

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);

    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;

    int err = dx - dy;
    int e2;

    while (true) {
      pCanvas[y0 * canvas.cols + x0] = color[0];
      if (x0 == x1 && y0 == y1) break;

      // Bresenham
      e2 = 2 * err;

      if (e2 > -dy) {
        err -= dy;
        x0 += sx;
      }

      if (e2 < dx) {
        err += dx;
        y0 += sy;
      }
    }

    // dx = abs(x1 - x0)
    // sx = x0 < x1 ? 1 : -1
    // dy = -abs(y1 - y0)
    // sy = y0 < y1 ? 1 : -1
    // error = dx + dy

    // while true
    //     plot(x0, y0)
    //     if x0 == x1 && y0 == y1 break
    //     e2 = 2 * error
    //     if e2 >= dy
    //         if x0 == x1 break
    //         error = error + dy
    //         x0 = x0 + sx
    //     end if
    //     if e2 <= dx
    //         if y0 == y1 break
    //         error = error + dx
    //         y0 = y0 + sy
    //     end if
    // end while
  }
  // 🚣
  void Draw_circle(cv::Point pt, int radius, cv::Scalar color = 255) {
    const double PI = 3.14159265358979323846;
    int points = 360 * 3;
    double angleStep = 2 * PI / points;  // 360 degree = 2 * pi Radian

    for (int i = 0; i < points; i++) {
      double angle = i * angleStep;
      int drawX = static_cast<int>(pt.x + radius * cos(angle));
      int drawY = static_cast<int>(pt.y + radius * sin(angle));
      pCanvas[drawY * canvas.cols + drawX] = color[0];
    }
  }
  void Draw_rectangle(cv::Point pt_topLeft, cv::Point pt_btmRight,
                      cv::Scalar color = 255) {
    Draw_line(cv::Point(pt_topLeft.x, pt_topLeft.y),
              cv::Point(pt_btmRight.x, pt_topLeft.y), color);
    Draw_line(cv::Point(pt_btmRight.x, pt_topLeft.y),
              cv::Point(pt_btmRight.x, pt_btmRight.y), color);
    Draw_line(cv::Point(pt_btmRight.x, pt_btmRight.y),
              cv::Point(pt_topLeft.x, pt_btmRight.y), color);
    Draw_line(cv::Point(pt_topLeft.x, pt_btmRight.y),
              cv::Point(pt_topLeft.x, pt_topLeft.y), color);
  }
};

// int main()
// {
// 	int rows = 400;
// 	int cols = 600;
// 	int channels = 1;//gray-1,color-3
// 	DrawCanvas dCan(rows, cols, channels);

// 	const cv::Mat& canvasImage = dCan.GetCanvas();

// 	cv::Point pt(cols / 2 - 1, rows / 2 - 1);
// 	int radius = std::min(rows, cols) / 3;
// 	dCan.Draw_circle(pt, radius);

// 	int gap = 10;
// 	cv::Point ptArry[4] = { cv::Point(0 + gap,0 + gap),
// 		cv::Point(cols - 1 - gap,0 + gap),
// 		cv::Point(cols - 1 - gap,rows - 1 - gap),
// 		cv::Point(0 + gap,rows - 1 - gap) };

// 	dCan.Draw_line(ptArry[0], ptArry[1], cv::Scalar(255));
// 	dCan.Draw_line(ptArry[0], ptArry[2], cv::Scalar(255));
// 	dCan.Draw_line(ptArry[0], ptArry[3], cv::Scalar(255));
// 	dCan.Draw_rectangle(ptArry[0], ptArry[2],cv::Scalar(255));

// 	//debug point, check image watch
// 	int a = 0;

// 	return 1;
// }