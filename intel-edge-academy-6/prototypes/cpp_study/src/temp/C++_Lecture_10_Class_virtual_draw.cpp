/*
class IDrawer
{
        virtual void draw_line(Point pt1, Point
pt2){cout<<"IDrawer::draw"<<endl;}
}

class DrawTxt:public IDrawer
{
        void draw_line(Point pt1, Point pt2)
        {
                cout<<"DrawTxt::draw_line"<<endl;
                std::cout << "pt1=" << pt1.x << "," << pt1.y << " ----> " <<
"pt2=" << pt2.x << "," << pt2.y << std::endl;
        }
}
class DrawMat:public IDrawer
{
        void set_image(const Mat& img)
        {
                _img = img;
        }
        void draw_line(Point pt1, Point pt2)
        {
                cout<<"DrawMat::draw_line"<<endl;
                cv::Size size = _img.size();
                int width = size.width;
                int height = size.height;
                uchar* pData = _img.data;
}
*/

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "ColorDefine.h"
using namespace std;
using namespace cv;

class IDrawer  // IDrawer = Interface + Class Drawer
{
 public:
  IDrawer() { cout << "IDrawer::Ctor" << endl; }
  virtual ~IDrawer() { cout << "IDrawer::Dtor" << endl; }
  virtual void line(Point pt1, Point pt2) = 0;
  virtual void circle(Point pt1, size_t radius) = 0;
  virtual void rectangle(Point pt_tl, Point pt_br) = 0;
  virtual void polygon(std::vector<Point> pts) = 0;
  virtual void txt(Point pt, std::string msg) = 0;
};

class DrawTxt : public IDrawer {
 public:
  DrawTxt() { cout << "DrawTxt::Ctor" << endl; }
  ~DrawTxt() { cout << "DrawTxt::Dtor" << endl; }
  void line(Point pt1, Point pt2) override {
    cout << "DrawTxt::line" << endl;
    std::cout << "pt1=" << pt1.x << "," << pt1.y
              << " <��������������������������������������������������> "
              << "pt2=" << pt2.x << "," << pt2.y << std::endl;
  };
  void circle(Point pt1, size_t radius) override {
    cout << "DrawTxt::circle" << endl;
    std::cout << "pt1=" << pt1.x << "," << pt1.y << " O " << "radius=" << radius
              << std::endl;
  };
  void rectangle(Point pt_tl, Point pt_br) override {
    cout << "DrawTxt::rectangle" << endl;
    std::cout << "pt1=" << pt_tl.x << "," << pt_tl.y << " �� "
              << "pt2=" << pt_br.x << "," << pt_br.y << std::endl;
  };
  void polygon(std::vector<Point> pts) override {
    cout << "DrawTxt::polygon" << endl;
    std::cout << "      *        " << std::endl;
    std::cout << "    *   *      " << std::endl;
    std::cout << "   *     *      " << std::endl;
    std::cout << "    *   *      " << std::endl;
    std::cout << "      *        " << std::endl;
  };
  void txt(Point pt, std::string msg) override {
    cout << "DrawTxt::txt" << endl;
    std::cout << "pt=" << pt.x << "," << pt.y << " hello world c++ "
              << "msg=" << msg << std::endl;
  };
};

class DrawMat : public IDrawer {
  Mat draw_img;

 public:
  DrawMat() { cout << "DrawMat::Ctor" << endl; }
  DrawMat(const Mat& color_img) : draw_img(color_img) {
    cout << "DrawMat::Ctor" << endl;
  }
  ~DrawMat() { cout << "DrawMat::Dtor" << endl; }
  void line(Point pt1, Point pt2) override {
    cout << "DrawMat::line" << endl;
    unsigned int seed = static_cast<unsigned int>(
        std::chrono::system_clock::now().time_since_epoch().count());
    RNG rng(seed);
    Scalar color =
        Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    cv::line(draw_img, pt1, pt2, color);
  };
  void circle(Point pt1, size_t radius) override {
    cout << "DrawMat::circle" << endl;
    unsigned int seed = static_cast<unsigned int>(
        std::chrono::system_clock::now().time_since_epoch().count());
    RNG rng(seed);
    Scalar color =
        Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    cv::circle(draw_img, pt1, radius, color);
  };
  void rectangle(Point pt_tl, Point pt_br) override {
    cout << "DrawMat::rectangle" << endl;
    unsigned int seed = static_cast<unsigned int>(
        std::chrono::system_clock::now().time_since_epoch().count());
    RNG rng(seed);
    Scalar color =
        Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    cv::rectangle(draw_img, pt_tl, pt_br, color);
  };
  void polygon(std::vector<Point> pts) override {
    cout << "DrawMat::polygon" << endl;
    unsigned int seed = static_cast<unsigned int>(
        std::chrono::system_clock::now().time_since_epoch().count());
    RNG rng(seed);
    Scalar color =
        Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

    for (size_t i = 0; i < pts.size(); i++) {
      cv::line(draw_img, pts[i % pts.size()], pts[(i + 1) % pts.size()], color);
    }
  };
  void txt(Point pt, std::string msg) override {
    cout << "DrawMat::txt" << endl;
    unsigned int seed = static_cast<unsigned int>(
        std::chrono::system_clock::now().time_since_epoch().count());
    RNG rng(seed);
    int fontFace = FONT_HERSHEY_COMPLEX;
    double fontScale = 2;
    int thickness = 2;
    Size textsize = getTextSize(msg, fontFace, fontScale, thickness, 0);
    Scalar color =
        Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    cv::putText(draw_img, msg, Point(pt.x, pt.y + textsize.height), fontFace,
                fontScale, color, thickness, cv::LINE_8);
  };
};

int main() {
  // IDrawer* pDrawer = nullptr;
  cv::Mat color_img = cv::Mat(400, 600, CV_8UC3);
  color_img = 0;

  {
    Point pt1(150, 120);
    Point pt2(10, color_img.rows / 2);

    std::vector<Point> vPt;
    vPt.push_back(Point(66, 97));
    vPt.push_back(Point(273, 33));
    vPt.push_back(Point(452, 120));
    vPt.push_back(Point(383, 330));
    vPt.push_back(Point(130, 280));
    vPt.push_back(Point(45, 160));

    // IDrawer* pDrawer[2] = { new DrawTxt, new DrawMat(color_img) };

    std::vector<IDrawer*> vDraw;
    vDraw.push_back(new DrawTxt);
    vDraw.push_back(new DrawMat(color_img));

    for (const auto& draw : vDraw) {
      draw->circle(pt1, 100);
      draw->line(pt1, pt2);
      draw->polygon(vPt);
      draw->rectangle(pt1, pt2);
      draw->txt(pt2, "KCCI C++ Class");
    }

    for (const auto& draw : vDraw) delete draw;
  }
  return 1;
}