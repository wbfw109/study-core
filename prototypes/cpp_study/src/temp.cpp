#include <iostream>
using namespace std;

int hello_world(){
    printf("Hello world!!\n");
    return 0;
}
// TODO: dynamic_cast, static_cast, const_cast, reinterpret_cast

// TODO: Downcasting ; https://en.wikipedia.org/wiki/Downcasting
//  class Parent
//  {
//  public:
//      virtual void show()
//      {
//          cout << "I'm Parent class" << endl;
//      }
//  };

// class Child : public Parent
// {
// public:
//     void show() override
//     {
//         cout << "I'm Child class" << endl;
//     }
//     void childSpecificFunction()
//     {
//         cout << "This function is only in Child class" << endl;
//     }
// };

// void process_downcasting(Parent &p)
// {
//     // Try Downcasting
//     Child *c = dynamic_cast<Child *>(&p);

//     if (c)
//     { // if downcasting succeeds
//         c->childSpecificFunction();
//     }
//     else
//     {
//         cout << "The object is not of type Child" << endl;
//     }
// }

// int test_downcasting()
// {
//     Child child;
//     Parent parent;

//     process_downcasting(child);  // Child 객체를 전달
//     process_downcasting(parent); // Parent 객체를 전달

//     return 0;
// }

// Class Inheritance: Up-casting, Down-casting

// 부모 포인터 = dynamic_cast<부모 클래스 포인터>(자식 클래스 참조); //up
// casting? #include <iostream> #include <string> #include <vector> #include
// <opencv2/opencv.hpp> #include <algorithm>

// /*
//     cv::Mat
//         //! pointer to the data
//         uchar* data;
// */
// void Calc()
// {
//     std::cout << "Function::Calc()" << std::endl;
// }
// // input : 1 ea, output : 1 ea
// int Calc(int x)
// {
//     std::cout << "Function::Calc(int x)" << std::endl;
//     int a = 10;
//     int b = 23;
//     int y = a * x + b;
//     return y;
// }
// // input : N ea, output : N ea

// int Calc(int &a, int b, int &x, int *y = nullptr)
// {
//     std::cout << "Function::Calc(int& a, int b, int& x, int* y=nullptr)" <<
//     std::endl; *y = a * x + b; return 1;
// }

// // TODO.... something wrong
// void show_plus_line_with_coners_image()
// {

//     std::cout << "+ White" << std::endl;
//     // paint + to white
//     int rows = 9;
//     int cols = 9;
//     cv::Mat img = cv::Mat::zeros(rows, cols, CV_8UC1);
//     cv::Size size = img.size();
//     int width = size.width;
//     int height = size.height;
//     uchar *pData = img.data;

//     int cols_mid = cols / 2;
//     int rows_mid = rows / 2;

//     // row+offsetY * width + (col+offset)
//     //!!!! int x_offset = std::abs(cols_mid - x);	// 짝수이면 중앙값을
//     아님..

//     int center_i_value = height * rows_mid + cols_mid;

//     for (int x = 0; x < cols; ++x)
//     {
//         for (int y = 0; y < rows; ++y)
//         {
//             int i_value = width * y + x;
//             int abs_i_value = std::abs(i_value - center_i_value);

//             if (abs_i_value % width == 0)
//             {
//                 pData[width * y + x] = PIXEL_WHITE;
//             }
//             else if (abs_i_value <= cols_mid)
//             {
//                 pData[width * y + x] = PIXEL_WHITE;
//             }
//             else if (abs_i_value == center_i_value || abs_i_value ==
//             center_i_value - width + 1)
//             {
//                 pData[width * y + x] = PIXEL_WHITE;
//             }
//         }
//     }
//     return;
// }
// // TODO.... something wrong
// void show_nine_nine_table_image()
// {
//     std::cout << "+ 9*9 table" << std::endl;
//     int rows = 10;
//     int cols = 10;
//     cv::Mat img = cv::Mat::zeros(rows, cols, CV_8UC1);
//     int width = img.size().width;
//     uchar *pData = img.data;
//     for (int x = 0; x < cols; ++x)
//     {
//         for (int y = 0; y < rows; ++y)
//         {
//             pData[x * width + y] = x * y;
//         }
//     }
//     return;
// }

// /////////////////////
// void GuGuDan()
// {
//     for (size_t i = 2; i < 10; i++)
//     {
//         for (size_t j = 2; j < 10; j++)
//         {
//             std::cout << j << " * " << i << " = " << i * j << "\t";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;
// }
// void GuGuDan(size_t startDan, size_t finishDan, size_t startStep, size_t
// finishStep)
// {
//     for (size_t i = startDan; i < finishDan + 1; i++)
//     {
//         for (size_t j = startStep; j < finishStep + 1; j++)
//         {
//             std::cout << j << " * " << i << " = " << i * j << "\t";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;
// }

// std::string GuGuDans(size_t startDan, size_t finishDan, size_t startStep,
// size_t finishStep)
// {
//     std::string lines;
//     for (size_t i = startDan; i < finishDan + 1; i++)
//     {
//         std::string line;
//         for (size_t j = startStep; j < finishStep + 1; j++)
//         {
//             // string interpolation
//             line += std::format("{} * {} = {}\t", std::to_string(j),
//             std::to_string(i), std::to_string(i * j));
//         }
//         lines += line + "\n";
//     }
//     return lines;
// }

// /////////////////////
// //  typedef Point2i Point;		// in cv::mat.hpp

// void runCanvasProcess()
// {
//     Canvas canvas1 = Canvas();
//     Canvas canvas2 = Canvas();
//     Canvas canvas3 = Canvas();

//     canvas1.draw_rectangle(cv::Point(0, 0), cv::Point(8, 8), Scalar8u(255));
//     canvas1.draw_circle(cv::Point(0, 0), cv::Point(8, 8), Scalar8u(255)); //
//     분해능?

//     return;
// }

// class GuGuDan
// {
//     // public, protected, private:
// public:
//     GuGuDan()
//         : a_public(0), a_private(0), a_protected(0), _startDan(0),
//         _finishDan(0), _startStep(0), _finishStep(0)
//     {
//         // class creator
//         std::cout << "GuGuDan::Ctor" << std::endl;
//     }
//     GuGuDan(const int &startDan,
//             const int &finishDan,
//             const int &startStep,
//             const int &finishStep)
//         : _startDan(startDan), _finishDan(finishDan), _startStep(startStep),
//         _finishStep(finishStep)
//     {
//         // class creator
//         std::cout << "GuGuDan::Ctor" << std::endl;
//     }
//     GuGuDan(const int &startDan,
//             const int &finishDan)
//         : GuGuDan(startDan, finishDan, 1, 9)
//     {
//         std::cout << "GuGuDan::Ctor" << std::endl;
//     }

//     ~GuGuDan()
//     {
//         // class destroy
//         std::cout << "GuGuDan::Dtor" << std::endl;
//     }
//     void setParams(const int &startDan,
//                    const int &finishDan,
//                    const int &startStep,
//                    const int &finishStep)
//     {
//         _startDan = startDan;
//         _finishDan = finishDan;
//         _startStep = startStep;
//         _finishStep = finishStep;
//     }
//     std::string Do()
//     {
//         std::cout << "Call Function::string GuGuDan(int startDan, int
//         finishDan, int startStep, int finishStep)" << std::endl; std::string
//         lines; for (size_t step = _startStep; step <= _finishStep; step++)
//         {
//             std::string line = "";
//             for (size_t dan = _startDan; dan <= _finishDan; dan++)
//             {
//                 int result = dan * step;
//                 line += std::to_string(dan);
//                 line += " x ";
//                 line += std::to_string(step);
//                 line += " = ";
//                 line += std::to_string(result);
//                 line += "\t";
//             }
//             lines += line + "\n";
//         }
//         return lines;
//     }

// public:
//     int a_public = 0;

// private:
//     int a_private = 0;

//     int _startDan = 0;
//     int _finishDan = 0;
//     int _startStep = 0;
//     int _finishStep = 0;

// protected:
//     int a_protected = 0;
// };
