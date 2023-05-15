
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "assignment/Canvas.hpp"
#include "assignment/Student.hpp"
#include "cpp_study.hpp"

#define PIXEL_WHITE 255
#define PIXEL_BLACK 0

void test_print_string() {
  std::vector<std::string> msg{"Hello", "C++",     "World",
                               "from",  "VS Code", "and the C++ extension!"};

  for (const std::string &word : msg) {
    std::cout << word << " ";
  }
  std::cout << std::endl;
}
void proceses_stduents_score() {
  std::vector<Student> students;
  uint8_t y_or_n;

  do {
    Student student;
    std::string temp_str("");
    int temp_int = 0;

    std::cout << "학생 이름 입력을 받으세요: ";
    std::cin >> temp_str;
    student.set_name(temp_str);

    std::cout << "국어성적을 입력하시오: ";
    std::cin >> temp_int;
    student.set_kor(temp_int);

    std::cout << "영어성적을 입력하시오: ";
    std::cin >> temp_int;
    student.set_eng(temp_int);

    std::cout << "수학성적을 입력하시오: ";
    std::cin >> temp_int;
    student.set_math(temp_int);

    student.set_sum(student.get_kor() + student.get_eng() + student.get_math());
    student.set_avg(student.get_sum() / 3);

    // ● std::format requires >= c++20 std
    std::cout << std::format("{} 학생의 과목별 합계, 평균은 {}, {} 입니다.\n",
                             student.get_name(), student.get_sum(),
                             student.get_avg());
    std::cout << "추가 학생 정보를 입력하겠습니까? Y or N: ";
    std::cin >> y_or_n;

    std::cout << std::endl;

    students.push_back(student);  // shallow copy the element !!
  } while (y_or_n == 'Y' || y_or_n == 'y');

  int kor_max_i = 0, eng_max_i = 0, math_max_i = 0;
  size_t student_count = students.size();
  for (size_t i = 1; i < student_count; i++) {
    if (students[i].get_kor() > students[kor_max_i].get_kor()) {
      kor_max_i = i;
    }
    if (students[i].get_eng() > students[eng_max_i].get_eng()) {
      eng_max_i = i;
    }
    if (students[i].get_math() > students[math_max_i].get_math()) {
      math_max_i = i;
    }
  }

  for (const auto &student : students) {
    std::cout << std::format("{} 학생의 과목별 합계, 평균은 {}, {} 입니다.\n",
                             student.get_name(), student.get_sum(),
                             student.get_avg());
  }
  std::cout << "국어 성적 최우수 학생은 " << students[kor_max_i].get_name()
            << " 학생입니다.\n";
  std::cout << "영어 성적 최우수 학생은 " << students[eng_max_i].get_name()
            << " 학생입니다.\n";
  std::cout << "수학 성적 최우수 학생은 " << students[math_max_i].get_name()
            << " 학생입니다.\n";
}

void show_plus_line_with_even_size_image() {
  std::cout << "even_size_image" << std::endl;
  int cols = 600;
  int rows = 400;
  cv::Mat img = cv::Mat::zeros(rows, cols, CV_8UC1);
  cv::Size size = img.size();
  int width = size.width;
  int height = size.height;
  uchar *pData = img.data;

  int width_minus_1 = width - 1;
  for (int y = 0; y < height; ++y) {  // Memory Locality !! optimzation
    for (int x = 0; x < width; ++x) {
      if (x == y || x + y == width_minus_1) {
        pData[y * width + x] = std::min(y + 1, PIXEL_WHITE);
      }
    }
  }
  return;
}

void test_image_show() {
  cv::Mat canvas1 = cv::Mat::zeros(1000, 1000, CV_8UC1);
  cv::Mat colorImage = cv::imread("", cv::ImreadModes::IMREAD_UNCHANGED);

  std::string nameWindow = "Show";
  cv::namedWindow(nameWindow);
  cv::imshow(nameWindow, canvas1);
  cv::waitKey();
  cv::destroyAllWindows();
}

void test_downcasintg() {
  // Downcasting ; https://en.wikipedia.org/wiki/Downcasting

  class Parent {
   public:
    virtual void show() { std::cout << "I'm Parent class" << std::endl; }
  };

  class Child : public Parent {
   public:
    void show() override { std::cout << "I'm Child class" << std::endl; }
    void childSpecificFunction() {
      std::cout << "This function is only in Child class" << std::endl;
    }
  };

  // lambda function
  auto process_downcasting = [](Parent &p) {
    // Try Downcasting
    Child *c = dynamic_cast<Child *>(&p);

    if (c) {  // if downcasting succeeds
      c->childSpecificFunction();
    } else {
      std::cout << "The object is not of type Child" << std::endl;
    }
  };

  // test downcasting
  Child child;
  Parent parent;

  process_downcasting(child);   // Child 객체를 전달
  process_downcasting(parent);  // Parent 객체를 전달
}

// void test_upcasting()
// {
//     class IDrawer
//     {
//         virtual ~IDrawer();

//         virtual void draw_line(cv::Point pt1, cv::Point pt2) { std::cout <<
//         "IDrawer::draw" << std::endl; }
//     };

//     class DrawTxt : public IDrawer
//     {
//         virtual ~DrawTxt();
//         void draw_line(cv::Point pt1, cv::Point pt2)
//         {
//             std::cout << "DrawTxt::draw_line" << std::endl;
//             std::cout << "pt1=" << pt1.x << "," << pt1.y << " ----> " <<
//             "pt2=" << pt2.x << "," << pt2.y << std::endl;
//         }
//     };

//     class DrawMat : public IDrawer
//     {
//         virtual ~DrawMat();
//         void set_image(const cv::Mat &img)
//         {
//             _img = img;
//         }
//         void draw_line(cv::Point pt1, cv::Point pt2)
//         {
//             std::cout << "DrawMat::draw_line" << std::endl;
//             cv::Size size = _img.size();
//             int width = size.width;
//             int height = size.height;
//             uchar *pData = _img.data;
//                 }
//     }
// }

int arr[3] = {1, 2, 3};

int (&foo())[3] { return arr; }
int main() {
  int(&myArray)[3] = foo();

  std::cout
      << "Hello World! Not Slow in Linux with Ninja, Ccache.. OO.. maybe; \n";
  // void (*pfnCalc)(void); // Function Pointer ●; plug-in.
  // pfnCalc = Calc;
  // proceses_stduents_score();

  // show_nine_nine_table_image();
  // show_plus_line_with_coners_image();
  // show_plus_line_with_even_size_image();

  // cpp_study();

  // std::vector<std::string> vec;
  // vec.push_back("test_package");

  // cpp_study_print_vector(vec);

  //// dynamic_cast, static_cast, const_cast, reinterpret_cast

  // ● img.at<uchar>(x, y) 에서 row, column 순이다. 즉, height, width 순서.
  // ● img.at 이 아니라, pData[row*width + column] 인덱싱으로 접근 가능하다.

  return 0;
}
