#pragma once
#include <string>
class Student {
 private:
  int _kor, _eng, _math;
  std::string _name;
  double _sum, _avg;

 public:
  Student();
  int get_kor() const;
  void set_kor(const int kor);

  int get_eng() const;
  void set_eng(const int eng);

  int get_math() const;
  void set_math(const int math);

  std::string get_name() const;
  void set_name(const std::string &name);

  double get_sum() const;
  void set_sum(const double sum);

  double get_avg() const;
  void set_avg(const double avg);
};
