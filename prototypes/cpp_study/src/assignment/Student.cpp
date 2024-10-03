#include "assignment/Student.hpp"
Student::Student()  // member initialized
    : _kor(0), _eng(0), _math(0), _name("hong"), _sum(0.0), _avg(0.0) {}

int Student::get_kor() const { return _kor; }
int Student::get_eng() const { return _eng; }
int Student::get_math() const { return _math; }
std::string Student::get_name() const { return _name; }
double Student::get_sum() const { return _sum; }
double Student::get_avg() const { return _avg; }

void Student::set_kor(const int kor) { _kor = kor; }
void Student::set_eng(const int eng) { _eng = eng; }
void Student::set_math(const int math) { _math = math; }
void Student::set_name(const std::string &name) { _name = name; }
void Student::set_sum(const double sum) { _sum = sum; }
void Student::set_avg(const double avg) { _avg = avg; }
