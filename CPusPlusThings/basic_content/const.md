```
// extern_file1.cpp
extern const int ext=12;//不是const的话这个不要加extern
// extern_file2.cpp
#include<iostream>
extern const int ext;
int main(){
    std::cout<<ext<<std::endl;
}
```

void func(int *const var); // 
改变var指向也影响不到实际值

// 此处的虚函数 who()，是通过类（Base）的具体对象（b）来调用的，编译期间就能确定了，所以它可以是内联的，但最终是否内联取决于编译器。 
    Base b;
    b.who();

// 此处的虚函数是通过指针调用的，呈现多态性，需要在运行时期间才能确定，所以不能为内联。  
    Base *ptr = new Derived();
    ptr->who();

// 因为Base有虚析构函数（virtual ~Base() {}），所以 delete 时，会先调用派生类（Derived）析构函数，再调用基类（Base）析构函数，防止内存泄漏。
    delete ptr;
    ptr = nullptr;

为什么空类对象占1字节？
即使类中没有任何成员变量，C++仍然需要给这个对象分配空间，以便它在内存中有一个唯一的地址。这样做有几个目的：

 区分不同对象的地址：如果类对象占据0字节，那么多个对象可能会共享同一个内存地址，这会导致无法区分不同的对象。因此，C++规定空类对象至少占用1字节的空间。
 
 确保类的指针行为一致：即使是空类的指针，指向不同对象时，它们也必须指向不同的内存地址。如果空类对象占0字节，就无法保证这一点。

 
```cpp
#include<iostream>
using namespace std;
class A{
    virtual void fun();
    virtual void fun1();
    virtual void fun2();
    virtual void fun3();
};
int main()
{
    cout<<sizeof(A)<<endl; // 8
    return 0;
}
```

 * @brief 8 8 16  派生类虚继承多个虚函数，会继承所有虚函数的vptr
     ！！！成员函数：成员函数的大小并不直接影响类的对象大小。成员函数的实现是存储在代码段中的，而不是对象实例中的。因此，只有成员变量和虚函数表指针的大小会影响对象的实际大小。

     构造函数不能是虚函数
原因：
对象的创建：构造函数用于初始化对象。当一个对象被创建时，首先调用其构造函数。在调用构造函数时，当前对象的类型已经确定。因此，在构造函数执行期间，虚表（vtable）尚未建立，无法进行虚函数的调用。

虚函数机制：虚函数依赖于对象的动态类型（即运行时类型），而构造函数执行时，当前对象的动态类型尚未完全确定。因此，构造函数不能是虚函数。

```cpp
template <typename T>
auto multiply(T x, T y)->decltype(x*y)
{
	return x*y;
}
do{

}while(0)

inline auto
operator-(const reverse_iterator<_Iterator> &__x,
          const reverse_iterator<_Iterator> &__y)
-> decltype(__x.base() - __y.base())

explicit Fraction1(int num, int den = 1) : m_numerator(num), m_denominator(den) {}

    Fraction1 operator+(const Fraction1 &f)
    {
        return Fraction1(this->m_numerator + f.m_numerator);
    }
```
```c++
class person
{
public:
    int m_A;
    mutable int m_B;//特殊变量 在常函数里值也可以被修改
};

int main()
{
    const person p = person();//修饰常对象 不可修改类成员的值
    p.m_A = 10;//错误，被修饰了指针常量
    p.m_B = 200;//正确，特殊变量，修饰了mutable
}
```

//const类型
const int i = 5;
auto j = i; // 变量i是顶层const, 会被忽略, 所以j的类型是int
auto k = &i; // 变量i是一个常量, 对常量取地址是一种底层const, 所以k的类型是const int*
const auto l = i; //如果希望推断出的类型是顶层const的, 那么就需要在auto前面加上cosnt

这是因为函数模板要被实例化后才能成为真正的函数，在使用函数模板的源文件中包含函数模板的头文件，如果该头文件中只有声明，没有定义，那编译器无法实例化该模板，最终导致链接错误。
