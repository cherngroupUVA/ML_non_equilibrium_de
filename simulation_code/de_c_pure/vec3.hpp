//
//  vec3.cpp
//  tibidy
//
//  Created by Kipton Barros on 6/19/14.
//
//

#include <cmath>
#include <complex>

template<typename T>
class Vec3 {
public:
    T x=0, y=0, z=0;
    
    Vec3(void) { };
    
    constexpr Vec3(T x, T y, T z): x(x), y(y), z(z) {}
    
    constexpr Vec3<T> operator-() const {
        return {-x, -y, -z};
    }
    
    constexpr Vec3<T> operator+(Vec3<T> that) const {
        return {x+that.x, y+that.y, z+that.z};
    }

    constexpr Vec3<T> operator-(Vec3<T> that) const {
        return {x-that.x, y-that.y, z-that.z};
    }

    constexpr Vec3<T> operator*(T a) const {
        return {x*a, y*a, z*a};
    }
    
    constexpr friend Vec3<T> operator *(T x, Vec3<T> y) {
        return y*x;
    }
    
    constexpr Vec3<T> operator/(T a) const {
        return {x/a, y/a, z/a};
    }
    
    constexpr T dot(Vec3<T> const& that) const {
        return x*that.x + y*that.y + z*that.z;
    }
    
    constexpr Vec3<T> cross(Vec3<T> const& that) const {
        return {y*that.z-z*that.y, z*that.x-x*that.z, x*that.y-y*that.x};
    }
    
    constexpr T norm2() const {
        return dot(*this);
    }
    
    constexpr T norm() const {
        return sqrt(norm2());
    }
    
    constexpr Vec3<T> normalized() const {
        return *this / norm();
    }
    
    void operator+=(Vec3<T> that) {
        x += that.x;
        y += that.y;
        z += that.z;
    }
    
    void operator-=(Vec3<T> that) {
        x -= that.x;
        y -= that.y;
        z -= that.z;
    }
    
    void operator*=(T a) {
        x *= a;
        y *= a;
        z *= a;
    }
    
    void operator/=(T a) {
        x /= a;
        y /= a;
        z /= a;
    }
    
    template <typename S>
    constexpr operator Vec3<S>() const {
        return Vec3<S>(x, y, z);
    }
    
    friend std::ostream& operator<< (std::ostream& os, Vec3<T> const& v) {
        return os << "<x=" << v.x << ", y=" << v.y << ", z=" << v.z << ">";
    }
};

template <typename S>
constexpr Vec3<S> real(Vec3<std::complex<S>> v) {
    return {std::real(v.x), std::real(v.y), std::real(v.z)};
}

template <typename S>
constexpr Vec3<S> imag(Vec3<std::complex<S>> v) {
    return {std::imag(v.x), std::imag(v.y), std::imag(v.z)};
}

typedef Vec3<float> fvec3;
typedef Vec3<double> vec3;
