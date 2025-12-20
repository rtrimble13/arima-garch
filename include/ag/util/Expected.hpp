#pragma once

#include <version>

// Check if std::expected is available (C++23)
#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
    #include <expected>
    namespace ag {
        using std::expected;
        using std::unexpected;
    }
#else
    // Fallback implementation for C++20
    #include <type_traits>
    #include <utility>
    #include <exception>
    #include <stdexcept>

    namespace ag {
        
        // Simple unexpected wrapper for error values
        template<typename E>
        class unexpected {
        public:
            constexpr explicit unexpected(const E& e) : error_(e) {}
            constexpr explicit unexpected(E&& e) : error_(std::move(e)) {}
            
            constexpr const E& error() const& noexcept { return error_; }
            constexpr E& error() & noexcept { return error_; }
            constexpr const E&& error() const&& noexcept { return std::move(error_); }
            constexpr E&& error() && noexcept { return std::move(error_); }
            
        private:
            E error_;
        };
        
        // Simple expected implementation for C++20
        template<typename T, typename E>
        class expected {
        public:
            using value_type = T;
            using error_type = E;
            
            // Constructors for value
            constexpr expected() : has_value_(true) {
                new (&value_) T();
            }
            
            constexpr expected(const T& value) : has_value_(true) {
                new (&value_) T(value);
            }
            
            constexpr expected(T&& value) : has_value_(true) {
                new (&value_) T(std::move(value));
            }
            
            // Constructor for error
            constexpr expected(const unexpected<E>& unexp) : has_value_(false) {
                new (&error_) E(unexp.error());
            }
            
            constexpr expected(unexpected<E>&& unexp) : has_value_(false) {
                new (&error_) E(std::move(unexp.error()));
            }
            
            // Copy constructor
            constexpr expected(const expected& other) : has_value_(other.has_value_) {
                if (has_value_) {
                    new (&value_) T(other.value_);
                } else {
                    new (&error_) E(other.error_);
                }
            }
            
            // Move constructor
            constexpr expected(expected&& other) noexcept : has_value_(other.has_value_) {
                if (has_value_) {
                    new (&value_) T(std::move(other.value_));
                } else {
                    new (&error_) E(std::move(other.error_));
                }
            }
            
            // Destructor
            ~expected() {
                if (has_value_) {
                    value_.~T();
                } else {
                    error_.~E();
                }
            }
            
            // Copy assignment
            expected& operator=(const expected& other) {
                if (this != &other) {
                    if (has_value_ && other.has_value_) {
                        value_ = other.value_;
                    } else if (!has_value_ && !other.has_value_) {
                        error_ = other.error_;
                    } else if (has_value_ && !other.has_value_) {
                        value_.~T();
                        new (&error_) E(other.error_);
                        has_value_ = false;
                    } else {  // !has_value_ && other.has_value_
                        error_.~E();
                        new (&value_) T(other.value_);
                        has_value_ = true;
                    }
                }
                return *this;
            }
            
            // Move assignment
            expected& operator=(expected&& other) noexcept {
                if (this != &other) {
                    if (has_value_ && other.has_value_) {
                        value_ = std::move(other.value_);
                    } else if (!has_value_ && !other.has_value_) {
                        error_ = std::move(other.error_);
                    } else if (has_value_ && !other.has_value_) {
                        value_.~T();
                        new (&error_) E(std::move(other.error_));
                        has_value_ = false;
                    } else {  // !has_value_ && other.has_value_
                        error_.~E();
                        new (&value_) T(std::move(other.value_));
                        has_value_ = true;
                    }
                }
                return *this;
            }
            
            // Check if has value
            constexpr bool has_value() const noexcept { return has_value_; }
            constexpr explicit operator bool() const noexcept { return has_value_; }
            
            // Value accessors
            constexpr const T& value() const& {
                if (!has_value_) {
                    throw std::runtime_error("Expected does not contain a value");
                }
                return value_;
            }
            
            constexpr T& value() & {
                if (!has_value_) {
                    throw std::runtime_error("Expected does not contain a value");
                }
                return value_;
            }
            
            constexpr const T&& value() const&& {
                if (!has_value_) {
                    throw std::runtime_error("Expected does not contain a value");
                }
                return std::move(value_);
            }
            
            constexpr T&& value() && {
                if (!has_value_) {
                    throw std::runtime_error("Expected does not contain a value");
                }
                return std::move(value_);
            }
            
            // Error accessors
            constexpr const E& error() const& {
                if (has_value_) {
                    throw std::runtime_error("Expected contains a value, not an error");
                }
                return error_;
            }
            
            constexpr E& error() & {
                if (has_value_) {
                    throw std::runtime_error("Expected contains a value, not an error");
                }
                return error_;
            }
            
            // Dereference operators
            constexpr const T* operator->() const { return &value_; }
            constexpr T* operator->() { return &value_; }
            constexpr const T& operator*() const& { return value_; }
            constexpr T& operator*() & { return value_; }
            constexpr const T&& operator*() const&& { return std::move(value_); }
            constexpr T&& operator*() && { return std::move(value_); }
            
            // value_or
            template<typename U>
            constexpr T value_or(U&& default_value) const& {
                return has_value_ ? value_ : static_cast<T>(std::forward<U>(default_value));
            }
            
            template<typename U>
            constexpr T value_or(U&& default_value) && {
                return has_value_ ? std::move(value_) : static_cast<T>(std::forward<U>(default_value));
            }
            
        private:
            bool has_value_;
            union {
                T value_;
                E error_;
            };
        };
        
    }  // namespace ag
#endif
