#pragma once
#include "compiler_directives.hpp"
#include "expressions.hpp"
#include <cassert>
#include <vector>
#include <span>
#include <algorithm>

namespace numPDE
{
     template <typename T>
    struct Vector : Expr<Vector<T>>
    {
      public:
        using value_type = T;
        Vector()         = default;
        Vector(std::initializer_list<T> l)
        {
            m_Datas.resize(l.size());
            std::copy_n(l.begin(), l.size(), m_Datas.begin());
        }
        Vector(size_t s, T val) : m_Datas(s, val) {}

        template <typename E>
        Vector(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
            m_Datas.resize(ex.size());

            for (size_t i = 0; i < ex.size(); ++i)
                m_Datas[i] = ex[i];
        }
        // -----------------------------//
        // ***** ACCESS OPERATORS ***** //
        // -----------------------------//
        const T& operator[](size_t i) const { return m_Datas[i]; }
        T&       operator[](size_t i) { return m_Datas[i]; }

        // -----------------------------//
        // ***** CONSTR FROM EXPR ***** //
        // -----------------------------//
        template <typename E>
        auto operator()(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
            m_Datas.resize(ex.size());
            for (size_t i = 0; i < ex.size(); ++i)
                m_Datas[i] = ex[i];
        }
        template <typename E>
        auto operator=(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);

            m_Datas.resize(ex.size());
            for (size_t i = 0; i < ex.size(); ++i)
                m_Datas[i] = ex[i];
            return (*this);
        }
        // -----------------------------//
        // ***** CONSTR FROM SPAN ***** // aka from VectorField
        // -----------------------------//
        auto operator=(std::span<T> span)
        {
            auto span_clean = static_cast<std::span<T>>(span);
            std::copy_n(span_clean.begin(), span_clean.size(), m_Datas.begin());
            return *this;
        }
        auto operator=(std::span<const T> span)
        {
            auto span_clean = static_cast<std::span<const T>>(span);
            std::copy_n(span_clean.begin(), span_clean.size(), m_Datas.begin());
            return *this;
        }

        // auto operator
        // std::copy_n(span.begin(), N_dim, test.begin());

        // -----------------------------//
        // *****STL-LIKE UTILITIES***** //
        // -----------------------------//
        size_t constexpr size() const { return m_Datas.size(); }
        decltype(auto) begin() { return m_Datas.begin(); }
        decltype(auto) end() { return m_Datas.end(); }
        decltype(auto) begin() const { return m_Datas.begin(); }
        decltype(auto) end() const { return m_Datas.end(); }

      private:
        std::vector<T> m_Datas{};
    };

     template <typename T, size_t N = DEF_DIM>
    struct Array : Expr<Array<T, N>>
    {
      public:
        using value_type = T;
        Array()          = default;
        Array(const T v) { std::fill_n(m_Datas.begin(), N, v); }
        Array(std::initializer_list<T> l)
        {
            assert(l.size() <= N && "Value bigger than the size of the  element");
            std::copy_n(l.begin(), l.size(), m_Datas.begin());
        }

        // -----------------------------//
        // ***** ACCESS OPERATORS ***** //
        // -----------------------------//
        const T& operator[](size_t i) const { return m_Datas[i]; }
        T&       operator[](size_t i) { return m_Datas[i]; }

        // -----------------------------//
        // ***** CONSTR FROM EXPR ***** //
        // -----------------------------//
        template <typename E>
        auto operator()(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
#ifdef PEDANTIC
            assert("Size mismatch" && ex.size() == N);
#endif
            for (size_t i = 0; i < ex.size(); ++i)
                m_Datas[i] = ex[i];
        }
        template <typename E>
        auto operator=(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);

#ifdef PEDANTIC
            assert("Size mismatch" && ex.size() == N);
#endif
            for (size_t i = 0; i < ex.size(); ++i)
                m_Datas[i] = ex[i];
            return (*this);
        }
        // -----------------------------//
        // ***** CONSTR FROM SPAN ***** // aka from VectorField
        // -----------------------------//
        auto operator=(std::span<T> span)
        {
            auto span_clean = static_cast<std::span<T>>(span);
            std::copy_n(span_clean.begin(), N, m_Datas.begin());
            return *this;
        }
        auto operator=(std::span<const T> span)
        {
            auto span_clean = static_cast<std::span<const T>>(span);
            std::copy_n(span_clean.begin(), N, m_Datas.begin());
            return *this;
        }

        // auto operator
        // std::copy_n(span.begin(), N_dim, test.begin());

        // -----------------------------//
        // *****STL-LIKE UTILITIES***** //
        // -----------------------------//
        size_t constexpr size() const { return N; }
        decltype(auto) begin() { return m_Datas.begin(); }
        decltype(auto) end() { return m_Datas.end(); }
        decltype(auto) begin() const { return m_Datas.begin(); }
        decltype(auto) end() const { return m_Datas.end(); }

      private:
        std::array<T, N> m_Datas{};
    };

} // namespace numPDE

