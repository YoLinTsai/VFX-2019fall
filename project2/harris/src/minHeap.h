#ifndef __MINHEAP_H__
#define __MINHEAP_H__

#include <vector>
#include <iostream>

class PRObject{
public:
    PRObject() {}
    PRObject(int row, int col, int r) : _x(row), _y(col), _r(r) {}
    ~PRObject() {}

    friend bool operator < (const PRObject& p1, const PRObject p2) { return (p1._r < p2._r); }
    friend bool operator > (const PRObject& p1, const PRObject p2) { return (p1._r > p2._r); }
    friend bool operator <= (const PRObject& p1, const PRObject p2) { return (p1._r <= p2._r); }
    friend bool operator >= (const PRObject& p1, const PRObject p2) { return (p1._r >= p2._r); }

    void print() const { std::cout << '(' << _x << ',' << _y << ')' << " -> " << _r << std::endl; }
    void get_coordinates(int& row, int& col) const { row = _x; col = _y; }

private:
    int _x;
    int _y;
    int _r;
};

template <class Data>
class minHeap{
public:
    minHeap() { _data.clear(); _data.resize(1); } // _data[0] is reserved (not used)
    ~minHeap() {}

    size_t size() const { return _data.size()-1; }

    Data top() const { return _data[1]; }

    void pop() {
        if (!this->size()) return;
        _data[1] = _data.back();
        _data.pop_back();
        if (this->size()) this->fix_down(1);
    }

    void insert(Data d) {
        _data.push_back(d);
        this->fix_up(this->size());
    }

    void fix_down(int id) {
        if (id > this->size()/2) return;
        int left_id  = 2*id;
        int right_id = 2*id + 1;
        int target_id;
        if (right_id <= this->size() && left_id <= this->size())
            target_id = (_data[left_id] < _data[right_id] ? left_id : right_id);
        else if (right_id > this->size() && left_id <= this->size())
            target_id = left_id;
        else if (right_id <= this->size() && left_id > this->size())
            target_id = right_id;
        else exit(0);
        if (_data[id] > _data[target_id]) {
            std::swap(_data[id], _data[target_id]);
            return fix_down(target_id);
        }
    }

    void fix_up(int id) {
        if (id < 1) return;
        int parent_id = id / 2;
        if (_data[parent_id] > _data[id]) {
            std::swap(_data[parent_id], _data[id]);
            return fix_up(parent_id);
        }
    }

    void print() const {
        for (int i = 1; i < _data.size(); ++i) {
            std::cout << _data[i] << ' ';
        }
        std::cout << std::endl;
    }

    class iterator
    {
    public:
        iterator(typename std::vector<Data>::iterator t) { _it = t; }
        iterator(const iterator& i) { _it = i._it; }
        ~iterator() {};

        const Data& operator * () const { return *_it; }
        iterator& operator ++ () {  ++_it; return *this; }
        iterator& operator ++ (int) { iterator ret = *this; _it++; return ret; }
        iterator& operator -- () {  --_it; return *this; }
        iterator& operator -- (int) { iterator ret = *this; _it--; return ret; }
        bool operator != (const iterator& i) const { return _it != i._it; }
        bool operator == (const iterator& i) const { return _it == i._it; }
        iterator& operator = (const iterator& i) { _it = i._it; }
    private:
        typename std::vector<Data>::iterator _it;
    };

    iterator begin() { return ++(_data.begin()); }
    iterator end() { return _data.end(); }

private:
    std::vector<Data> _data;
};

#endif /* __MINHEAP_H__ */
