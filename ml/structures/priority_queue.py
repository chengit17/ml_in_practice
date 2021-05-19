import heapq


class PriorityQueue:
    def __init__(self, capacity=-1, order='max'):
        self._hp = []
        self._index = 0
        self.capacity = capacity
        self.order = order

    def push(self, item, priority):
        if self.capacity >= 0:
            if self.capacity != 0:
                while self.qsize() >= self.capacity:
                    self.pop()
            self._push(item, priority)
        else:
            self._push(item, priority)

    def pop(self):
        if self.qsize() > 0:
            return self._pop()

    def first(self):
        if self.qsize() > 0:
            return self._first()

    def last(self):
        if self.qsize() > 0:
            return self._last()

    def qsize(self):
        return self.__len__()

    def _push(self, item, priority):
        if self.order == 'min':
            heapq.heappush(self._hp, (priority, self._index, item))
            self._index += 1
        elif self.order == 'max':
            heapq.heappush(self._hp, (-priority, self._index, item))
            self._index += 1

    def _pop(self):
        return heapq.heappop(self._hp)[-1]

    def _first(self):
        if self.order == 'max':
            return heapq.nsmallest(1, self._hp)[0][-1]
        elif self.order == 'min':
            return heapq.nlargest(1, self._hp)[0][-1]

    def _last(self):
        if self.order == 'max':
            return heapq.nlargest(1, self._hp)[0][-1]
        elif self.order == 'min':
            return heapq.nsmallest(1, self._hp)[0][-1]

    def __len__(self):
        return len(self._hp)

    def __item__(self, index):
        return self._hp[index][-1]

    def __iter__(self):
        return iter(tup[-1] for tup in heapq.nsmallest(len(self._hp), self._hp))