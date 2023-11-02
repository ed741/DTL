import abc
import itertools
from typing import Union

import numpy as np

class Holder:
    def __init__(self):
        self.arrays = []
        self.usage = []
        self.allocations = []
        self.normal_size = 64

    def alloc(self, size: int):
        address = 0
        for i in range(len(self.arrays)):
            array = self.arrays[i]
            used = self.usage[i]
            if array.shape[0] - used > size:
                self.usage[i] = used+size
                address += used
                self.allocations.append((address, address+size))
                return address
            else:
                address += array.shape[0]


        array = np.zeros(tuple([np.max([size, self.normal_size])]))
        self.arrays.append(array)
        self.usage.append(size)
        self.allocations.append((address, address + size))
        return address


    def get(self, address):
        running_address = 0
        for i in range(len(self.arrays)):
            array = self.arrays[i]
            if address - running_address < array.shape[0]:
                return array[address-running_address]
            else:
                running_address += array.shape[0]
        raise IndexError(f"Index address {address} out of Bounds!")

    def set(self, address, value):
        running_address = 0
        for i in range(len(self.arrays)):
            array = self.arrays[i]
            if address - running_address < array.shape[0]:
                array[address-running_address] = value
                return
            else:
                running_address += array.shape[0]
        raise IndexError(f"Index address {address} out of Bounds!")

    def printAll(self):
        address = 0
        allocs = 0
        for a, arr in enumerate(self.arrays):
            print(f"{address} - {address+arr.shape[0]}:")
            m = 4
            for i in range(arr.shape[0]):
                m = max(m, len(str(address + i)), len(str(arr[i])))

            print(end='[')
            for i in range(arr.shape[0]):
                addr = address+i
                while allocs < len(self.allocations) and addr >= self.allocations[allocs][1]:
                    allocs += 1
                end = ' | '
                if allocs+1 < len(self.allocations) and addr == self.allocations[allocs][1]-1 and addr+1 == self.allocations[allocs+1][0]:
                    end = ' ]['
                elif allocs < len(self.allocations) and addr+1 == self.allocations[allocs][1]:
                    end = ' ] '
                elif allocs >= len(self.allocations):
                    end = ' . '

                print(f"{str(addr).rjust(m)}", end=end)
            print()

            print(end='|')
            for i in range(arr.shape[0]):
                if i < self.usage[a]:
                    print(f"{str(arr[i]).rjust(m)}", end='   ')
                elif arr[i] == 0:
                    print(f"{str('_').rjust(m)}", end='   ')
                else:
                    print(f"{str(arr[i]).rjust(m)}*", end='   ')
            print()
            address += arr.shape[0]


class SizeExpr(abc.ABC):
    Static = 1
    CompileTime = 2
    RunTime = 3
    Variable = 4

    @abc.abstractmethod
    def property(self) -> int:
        pass

    @abc.abstractmethod
    def variables(self) -> list[str]:
        pass

    @abc.abstractmethod
    def get(self, map: dict[str, int]):
        pass

    def __mul__(self, other):
        return ProductSize(self, other)
    def __add__(self, other):
        return SumSize(self, other)

class UnknownSize(SizeExpr):

    def property(self) -> int:
        return SizeExpr.Variable

    def variables(self) -> list[str]:
        return []

    def get(self, map: dict[str, int]):
        raise Exception("Unknown Size")


class RuntimeSize(SizeExpr):
    def __init__(self, size: int):
        self.size = size

    def property(self) -> int:
        return SizeExpr.RunTime

    def variables(self) -> list[str]:
        return []

    def get(self, map: dict[str, int]):
        return self.size


class StaticSize(SizeExpr):
    def __init__(self, size: int):
        self.size = size

    def property(self) -> int:
        return SizeExpr.Static

    def variables(self) -> list[str]:
        return []

    def get(self, map: dict[str, int]):
        return self.size


class Float32(StaticSize):
    def __init__(self):
        super().__init__(1)

class Index(StaticSize):
    def __init__(self):
        super().__init__(1)

class NoSize(StaticSize):
    def __init__(self):
        super().__init__(0)

class ProductSize(SizeExpr):
    def __init__(self, lhs: SizeExpr, rhs: SizeExpr):
        self.lhs = lhs
        self.rhs = rhs

    def property(self) -> int:
        return max(self.lhs.property(), self.rhs.property())

    def variables(self) -> list[str]:
        l = self.lhs.variables()
        r = self.rhs.variables()
        return l + [v for v in r if v not in l]

    def get(self, map: dict[str, int]):
        return self.lhs.get(map) * self.rhs.get(map)

class SumSize(SizeExpr):
    def __init__(self, lhs: SizeExpr, rhs: SizeExpr):
        self.lhs = lhs
        self.rhs = rhs

    def property(self) -> int:
        return max(self.lhs.property(), self.rhs.property())

    def variables(self) -> list[str]:
        l = self.lhs.variables()
        r = self.rhs.variables()
        return l + [v for v in r if v not in l]

    def get(self, map: dict[str, int]):
        return self.lhs.get(map) + self.rhs.get(map)

class Extent(SizeExpr):
    pass

class StaticExtent(Extent):
    def __init__(self, extent: int):
        self.extent = extent

    def property(self) -> int:
        return SizeExpr.Static

    def variables(self) -> list[str]:
        return []

    def get(self, map: dict[str, int]):
        return self.extent


class CompileExtent(Extent):
    def __init__(self, name: str):
        self.name = name
        self.const = None

    def property(self) -> int:
        return SizeExpr.CompileTime

    def variables(self) -> list[str]:
        return [self.name]

    def get(self, map: dict[str, int]):
        return self.const.get(map)

    def set_extent(self, extent: Extent):
        self.const = extent

class RuntimeExtent(Extent):
    def __init__(self, name: str):
        self.name = name

    def property(self) -> int:
        return SizeExpr.RunTime

    def variables(self) -> list[str]:
        return [self.name]

    def get(self, map: dict[str, int]):
        return map[self.name]

class VariableExtent(Extent):
    def __init__(self, name: str):
        self.name = name

    def property(self) -> int:
        return SizeExpr.Variable

    def variables(self) -> list[str]:
        return [self.name]

    def get(self, map: dict[str, int]):
        return map[self.name]


class DataHandle(abc.ABC):
    @abc.abstractmethod
    def get(self, indexMap: dict[str, Union[int, str]]):
        pass

    @abc.abstractmethod
    def has(self, indexMap: dict[str, Union[int, str]]) -> bool:
        pass


class DerivedDataHandle(DataHandle):
    def __init__(self, handle: DataHandle, indexMap: dict[str, Union[int, str]]):
        self.handle = handle
        self.indexMap = indexMap

    def get(self, indexMap: dict[str, Union[int, str]]):
        map = self.indexMap.copy()
        map.update(indexMap)
        return self.handle.get(map)

    def has(self, indexMap: dict[str, Union[int, str]]) -> bool:
        map = self.indexMap.copy()
        map.update(indexMap)
        return self.handle.has(map)


class ConstHandle(DataHandle):

    def __init__(self, val) -> None:
        self.val = val
        super().__init__()

    def get(self, indexMap: dict[str, Union[int, str]]):
        return self.val

    def has(self, indexMap: dict[str, Union[int, str]]) -> bool:
        return True


class NumpyDataHandle(DataHandle):

    def __init__(self, array, indices: list[str], mask=None) -> None:
        self.array = array
        self.indices = indices
        self.mask = mask
        super().__init__()

    def get(self, indexMap: dict[str, Union[int, str]]):
        assert set(indexMap.keys()) == set(self.indices)
        val = self.array[*[indexMap[i] for i in self.indices]]
        return None if val == self.mask else val

    def has(self, indexMap: dict[str, Union[int, str]]) -> bool:
        if set(indexMap.keys()) != set(self.indices):
            return False
        val = self.array[*[indexMap[i] for i in self.indices]]
        return val != self.mask


class SparseDataHandle(DataHandle):

    def __init__(self, indices: list[str]) -> None:
        self.map = {}
        self.indices = indices
        super().__init__()

    def get(self, indexMap: dict[str, Union[int, str]]):
        key = tuple([indexMap[i] for i in self.indices])
        return self.map[key]

    def set(self, indexMap: dict[str, Union[int, str]], value):
        assert set(indexMap.keys()) == set(self.indices)
        self.map[tuple([indexMap[i] for i in self.indices])] = value

    def has(self, indexMap: dict[str, Union[int, str]]) -> bool:
        return tuple([indexMap[i] for i in self.indices]) in self.map


class SingletonDataHandle(SparseDataHandle):

    def __init__(self, key_map: dict[str, Union[int,str]], value) -> None:
        indices = list(key_map.keys())
        super().__init__(indices)
        super().set(key_map, value)

    def set(self, indexMap: dict[str, Union[int, str]], value):
        raise NotImplementedError()



class DataTreeNode(abc.ABC):

    def setParent(self, parentNode):
        self.parent = parentNode
        assert self in self.parent.children()

    @abc.abstractmethod
    def children(self) -> list["DataTreeNode"]:
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def baseType(self) -> type:
        pass # the type (union types if needed) of the values stored in the structure

    @abc.abstractmethod
    def indexOrder(self) -> list[tuple[str, Extent, str]]:
        pass # order of storage of indices, tuples are (<index name> , <'+' | '-' for forwards or backwards>)

    @property
    def is_addressing_range(self):
        return self.parent.requires_address_range(self)

    def requires_address_range(self, child_node):
        # assert child_node in self.children()
        # return self.parent.requires_address_range(self)
        pass

    def get_default(self):
        return self.parent.get_default()


class DirectDataTreeNode(DataTreeNode):

    @abc.abstractmethod
    def size(self) -> SizeExpr:
        pass

    @abc.abstractmethod
    def size_overflow(self) -> SizeExpr:
        pass

    def make(self, holder, sizeMap: dict[str, int], data_handle: DataHandle, **kwargs):
        length = (self.size()+self.size_overflow()).get(sizeMap)
        address = holder.alloc(length)
        self.init(holder, address, sizeMap, data_handle, write_overflow=True, **kwargs)
        return address

    @abc.abstractmethod
    def init(self, holder, address, sizeMap: dict[str, int], data_handle: DataHandle, write_overflow=False, **kwargs):
        pass

    @abc.abstractmethod
    def get(self, holder: Holder, address: int, index_map: dict, size_map: dict[str, int], include_next: bool = False,
            null_value=None):
        pass

    @abc.abstractmethod
    def set(self, holder: Holder, address: int, indexMap: dict, sizeMap: dict[str, int], value):
        pass


class IndexedDataTreeNode(DataTreeNode):

    @abc.abstractmethod
    def generate(self, holder: Holder, address: int, sizeMap: dict[str, int], data_handle: DataHandle, index_order: list[tuple[str, Extent, str]], **kwargs):
        pass

    @property
    @abc.abstractmethod
    def requires_range(self) -> bool:
        pass

    @abc.abstractmethod
    def base_size(self) -> SizeExpr:
        pass

    @abc.abstractmethod
    def indexed_get(self, holder: Holder, address: int, index: tuple, indexMap: dict, sizeMap: dict[str, int],
                    include_next: bool = False, null_value=None):
        pass

    @abc.abstractmethod
    def indexed_set(self, holder: Holder, address: int, index: tuple, indexMap: dict, sizeMap: dict[str, int], value):
        pass


class DTRoot(DirectDataTreeNode):

    name="root"

    def __init__(self, child: DirectDataTreeNode, default=0):
        self.child = child
        self.child.setParent(self)
        self.default = default
        super().__init__()

    def children(self) -> list[DirectDataTreeNode]:
        return [self.child]

    def size(self) -> SizeExpr:
        return self.child.size() + self.child.size_overflow()

    def size_overflow(self) -> SizeExpr:
        return NoSize()

    def init(self, holder, address, sizeMap: dict[str, int], data_handle: DataHandle, write_overflow=False, **kwargs):
        return self.child.init(holder, address, sizeMap, data_handle, write_overflow=write_overflow, **kwargs)

    def get(self, holder: Holder, address: int, index_map: dict, size_map: dict[str, int], include_next: bool = False,
            null_value=None):
        return self.child.get(holder, address, index_map, size_map, include_next, null_value)

    def set(self, holder: Holder, address: int, indexMap: dict, sizeMap: dict[str, int], value):
        return self.child.set(holder, address, indexMap, sizeMap, value)

    def baseType(self) -> type:
        return self.child.baseType()

    def indexOrder(self) -> list[tuple[str, Extent, str]]:
        return self.child.indexOrder()

    def setParent(self, parentNode):
        assert False

    def requires_address_range(self, child_node):
        return False

    def get_default(self):
        return self.default
class DataTreeNodeTerminal(DirectDataTreeNode):
    def children(self) -> list[DirectDataTreeNode]:
        return []

    def requires_address_range(self, child_node):
        return False

class DTFloat32(DataTreeNodeTerminal):
    name = "float32"

    def size(self) -> SizeExpr:
        return Float32()

    def size_overflow(self) -> SizeExpr:
        return NoSize()

    def init(self, holder, address, sizeMap: dict[str, int], data_handle: DataHandle, write_overflow=False, **kwargs):
        assert not self.is_addressing_range
        if data_handle.has({}):
            value = data_handle.get({})
            holder.set(address, value)
        else:
            holder.set(address,self.get_default())

    def get(self, holder: Holder, address: int, index_map: dict, size_map: dict[str, int], include_next: bool = False,
            null_value=None):
        assert not include_next
        val = holder.get(address)
        return val

    def set(self, holder: Holder, address: int, indexMap: dict, sizeMap: dict[str, int], value):
        return holder.set(address, value)

    def baseType(self) -> type:
        return Float32

    def indexOrder(self) -> list[tuple[str, Extent, str]]:
        return []

class DTIndex(DataTreeNodeTerminal):
    name = "index"

    def size(self) -> SizeExpr:
        return Index()

    def size_overflow(self) -> SizeExpr:
        return Index() if self.is_addressing_range else NoSize()

    def init(self, holder, address, sizeMap: dict[str, int], data_handle: DataHandle, write_overflow=False, **kwargs):
        value = data_handle.get({})
        if write_overflow:
            assert self.is_addressing_range
            start, end = value
            holder.set(address, start)
            holder.set(address + (Index().get(sizeMap)), end)
        else:
            if self.is_addressing_range:
                value, end = value
            holder.set(address, value)

    def get(self, holder: Holder, address: int, index_map: dict, size_map: dict[str, int], include_next=False,
            null_value=None):
        val = holder.get(address)
        if not include_next:
            return val
        else:
            size = Index().get(size_map)
            next = holder.get(address + size)
            return val, next

    def set(self, holder: Holder, address: int, indexMap: dict, sizeMap: dict[str, int], value):
        return holder.set(address, value)

    def baseType(self) -> type:
        return Index

    def indexOrder(self) -> list[tuple[str, Extent, str]]:
        return []


class DTDense(DirectDataTreeNode):
    name = "dense"

    def __init__(self, index: str, extent: Extent, child: DirectDataTreeNode):
        self.child = child
        self.index = index
        self.extent = extent
        child.setParent(self)
        assert self.child.size().property() <= SizeExpr.RunTime

    def children(self) -> list["DataTreeNode"]:
        return [self.child]

    def size(self) -> SizeExpr:
        return self.extent*self.child.size()

    def size_overflow(self) -> SizeExpr:
        return self.child.size_overflow()

    def init(self, holder, address, sizeMap: dict[str, int], data_handle: DataHandle, write_overflow=False, **kwargs):
        addr = address
        addr_step = self.child.size().get(sizeMap)
        extent = self.extent.get(sizeMap)
        for idx in range(extent):
            data = DerivedDataHandle(data_handle, {self.index: idx})
            self.child.init(holder, addr, sizeMap, data, write_overflow=write_overflow and idx==(extent-1), **kwargs)
            addr += addr_step


    def get(self, holder: Holder, address: int, index_map: dict, size_map: dict[str, int], include_next: bool = False,
            null_value=None):
        assert index_map[self.index]<self.extent.get(size_map)
        child_size = self.child.size().get(size_map)
        addr = address + (index_map[self.index]*child_size)
        return self.child.get(holder, addr, index_map, size_map, include_next=include_next, null_value=null_value)

    def set(self, holder: Holder, address: int, indexMap: dict, sizeMap: dict[str, int], value):
        return self.child.set(holder, address + (indexMap[self.index]*self.child.size().get(sizeMap)), indexMap, sizeMap, value)

    def baseType(self) -> type:
        return self.child.baseType()

    def indexOrder(self) -> list[tuple[str, Extent, str]]:
        return [(self.index, self.extent, "+")]+self.child.indexOrder()

    def requires_address_range(self, child_node):
        assert child_node in self.children()
        return self.is_addressing_range

class DTIndexingNode(DirectDataTreeNode):
    name = "indexing"

    def __init__(self, index_source: DirectDataTreeNode, data: IndexedDataTreeNode):
        self.index_source = index_source
        self.data = data
        assert self.index_source.baseType() == Index
        self.index_source.setParent(self)
        self.data.setParent(self)

    def children(self) -> list[DataTreeNode]:
        return [self.index_source, self.data]

    def size(self) -> SizeExpr:
        return self.data.base_size()+self.index_source.size()+self.index_source.size_overflow()

    def size_overflow(self) -> SizeExpr:
        return NoSize()

    def init(self, holder, address, sizeMap: dict[str, int], data_handle: DataHandle, write_overflow=False, **kwargs):
        index_order = self.index_source.indexOrder()
        dataAddresses = self.data.generate(holder, address, sizeMap, data_handle, index_order, **kwargs)

        index_source_address = address+(self.data.base_size().get(sizeMap))
        self.index_source.init(holder, index_source_address, sizeMap, dataAddresses, write_overflow=True, **kwargs)

    def get(self, holder: Holder, address: int, index_map: dict, size_map: dict[str, int], include_next=False,
            null_value=None):
        index_source_size = self.index_source.size().get(size_map)
        index_source_address = address+(self.data.base_size().get(size_map))
        index_range = self.index_source.get(holder, index_source_address, index_map, size_map, include_next=True)
        index_start, index_end = index_range
        index = int(index_start), int(index_end) # cast to int as these are addresses not regular 'data'
        return self.data.indexed_get(holder, address, index, index_map, size_map, include_next=include_next, null_value=null_value)

        # size = self.size().get(sizeMap)
        # index_source_size = self.index_source.size().get(sizeMap)
        # addr = address + (indexMap[self.index]*index_source_size)
        # if addr < address_end:
        #     return self.child.get(holder, addr, min(addr+size, address_end), indexMap, sizeMap, includeNext = includeNext)
        # else:
        #     return None

    def set(self, holder: Holder, address: int, indexMap: dict, sizeMap: dict[str, int], value):
        index_source_size = self.index_source.size().get(sizeMap)
        index_source_address = address + (self.data.base_size().get(sizeMap))
        index_start, index_end = self.index_source.get(holder, index_source_address, indexMap, sizeMap,
                                                       include_next=True)
        index = int(index_start), int(index_end)  # cast to int as these are addresses not regular 'data'
        return self.data.indexed_set(holder, address, index, indexMap, sizeMap, value)

    def baseType(self) -> type:
        return self.data.baseType()

    def indexOrder(self) -> list[tuple[str, Extent, str]]:
        return self.index_source.indexOrder()+self.data.indexOrder()

    def requires_address_range(self, child_node):
        if child_node is self.data:
            return self.is_addressing_range
        else:
            assert child_node is self.index_source
            return self.data.requires_range


class DTUpCOO(IndexedDataTreeNode):

    name = "UpCOO"

    requires_range = True

    def __init__(self, indices: list[str], extents: list[Extent], child: DirectDataTreeNode):
        self.child = child
        self.indices = indices
        self.extents = extents
        child.setParent(self)
        super().__init__()

    def children(self) -> list[DirectDataTreeNode]:
        return [self.child]

    def base_size(self) -> SizeExpr:
        return Index()*StaticSize(2)

    def generate(self, holder, address: int, sizeMap: dict[str, int], data_handle: DataHandle, index_order: list[tuple[str, Extent, str]], **kwargs):
        non_zeros = 0
        outer_spaces = []
        inner_spaces = []
        child_spaces = []
        outer_space_indices = []
        inner_space_indices = []
        child_space_indices = []
        for outer_idx, outer_ext, outer_dir in index_order:
            if outer_dir == "+":
                outer_spaces.append(range(outer_ext.get(sizeMap)))
            elif outer_ext == "-":
                outer_spaces.append(range(outer_ext.get(sizeMap), 0, -1))
            else:
                raise NotImplementedError()
            outer_space_indices.append(outer_idx)

        for inner_idx, inner_ext in zip(self.indices, self.extents):
            inner_spaces.append(range(inner_ext.get(sizeMap)))
            inner_space_indices.append(inner_idx)
            assert inner_idx not in outer_space_indices

        for child_idx, child_ext, child_dir in self.child.indexOrder():
            if child_dir == "+":
                child_spaces.append(range(child_ext.get(sizeMap)))
            elif child_ext == "-":
                child_spaces.append(range(child_ext.get(sizeMap), 0, -1))
            else:
                raise NotImplementedError()
            child_space_indices.append(child_idx)


        for outer_idx in itertools.product(*(outer_spaces)):
            outer_index_map = {i: v for i, v in zip(outer_space_indices, outer_idx)}
            for inner_idx in itertools.product(*(inner_spaces)):
                inner_index_map = outer_index_map | {i: v for i, v in zip(inner_space_indices, inner_idx)}
                has_non_zero = False
                for child_idx in itertools.product(*(child_spaces)):
                    index_map = inner_index_map | {i: v for i, v in zip(child_space_indices, child_idx)}
                    print(index_map)
                    if data_handle.has(index_map):
                        has_non_zero = True
                        break
                if has_non_zero:
                    non_zeros +=1
                    print(f"{inner_index_map} +")

        print(f"non-zeros: {non_zeros}")
        idx_size = Index()*StaticSize(len(self.indices))*RuntimeSize(non_zeros)
        data_overflow = self.child.size_overflow()
        has_overflow = data_overflow.get(sizeMap) > 0
        data_size = self.child.size()*RuntimeSize(non_zeros)+data_overflow
        print(f"idx_size: {idx_size.get(sizeMap)}")
        print(f"data_size: {data_size.get(sizeMap)}")
        index_size = Index().get(sizeMap)

        idx_base_address = holder.alloc(idx_size.get(sizeMap))
        holder.set(address, idx_base_address)
        print(f"idx_base address: {idx_base_address}")
        index_address_step = Index()*StaticSize(len(self.indices)).get(sizeMap)

        data_base_address = holder.alloc(data_size.get(sizeMap))
        holder.set(address+index_size, data_base_address)
        print(f"data_base address: {data_base_address}")
        data_address_step = self.child.size().get(sizeMap)

        idx_address = idx_base_address
        data_address = data_base_address
        current_index = 0

        last_data_range = 0

        index_data_handle = SparseDataHandle(outer_space_indices)

        for outer_idx in itertools.product(*(outer_spaces)):
            outer_index_map = {i: v for i, v in zip(outer_space_indices, outer_idx)}
            range_start = current_index
            print(f"idxmap: {outer_index_map} | start: {range_start}")
            for inner_idx in itertools.product(*(inner_spaces)):
                inner_index_map = outer_index_map | {i: v for i, v in zip(inner_space_indices, inner_idx)}
                inner_non_zero = False
                data_values = []
                data_value_handle = SparseDataHandle(child_space_indices)
                for child_idx in itertools.product(*(child_spaces)):
                    child_index_map = {i: v for i, v in zip(child_space_indices, child_idx)}
                    index_map = inner_index_map | child_index_map
                    if data_handle.has(index_map):
                        inner_non_zero = True
                        data_value = data_handle.get(index_map)
                        data_values.append(data_value)
                        if self.is_addressing_range:
                            data_value_range_start, data_value_range_end = data_value
                            assert data_value_range_start == last_data_range
                            last_data_range = data_value_range_end
                        data_value_handle.set(child_index_map, data_value)
                        # data_value_handle = SingletonDataHandle(child_index_map, data_value)
                        # self.child.init(holder, data_address, sizeMap, data_value_handle)
                        # self.child.set(holder, data_address, child_index_map, sizeMap, data_value)
                if inner_non_zero:
                    self.child.init(holder, data_address, sizeMap, data_value_handle)
                    print(f"index Map:{inner_index_map} | data Values: {data_values} | idx_address: {idx_address} | data:address {data_address} ")
                    for index in self.indices:
                        val = inner_index_map[index]
                        holder.set(idx_address, val)
                        idx_address += index_size
                    data_address += data_address_step
                    current_index += 1
            if range_start != current_index:
                print(f"idxmap: {outer_index_map} | [{range_start}, {current_index})")
                index_data_handle.set(outer_index_map, (range_start, current_index))
            else:
                print(f"idxmap: {outer_index_map} | [{range_start}, {current_index}) | no content")

        if self.is_addressing_range and has_overflow:
            holder.set(data_address, last_data_range)

        print("generate hello2")
        return index_data_handle

    def indexed_get(self, holder: Holder, address: int, index: tuple, indexMap: dict, sizeMap: dict[str, int],
                    include_next=False, null_value=None):
        index_size = Index().get(sizeMap)

        idx_start, idx_end = index
        idx_address = int(holder.get(address))
        data_address = int(holder.get(address + index_size))
        current_address = idx_address + idx_start
        n_indices = len(self.indices)
        idx_step = index_size*n_indices
        data_step = self.child.size().get(sizeMap)
        while current_address < idx_address + idx_end:
            index_values = [int(holder.get(c)) for c in range(current_address, current_address+(index_size*n_indices), index_size)]
            match = True
            for index, value in zip(self.indices, index_values):
                print(f"index: {index} | value: {value} | index_map: {indexMap}")
                if value != indexMap[index]:
                    match = False
                    break
            if match:
                return self.child.get(holder, data_address, indexMap, sizeMap, include_next, null_value=null_value)

            current_address += idx_step
            data_address += data_step
        return null_value


    def indexed_set(self, holder: Holder, address: int, index: tuple, indexMap: dict, sizeMap: dict[str, int], set_value):
        index_size = Index().get(sizeMap)

        idx_start, idx_end = index
        idx_address = int(holder.get(address))
        data_address = int(holder.get(address + index_size))

        n_indices = len(self.indices)
        idx_step = index_size * n_indices
        data_step = self.child.size().get(sizeMap)

        current_address = idx_address + (idx_start * idx_step)
        current_data_address = data_address + (idx_start * data_step)

        while current_address < idx_address + idx_end:
            index_values = [int(holder.get(c)) for c in
                            range(current_address, current_address + (index_size * n_indices), index_size)]
            match = True
            for index, value in zip(self.indices, index_values):
                if value != indexMap[index]:
                    match = False
                    break
            if match:
                return self.child.set(holder, current_data_address, indexMap, sizeMap, set_value)

            current_address += idx_step
            current_data_address += data_step
        assert False, "Cannot change the sparsity of COO after generation."

    def baseType(self) -> type:
        return self.child.baseType()

    def indexOrder(self) -> list[tuple[str, Extent, str]]:
        return [(i, e, "+") for i, e in zip(self.indices, self.extents)]+self.child.indexOrder()

    def requires_address_range(self, child_node):
        assert child_node in self.children()
        return self.is_addressing_range




# tree = DTRoot(DTDense("A", StaticExtent(5), DTDense("B", StaticExtent(6), DTFloat32())))
# tree = DTRoot(DTIndexingNode(DTDense("A", StaticExtent(5), DTIndex()), DTUpCOO(["B"], [StaticExtent(6)], DTFloat32())))
# tree = DTDense("B", StaticExtent(12), DTDense("A", StaticExtent(10), DTFloat32()))
# tree2 = DTDense("B", StaticExtent(10), DTDense("A", StaticExtent(12), DTFloat32()))
# tree = DTRoot(DTIndexingNode(DTIndex(), DTUpCOO(['A'], [StaticExtent(5)], DTFloat32())))
# tree = DTRoot(DTIndexingNode(DTIndexingNode(DTIndex(), DTUpCOO(["A"], [RuntimeExtent("a")], DTIndex())), DTUpCOO(["B"], [StaticExtent(4)], DTFloat32())))

st2 = StaticExtent(2)
st3 = StaticExtent(3)
f = DTFloat32()
B = DTDense("B", st2, f)
tree = DTRoot(B)

tree = DTRoot(DTDense("I", st2, DTDense("II", st2, DTDense("J", st3, DTDense("JJ", st2, DTFloat32())))))
# tree = DTRoot(DTIndexingNode(DTDense("I", st2, DTIndex()), DTUpCOO(["J"], [st3], DTDense("II", st2, DTDense("JJ", st2, DTFloat32())))))
tree = DTRoot(DTIndexingNode(DTIndexingNode(DTDense("I", st2, DTIndex()), DTUpCOO(["J"], [st3], DTDense("II", st2, DTIndex())) ), DTUpCOO(["JJ"], [st2], DTFloat32())))

size_map = {"a":3}

holder = Holder()
nparray = np.array([[1,2,0,0,4,0],
                    [0,3,0,0,0,5],
                    [0,0,6,7,0,0],
                    [0,0,8,0,0,0]]).reshape((2,2,3,2))
dhandle = NumpyDataHandle(nparray, ["I", "II", "J", "JJ"], mask=0)
address = tree.make(holder, size_map, dhandle)
print(address)
holder.printAll()



# tree.set(holder, 0, {"A":1, "B":0}, size_map, 1000)
# holder.printAll()
# tree.set(holder, 0, {"A":1, "B":2}, size_map, 1001)
# holder.printAll()
# tree.set(holder, 0, {"A":2, "B":0}, size_map, 1002)
# holder.printAll()
# v = tree.get(holder, 0, {"A": 1, "B": 2}, size_map)
#
# holder.printAll()
#
# print(v)