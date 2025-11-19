from typing import Type, TypeVar

from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

T = TypeVar("T")

FieldDescriptor = FieldDescriptor

class PuzzleState(Xtructurable):
    @property
    def packed(self, **kwargs) -> "PuzzleState":
        """
        This function should return a bit packed array that represents
        """
        return self

    @property
    def unpacked(self, **kwargs) -> "PuzzleState":
        """
        This function should return a Xtructurable object that represents the state.
        raw state is bit packed, so it is space efficient, but it is not good for observation & state transition.
        """
        return self


def state_dataclass(cls: Type[T]) -> Type[T]:
    """
    This decorator should be used to define a dataclass that represents the state.
    """

    cls = xtructure_dataclass(cls)

    if not hasattr(cls, "packed") and not hasattr(cls, "unpacked"):
        # if packing and unpacking are not implemented, return the state as is
        def packed(self) -> cls:
            return self

        setattr(cls, "packed", property(packed))

        def unpacked(self) -> cls:
            return self

        setattr(cls, "unpacked", property(unpacked))

    elif hasattr(cls, "packed") ^ hasattr(cls, "unpacked"):
        # packing and unpacking must be implemented together
        raise ValueError("State class must implement both packing and unpacking or neither")
    else:
        # packing and unpacking are implemented
        pass

    return cls
