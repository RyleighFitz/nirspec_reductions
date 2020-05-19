from collections import defaultdict
import numpy as np
from typing import Dict, Any, List, Union, Callable, Set, Iterable, Iterator, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Image:
    """ Representation of an image, along with metadata. """

    data: np.ndarray
    header: Dict[str, Any]

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape

    def metadata(self, *keys: str) -> Tuple[Any]:
        return tuple(self.header.get(key, None) for key in keys)


class ImageSet:
    """ A queryable set of images. """

    def __init__(self, *imagesets: Union[Image, Iterable[Image]]):
        self.images = []
        for ims in imagesets:
            if isinstance(ims, Image):
                self.images.append(ims)
            else:
                for im in ims:
                    self.images.append(im)

    def __iter__(self) -> Iterator[Image]:
        return iter(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def query(self, **kwargs) -> 'ImageSet':
        """ Get a subset of images with metadata satisfying certain conditions. """
        def matches(image: Image) -> bool:
            return image.metadata(*kwargs.keys()) == tuple(kwargs.values())
        return ImageSet([img for img in self if matches(img)])

    def __repr__(self):
        return f"<ImageSet of size {len(self)}>"


@dataclass
class Transform:
    """ A transformation mapping an ImageSet to an ImageSet. """
    transform_op: Callable[..., Union[Image, Iterable[Image]]]
    partition_keys: Optional[List[str]] = None

    def __call__(self, images: ImageSet) -> ImageSet:
        def partition_value(image: Image):
            if self.partition_keys is not None:
                return image.metadata(*self.partition_keys)
            else:
                return id(image)

        partition = defaultdict(list)
        for image in images:
            partition[partition_value(image)].append(image)
            
        transformed = [img for _images in partition.values()
                           for img in list(self.transform_op(*_images))]
        
        result = ImageSet(transformed)

        return result
